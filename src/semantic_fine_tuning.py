import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from fine_tuning import FineTuner
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from torch import nn
from torch.utils.data import Dataset
from utils.transforms import CopyPaste
#from utils.tree_energy_loss import TreeEnergyLoss

# Ignore some torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')


class SemanticFineTuner(FineTuner):
    """Fine-tunes a small head on top of the DINOv2 model for semantic segmentation.

    Parameters
    ----------
    dinov2_vit_model : str
        ViT model name of DINOv2. One of ['vits14', 'vitl14', 'vitg14', 'vitb14'].
    num_classes : int
        Number of classes for semantic segmentation.
    output_size : Tuple[int, int]
        Output size [H, W] after model.
    blocks : List[int]
        List of block indices of ViT to use for feature extraction. If None, use the last block.
    upsample_factor : float
        Upsample factor of the feature map after the ViT and before the head.
    head : str
        Head to use for semantic segmentation. One of ['linear', 'knn', 'cnn', 'mlp'].
    ignore_index : int
        Index to ignore in the loss.
    top_k_percent_pixels : float
        Percentage of hardest pixels to keep for the loss.
    test_multi_scales : List[int]
        List of scales to use for multi-scale during prediction/testing e.g. [1, 2, 3].
    test_plot : bool
        Whether to plot the predictions during testing.
    test_save_dir : str
        Directory to save the predictions during testing.
    """

    def __init__(self, dinov2_vit_model: str, num_classes: int, output_size: Tuple[int, int],
                 blocks: Optional[List[int]] = None, upsample_factor: Optional[float] = None,
                 head: str = 'mlp',
                 ignore_index: int = -100, top_k_percent_pixels: float = 1.0,
                 test_multi_scales: Optional[List[int]] = None,
                 test_multi_scales_stride_divider: Optional[List[int]] = None,
                 test_plot: bool = False, test_save_dir: Optional[str] = None,
                 test_save_vis: bool = False,
                 model_ckpt: Optional[str] = None):
        super().__init__(vit_model=dinov2_vit_model, num_classes=num_classes,
                         blocks=blocks, upsample_factor=upsample_factor,
                         test_multi_scales=test_multi_scales,
                         test_multi_scales_stride_divider=test_multi_scales_stride_divider)
        self.output_size = output_size
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.test_multi_scales = test_multi_scales
        self.test_plot = test_plot
        self.test_save_dir = test_save_dir
        self.test_save_vis = test_save_vis

        head_input_dim = self.feat_dim * self.num_blocks
        if head == 'linear':
            self.head = nn.Conv2d(head_input_dim, num_classes, kernel_size=1, stride=1, padding=0)
        elif head == 'cnn':
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, 300, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(300, 200, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(200, num_classes, kernel_size=3, stride=1, padding=1),
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, 300, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(300, 300, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(300, 200, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(200, num_classes, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError(f'Unknown head {head}')

        #self.tree_loss = TreeEnergyLoss().to(self.device)

        if model_ckpt is not None:  # TODO: remove
            model_ckpt_dict = torch.load(model_ckpt, map_location=self.device)
            self.load_state_dict(model_ckpt_dict['state_dict'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_encoder(x)  # (B, feat_dim, H, W)
        x = self.head(x)  # (B, num_classes, H, W)
        x = nn.functional.interpolate(x, size=self.output_size, mode='bilinear',
                                      align_corners=False)
        return x

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        rgb = train_batch['rgb']
        sem = train_batch['semantic'].long()

        sem = TF.resize(sem, self.output_size, interpolation=T.InterpolationMode.NEAREST)
        pred = self(rgb)
        seg_loss = F.cross_entropy(pred, sem, ignore_index=self.ignore_index, reduction='none')
        if self.top_k_percent_pixels < 1.0:
            seg_loss = seg_loss.contiguous().view(-1)
            # Hard pixel mining
            top_k_pixels = int(self.top_k_percent_pixels * seg_loss.numel())
            seg_loss, _ = torch.topk(seg_loss, top_k_pixels)
        seg_loss = seg_loss.mean()

        loss = seg_loss

        # ------------------------------------------------------------

        if "other_sample" in train_batch:

            rgb_unlabeled = train_batch['other_sample']['rgb']
            sem_unlabeled = train_batch['other_sample']['semantic'].long()

            x = self(rgb_unlabeled)
            tree_pred = nn.functional.interpolate(x, size=self.output_size, mode='bilinear',
                                                  align_corners=False)

            sem_unlabeled = TF.resize(sem_unlabeled, self.output_size,
                                      interpolation=T.InterpolationMode.NEAREST)
            tree_seg_loss = F.cross_entropy(tree_pred, sem_unlabeled, ignore_index=self.ignore_index,
                                            reduction='mean')

            loss *= .5
            loss += tree_seg_loss

        # ------------------------------------------------------------

        self.log('train_loss', loss)
        return loss

    def predict(self, rgb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.test_multi_scales is None:
            pred = self(rgb)  # (B, num_classes, H, W)
            pred = torch.softmax(pred, dim=1)  # (B, num_classes, H, W)
        else:
            pred = self.multi_scale_test_augmentation(rgb, apply_softmax=True)
            # (B, num_classes, H, W)

        pred = nn.functional.interpolate(pred, size=self.output_size, mode='bilinear',
                                         align_corners=False)
        pred = pred.argmax(dim=1)  # (B, H, W)

        if mask is not None:
            pred[mask] = self.ignore_index
        return pred

    def get_dataset(self) -> Dataset:
        if self.trainer.test_dataloaders is None:
            return self.trainer.train_dataloader.loaders.dataset
        return self.trainer.test_dataloaders[0].dataset

    def plot(self, rgb: np.array, pred: np.array, title: str = None, save_dir=None):
        fig = plt.figure(figsize=(10, 6))
        # plt.figure(figsize=(20, 6))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(10, 10)

        rgb = rgb.transpose((1, 2, 0))  # (H, W, 3)
        dataset = self.get_dataset()
        pred_color = dataset.class_id_to_color()[pred, :]  # (H, W, 3)

        # plt.subplot(1, 2, 1)
        # plt.axis('off')
        # plt.grid(False)
        # plt.imshow(rgb)
        #
        # plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.grid(False)
        if title is not None:
            plt.title(title)
        plt.imshow(rgb)
        plt.imshow(pred_color, cmap='jet', alpha=0.5, interpolation='nearest')
        fig.tight_layout()
        plt.show()

        if save_dir is not None:
            img_sem = Image.fromarray(pred_color.astype(np.uint8))
            img_sem.save(save_dir)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']  # (B, 3, H, W)
        ego_car_mask = batch.get('ego_car_mask', None)  # (B, H, W)

        pred = self.predict(rgb, ego_car_mask)  # (B, H, W)
        pred = pred.cpu().numpy()  # (B, H, W)

        rgb_original = batch['rgb_original']  # (B, 3, H, W)
        # rgb_original = rgb_original.cpu().numpy()  # (B, 3, H, W)

        if self.test_plot:
            for rgb_i, pred_i in zip(rgb_original.cpu().numpy(), pred):
                self.plot(rgb_i, pred_i, str(batch_idx))

        if self.test_save_dir is not None:
            semantic_paths = batch['semantic_path']
            dataset = self.get_dataset()
            dataset_path_base = str(dataset.path_base)
            for pred_i, rgb_i, semantic_path in zip(pred, rgb_original, semantic_paths):
                pred_path = semantic_path.replace(dataset_path_base, self.test_save_dir)
                if not os.path.exists(os.path.dirname(pred_path)):
                    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                pred_img = Image.fromarray(pred_i.astype(np.uint8))
                pred_img.save(pred_path)

                if self.test_save_vis:
                    pred_sem_i_color = self.get_dataset().class_id_to_color()[pred_i,
                                       :]  # (H, W, 3)
                    pred_sem_i_color_path = pred_path.replace('.png', '_color.png')
                    pred_img = Image.fromarray(pred_sem_i_color)
                    rgb_img = T.ToPILImage()(rgb_i)
                    pred_img = Image.blend(rgb_img, pred_img, alpha=0.75)
                    pred_img.save(pred_sem_i_color_path)


class SemanticFineTunerCLI(LightningCLI):

    def __init__(self):
        super().__init__(
            model_class=SemanticFineTuner,
            seed_everything_default=0,
            parser_kwargs={'parser_mode': 'omegaconf'},
            save_config_callback=None,
        )

    def add_arguments_to_parser(self, parser):
        # Dataset
        parser.add_argument('--data_params', type=Dict)


if __name__ == '__main__':
    cli = SemanticFineTunerCLI()
