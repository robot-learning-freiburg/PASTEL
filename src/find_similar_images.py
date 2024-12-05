import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import faiss
import numpy as np
import torch
from fine_tuning import FineTuner
from pytorch_lightning.cli import LightningCLI

# Ignore some torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')


class ImageFinder(FineTuner):
    def __init__(self, dinov2_vit_model: str, num_classes: int, num_neighbors: int,
                 upsample_factor: Optional[float] = None):
        super().__init__(dinov2_vit_model=dinov2_vit_model, num_classes=num_classes,
                         blocks=None, upsample_factor=upsample_factor)

        self.num_neighbors = num_neighbors

        self.faiss_index = None
        self.test_step_features = []

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.forward_encoder(img)  # (B, feat_dim, H, W)
        return x

    def training_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']

        features = self.forward(rgb)
        features = features.mean(dim=(2, 3))  # (B, feat_dim)
        features = features.detach().cpu().numpy()

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexIDMap(
                faiss.index_factory(features.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT))
        faiss.normalize_L2(features)
        self.faiss_index.add_with_ids(features, np.array([batch_idx]))

        return None

    def on_train_end(self) -> None:
        print('Save FAISS index')
        save_faiss_index(self.faiss_index, Path(__file__).absolute().parent)

    def on_test_start(self) -> None:
        print('Load FAISS index')
        self.faiss_index = load_faiss_index(Path(__file__).absolute().parent)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']

        features = self.forward(rgb)
        features = features.mean(dim=(2, 3))  # (B, feat_dim)
        features = features.detach().cpu().numpy()

        faiss.normalize_L2(features)

        self.test_step_features.append(features)

    def on_test_end(self) -> None:
        train_subset = self.trainer.test_dataloaders[0].dataset.subset

        similarity, train_batch_id = [], []
        for i, features in enumerate(self.test_step_features, start=1):
            similarity_, train_batch_id_ = self.faiss_index.search(
                features, int(self.num_neighbors * 1.5))
            similarity_ = similarity_.tolist()[0]
            train_batch_id_ = train_batch_id_.tolist()[0]
            for sim, b_id in zip(similarity_, train_batch_id_):
                if b_id not in train_batch_id and b_id not in train_subset:
                    similarity.append(sim)
                    train_batch_id.append(b_id)
                if len(train_batch_id) == self.num_neighbors * i:
                    break
        print(similarity, train_batch_id)
        print(sorted(train_batch_id))


def save_faiss_index(index: faiss.IndexIDMap, path: Path):
    features = []
    for i in range(index.ntotal):
        features.append(index.index.reconstruct(i))
    with open(path / 'faiss_index.pkl', 'wb') as f:
        pickle.dump({'faiss_features': features, }, f)


def load_faiss_index(path: Path) -> faiss.IndexIDMap:
    with open(path / 'faiss_index.pkl', 'rb') as f:
        state = pickle.load(f)
    features = state['faiss_features']
    index = faiss.IndexIDMap(
        faiss.index_factory(features[0].size, 'Flat', faiss.METRIC_INNER_PRODUCT))
    for i, feature in enumerate(features):
        index.add_with_ids(feature.reshape(1, feature.size), np.array([i]))
    return index


class ImageFinderCLI(LightningCLI):

    def __init__(self):
        super().__init__(
            model_class=ImageFinder,
            seed_everything_default=0,
            parser_kwargs={'parser_mode': 'omegaconf'},
            save_config_callback=None,
        )

    def add_arguments_to_parser(self, parser):
        # Dataset
        parser.add_argument('--data_params', type=Dict)


if __name__ == '__main__':
    cli = ImageFinderCLI()

    faiss_index_path = Path(__file__).absolute().parent
    faiss_index = load_faiss_index(faiss_index_path)
