from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN

from .dataset import Dataset


class PhenoBench(Dataset):
    CLASS_COLOR = np.zeros((256, 3), dtype=np.uint8)
    CLASS_COLOR[:5, :] = np.array([
        [0, 0, 0],  # soil
        [0, 255, 0],  # crop
        [255, 0, 0],  # weed
        [0, 255, 255],  # partial_crop
        [255, 0, 255],  # partial_weed
    ])

    def __init__(
            self,
            mode: str,
            cfg: CN,
            transform: List[Callable],
            label_mode: str,
            return_only_rgb, bool=False,
            subset: List[int] = None,
            load_test_image: bool = False,
    ):
        super().__init__("phenobench", ["train", "val", "test"], mode, cfg,
                         transform, return_only_rgb, label_mode)

        if self.return_only_rgb:
            self.frame_paths = self._get_frames_only_rgb()
        else:
            self.frame_paths = self._get_frames()

        if subset is not None:
            self.frame_paths = [self.frame_paths[i] for i in subset]
        self.subset = subset

        # ToDo: Temporary hack to load unlabeled images
        self.load_test_image = load_test_image
        if self.mode == "train" and load_test_image:
            assert False
        else:
            self.test_frame_paths = []

    def _get_frames(self) -> List[Dict[str, Path]]:
        rgb_files = sorted(list((self.path_base / self.mode / "images").glob("*.png")))
        frames = []
        for rgb in tqdm(rgb_files, desc=f"Collect PhenoBench frames [{self.mode}]"):
            semantic = self.path_base / self.mode / "semantics" / rgb.name
            instance = self.path_base / self.mode / "plant_instances" / rgb.name
            leaf_instance = self.path_base / self.mode / "leaf_instances" / rgb.name
            frames.append({
                "rgb": rgb,
                "semantic": semantic,
                "instance": instance,
                "leaf_instance": leaf_instance,
            })
            for path in frames[-1].values():
                if path is not None:
                    assert path.exists(), f"File does not exist: {path}"
        return frames

    def _get_frames_only_rgb(self) -> List[Dict[str, Path]]:
        rgb_files = sorted(list((self.path_base / self.mode / "images").glob("*.png")))
        frames = []
        for rgb in tqdm(rgb_files, desc=f"Collect PhenoBench frames [{self.mode}]"):
            frames.append({
                "rgb": rgb,
            })
            for path in frames[-1].values():
                if path is not None:
                    assert path.exists(), f"File does not exist: {path}"
        return frames

    def __getitem__(
            self,
            index: int,
            do_transform=True,
            return_only_rgb: bool = False,
            add_random_other_sample: bool = True,
            load_test_image: bool = False,
    ) -> Dict[str, Any]:
        if self.load_test_image and load_test_image:
            frame_path = self.test_frame_paths[index]
        else:
            frame_path = self.frame_paths[index]

        # Read image
        image_path = frame_path["rgb"]
        image = Image.open(image_path).convert("RGB")
        # image_size = image.size
        image = self.resize(image)
        # height, width = self.image_size

        output = {
            "rgb": image,
            "rgb_path": str(image_path),
            "index": index,
        }

        if not (self.return_only_rgb or return_only_rgb):
            # Read semantic map
            semantic_path = frame_path["semantic"]
            semantic = np.array(Image.open(semantic_path))
            semantic = cv2.resize(semantic,
                                  list(reversed(self.image_size)),
                                  interpolation=cv2.INTER_NEAREST)

            # Remap partial_crop to crop and partial_weed to weed
            semantic[semantic == 3] = 1
            semantic[semantic == 4] = 2

            # Read plant instance map
            instance_path = frame_path["instance"]
            instance = np.array(Image.open(instance_path))
            instance = cv2.resize(instance,
                                  list(reversed(self.image_size)),
                                  interpolation=cv2.INTER_NEAREST)

            # Remap the instance IDs to follow the Cityscapes convention
            # 1) Remap the IDs to consecutive IDs
            def replace(array: np.array, values, replacements):
                temp_array = array.copy()
                for v, r in zip(values, replacements):
                    temp_array[array == v] = r
                return temp_array

            crop_ids = np.unique(instance[semantic == 1])
            weed_ids = np.unique(instance[semantic == 2])
            instance[semantic == 1] = replace(instance[semantic == 1], crop_ids,
                                              np.arange(len(crop_ids)))
            instance[semantic == 2] = replace(instance[semantic == 2], weed_ids,
                                              np.arange(len(weed_ids)))
            # 2) Now, convert to Cityscapes format
            instance = semantic * 1000 + instance

            # Read leaf instance map
            leaf_instance_path = frame_path["leaf_instance"]
            leaf_instance = np.array(Image.open(leaf_instance_path))
            leaf_instance = cv2.resize(leaf_instance,
                                       list(reversed(self.image_size)),
                                       interpolation=cv2.INTER_NEAREST)

            # Remap the leaf instance IDs to follow the Cityscapes convention
            # 1) Remap the IDs to consecutive IDs
            def replace(array: np.array, values, replacements):
                temp_array = array.copy()
                for v, r in zip(values, replacements):
                    temp_array[array == v] = r
                return temp_array

            leaf_ids = np.unique(leaf_instance[leaf_instance > 0])
            leaf_instance[leaf_instance > 0] = replace(leaf_instance[leaf_instance > 0], leaf_ids,
                                                       np.arange(len(leaf_ids)))
            # 2) Now, convert to Cityscapes format
            semantic[semantic == 2] = 0  # Ignore weed
            leaf_instance = semantic * 1000 + leaf_instance

            output.update({
                "semantic": semantic,
                # "instance": instance
                "instance": leaf_instance
            })

            # For saving predictions in the Cityscapes format
            semantic_path = frame_path["semantic"]
            # instance_path = frame_path["instance"]
            instance_path = frame_path["leaf_instance"]
            output.update({
                "semantic_path": str(semantic_path),
                "instance_path": str(instance_path),
            })
        else:
            # For saving predictions in the Cityscapes format
            semantic_path = str(frame_path["rgb"]).replace("images", "semantics")
            instance_path = str(frame_path["rgb"]).replace("images", "leaf_instances")
            output.update({
                "semantic_path": str(semantic_path),
                "instance_path": str(instance_path),
            })

        if add_random_other_sample and len(self.test_frame_paths) > 0:
            random_other_sample = self.__getitem__(np.random.randint(0, len(self.test_frame_paths)),
                                                   add_random_other_sample=False,
                                                   do_transform=False,
                                                   return_only_rgb=False,
                                                   load_test_image=True)
            random_other_sample = self.test_transform(random_other_sample)
            output["other_sample"] = random_other_sample

        if do_transform:
            output = self.transform(output)

        return output

    def class_id_to_color(self):
        if self.label_mode == "phenobench":
            return self.CLASS_COLOR
        raise NotImplementedError(f"Unsupported label mode: {self.label_mode}")

    def compute_panoptic_label_in_gt_format(self, pred_semantic: np.array,
                                            pred_instance: np.array) -> Tuple[np.array, np.array]:
        semantic = pred_semantic.astype(np.uint8)
        panoptic = pred_instance.astype(np.uint16)

        return semantic, panoptic


class PhenoBenchDataModule(pl.LightningDataModule):

    def __init__(self, cfg_dataset: Dict[str, Any], num_classes: int, batch_size: int,
                 num_workers: int,
                 transform_train: List[Callable], transform_test: List[Callable], label_mode: str,
                 train_sample_indices: Optional[List[int]] = None,
                 test_sample_indices: Optional[List[int]] = None,
                 train_load_test_image: bool = False,
                 train_set: str = "train",
                 test_set: str = "val"):
        super().__init__()
        self.cfg_dataset = CN(init_dict=cfg_dataset)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.label_mode = label_mode
        self.train_sample_indices = train_sample_indices
        self.test_sample_indices = test_sample_indices
        self.train_load_test_image = train_load_test_image
        self.train_set = train_set
        self.test_set = test_set

        self.phenobench_train: Optional[PhenoBench] = None
        self.phenobench_test: Optional[PhenoBench] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.phenobench_train = PhenoBench(self.train_set, self.cfg_dataset,
                                               transform=self.transform_train,
                                               return_only_rgb=False,
                                               label_mode=self.label_mode,
                                               subset=self.train_sample_indices,
                                               load_test_image=self.train_load_test_image)
            assert self.phenobench_train.num_classes == self.num_classes
        if stage == "validate" or stage is None:
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.phenobench_test = PhenoBench(self.test_set, self.cfg_dataset,
                                              transform=self.transform_test,
                                              return_only_rgb=True,
                                              label_mode=self.label_mode,
                                              subset=self.test_sample_indices)
            assert self.phenobench_test.num_classes == self.num_classes

        if stage == "predict" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.phenobench_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.phenobench_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=False, drop_last=False)
