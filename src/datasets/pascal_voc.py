import itertools
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
from numpy.typing import ArrayLike
from PIL import Image, ImageDraw
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torchvision.datasets import SBDataset, VOCSegmentation
from torchvision.datasets.voc import _VOCBase
from tqdm import tqdm
from yacs.config import CfgNode as CN

from .dataset import Dataset


# There is no class for panoptic in the open source implementation. That's why we write our own one
class VOCInstance(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or
            ``"val"``. If ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its
            target as entry and returns a transformed version.
    """

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationObject"
    _TARGET_FILE_EXT = ".png"

    @property
    def masks(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class SBDatasetPanoptic(SBDataset):
    def __init__(
            self,
            root: str,
            image_set: str = "train",
            mode: str = "boundaries",
            download: bool = False,
            transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, image_set, mode, download, transforms)

        sbd_root = self.root
        image_dir = os.path.join(sbd_root, "img")
        inst_dir = os.path.join(sbd_root, "inst")

        if not os.path.isdir(sbd_root):
            raise RuntimeError("Dataset not found or corrupted." +
                               " You can use download=True to download it")

        split_f = os.path.join(sbd_root, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_f), "r") as fh:
            file_names = [x.strip() for x in fh.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.instances = [os.path.join(inst_dir, x + ".mat") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def _get_instance_boundaries(self, filepath: str) -> np.ndarray:
        mat = loadmat(filepath)
        instance_boundaries = mat["GTinst"][0]["Boundaries"][0][:]
        return [np.array(instance_boundaries[i][0].toarray()) for i in
                range(len(instance_boundaries))]


# ToDo: Make it possible to train with PascalVOC official subset without extra annotations
#  from the SB dataset

class PascalVOC(Dataset):
    CLASS_COLOR = np.zeros((256, 3), dtype=np.uint8)
    CLASS_COLOR[:21, :] = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ])

    def __init__(
            self,
            mode: str,
            cfg: CN,
            transform: List[Callable],
            label_mode: str,
            return_only_rgb: bool = False,
            subset: List[int] = None,
            load_test_image: bool = False,
            add_sb: bool = True,
    ):
        super().__init__("pascal_voc", ["train", "val", "trainval"],
                         mode, cfg, transform, return_only_rgb, label_mode)

        self.add_sb = add_sb
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
            path_base = Path(__file__).absolute().parent.parent / "test" / "pascalvoc"
            rgb = sorted([str(x) for x in (path_base / 'Images').glob("*.jpg")])
            instance = sorted([str(x) for x in (path_base / 'SegmentationObject').glob("*.png")])
            semantic = sorted([str(x) for x in (path_base / 'SegmentationClass').glob("*.png")])
            self.test_frame_paths = []
            for a, b, c in zip(rgb, instance, semantic):
                self.test_frame_paths.append({
                    "rgb": a,
                    "instance": b,
                    "semantic": c,
                })
            self.test_transform = self.transform
        else:
            self.test_frame_paths = []

    @staticmethod
    def _sample_name(path: str) -> str:
        return path.split("/")[-1].split(".")[0]

    def _get_frames(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image, annotation, and camera intrinsics files
        Returns
        -------
        frames : list of dictionaries
            List containing the file paths of the RGB image, the semantic and instance annotations,
            and the camera intrinsics
        """
        if self.mode == "trainval":
            self.mode = "train"
            frames = self._get_frames()
            self.mode = "val"
            frames += self._get_frames()
            self.mode = "trainval"
            return frames

        # Get datasets for VOC and SBD from torchvision
        # Please note that the point annotations were provided for the full VOC2012 semantic dataset
        # and the additional images from SBD. For the latter, also semantic masks and boundaries
        # exist - these may have a lower quality though in comparison to the VOC2012 dataset

        # We add up also the validation images from SB and remove the VOC validation data later
        self.dataset_voc_is_train = VOCInstance((self.path_base / "VOC"), image_set="train")
        self.dataset_voc_is_val = VOCInstance((self.path_base / "VOC"), image_set="val")
        self.dataset_voc_ss_train = VOCSegmentation((self.path_base / "VOC"), image_set="train")
        self.dataset_voc_ss_val = VOCSegmentation((self.path_base / "VOC"), image_set="val")
        if self.add_sb:
            self.dataset_sb_train = SBDatasetPanoptic((self.path_base / "SBD"), image_set="train")
            self.dataset_sb_val = SBDatasetPanoptic((self.path_base / "SBD"), image_set="val")

        if self.mode == "train":
            rgb_lst = []
            rgb_lst += self.dataset_voc_ss_train.images
            if self.add_sb:
                rgb_lst += self.dataset_sb_train.images
                rgb_lst += self.dataset_sb_val.images

            semantic_lst = []
            semantic_lst += self.dataset_voc_ss_train.masks
            if self.add_sb:
                semantic_lst += self.dataset_sb_train.masks
                semantic_lst += self.dataset_sb_val.masks

            instance_lst = []
            instance_lst += self.dataset_voc_is_train.masks
            if self.add_sb:
                instance_lst += self.dataset_sb_train.instances
                instance_lst += self.dataset_sb_val.instances

            #rgb_lst = list(itertools.chain(self.dataset_voc_ss_train.images,
            #                               self.dataset_sb_train.images,
            #                               self.dataset_sb_val.images))
            #semantic_lst = list(itertools.chain(self.dataset_voc_ss_train.masks,
            #                                    self.dataset_sb_train.masks,
            #                                    self.dataset_sb_val.masks))
            #instance_lst = list(itertools.chain(self.dataset_voc_is_train.masks,
            #                                    self.dataset_sb_train.instances,
            #                                    self.dataset_sb_val.instances))
        elif self.mode == "val":
            rgb_lst = self.dataset_voc_ss_val.images
            semantic_lst = self.dataset_voc_ss_val.masks
            instance_lst = self.dataset_voc_is_val.masks
        else:
            raise NotImplementedError(f"Unsupported label mode: {self.label_mode}")

        # Convert to dictionary, where filename is the key for easier access later when removing
        # validation files
        rgb_dict = {self._sample_name(path): path for path in rgb_lst}
        semantic_dict = {self._sample_name(path): path for path in semantic_lst}
        instance_dict = {self._sample_name(path): path for path in instance_lst}

        # Get point annotations for the same set of files (defined on 10582 images). This is only
        # relevant in train mode, as evaluation with point annotations is not reasonable
        if self.mode == "train":
            path_points_fg = os.path.join(self.path_base, "voc_whats_the_point.json")
            path_points_bg = os.path.join(self.path_base,
                                          "voc_whats_the_point_bg_from_scribbles.json")
            with open(path_points_fg, "r", encoding="utf-8") as f:
                ds_clicks_fg = dict(sorted(json.load(f).items()))
            with open(path_points_bg, "r", encoding="utf-8") as f:
                ds_clicks_bg = dict(sorted(json.load(f).items()))


        # Remove the validation data from the official PascalVoc challenge benchmark. We should
        # end up with 10582 images and annotations when extra data is used
        if self.mode == "train":
            for path in self.dataset_voc_ss_val.images:
                name = self._sample_name(path)
                # No need to remove fg and bg points here as they are directly accessed via
                # valid filenames lateron
                rgb_dict.pop(name, None)
                semantic_dict.pop(name, None)
                instance_dict.pop(name, None)

        # Sort files to make sure that a single index refers to the files that belong together
        # ToDo: It is not really necessary to convert it to lists here...
        rgb_files = sorted(rgb_dict.values())
        semantic_files = sorted(semantic_dict.values())
        instance_files = sorted(instance_dict.values())

        frames = []
        for idx, elem in enumerate(
                tqdm(rgb_files, desc=f"Collect PascalVOC frames [{self.mode}]")):
            rgb = elem
            semantic = semantic_files[idx]
            instance = instance_files[idx]

            dict_path = {
                "rgb": rgb,
                "semantic": semantic,
                "instance": instance,
            }

            if self.mode == "train":
                points_fg = ds_clicks_fg[self._sample_name(rgb)]
                points_bg = ds_clicks_bg[self._sample_name(rgb)]
                dict_path.update({
                    "points_fg": points_fg,
                    "points_bg": points_bg,
                })

            frames.append(dict_path)

        return frames

    # We do not really need this function but for the sake of completeness we keep it
    def _get_frames_only_rgb(self) -> List[Dict[str, Path]]:

        """Gather the paths of the image files if only the RGB images should be returned
        For instance, when training depth only (unsupervised), we can exploit the full sequences
        instead of only the image tuples where there are semantic annotations for the center image.
        """
        frames = []
        rgb_files = sorted(
            list((self.path_base / "VOC/VOCdevkit/VOC2012/JPEGImages").glob("*.jpg")))
        with tqdm(desc="Collect PascalVOC RGB frames") as pbar:
            for frame in rgb_files:
                frames.append({"rgb": frame})
                pbar.update(1)
        return frames

    # Transform instance boundaries from SBDataset to instances
    def get_instances(self, instance_boundaries: ArrayLike) -> ArrayLike:
        instance = np.zeros_like(instance_boundaries[0])
        for idx, boundary_map in enumerate(instance_boundaries):
            label_count = idx + 1
            instance_mask = np.zeros_like(boundary_map)
            contours, hierarchy = cv.findContours(boundary_map, cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_NONE)
            # ToDo: We may be able to solve this without a loop using the contourIdx parameter
            #  of cv.drawContours
            for j, contour in enumerate(contours):
                instance_mask = cv.drawContours(instance_mask, [contour], 0, label_count, -1)
            instance[instance_mask != 0] = instance_mask[instance_mask != 0]
        return instance

    def rasterize_clicks(self, data: Image, width: int, height: int, stroke_width: int =3) -> Image:
        img = Image.new("L", (width, height), color=255)
        draw = ImageDraw.Draw(img)
        for clsid, click in data:
            click = click[0]
            if stroke_width > 1:
                draw.ellipse((
                    click[0] - stroke_width / 2, click[1] - stroke_width / 2,
                    click[0] + stroke_width / 2, click[1] + stroke_width / 2),
                    clsid
                )
            else:
                draw.point(click, clsid)
        return img

    # ToDo: The instances counter does not start with 0 for each class, but we cannot help
    #  this. It should still be fine for the center/offset computation
    # The instances do not perfectly align with the semantics. That's why we go through each
    # instance and create the corresponding mask with the biggest overlapping semantic area
    #  as otherwise it may overlap with multiple semantics and produce smaller instances
    def fuse_sem_inst(self, semantic: ArrayLike, instance: ArrayLike) -> ArrayLike:
        instance_pascal = np.zeros_like(instance, dtype=np.uint16)

        for instance_id in np.unique(instance):
            if instance_id == 0 or instance_id == 255:
                continue
            msk_pre = np.zeros_like(instance, dtype=np.uint16)
            msk_pre[instance == instance_id] = semantic[instance == instance_id] * 1000 + instance[
                instance == instance_id]
            # Remove background
            msk_valid = msk_pre[msk_pre > 0]
            # Get the blob with the maximum amount of pixels as all other ones result from
            # imperfections in the annotation
            ids, id_counts = np.unique(msk_valid, return_counts=True)
            id_max = np.argmax(id_counts)
            msk_pre[msk_pre != ids[id_max]] = 0
            instance_pascal[msk_pre > 0] = msk_pre[msk_pre > 0]

        return instance_pascal

    def __getitem__(self, index: int,
                    do_transform: bool = True,
                    return_only_rgb: bool = False,
                    add_random_other_sample: bool = True,
                    load_test_image: bool = False,
                    ) -> Dict[str, Any]:
        """Collect all data for a single sample
        Parameters
        ----------
        index : int
            Will return the data sample with this index
        Returns
        -------
        output : dict
            The output contains the following data:
            1) RGB images: center and offset images (3, H, W)
            2) semantic annotations (H, W)
            3) loss weight for semantic prediction defined by instance size
            4) center heatmap of the instances (1, H, W)
            5) (x,y) offsets to the center of the instances (2, H, W)
            6) loss weights for the center heatmap and the (x,y) offsets (H, W)
            7) foreground point annotations
            8) background point annotations
        """
        if self.load_test_image and load_test_image:
            frame_path = self.test_frame_paths[index]
        else:
            frame_path = self.frame_paths[index]

        # Read center and offset images
        image_path = frame_path["rgb"]
        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        height, width = self.image_size

        output = {
            "rgb": self.resize(image),
            "rgb_path": image_path,
            "index": index,
        }

        if not (self.return_only_rgb or return_only_rgb):
            # Read semantic map
            semantic_path = frame_path["semantic"]
            if semantic_path.endswith("mat"):
                # You could also use sb_val, it doesn't matter, should've been a static method
                semantic = np.array(self.dataset_sb_train._get_segmentation_target(semantic_path))
            else:
                semantic = np.array(Image.open(semantic_path))
            semantic = cv.resize(semantic,
                                  list(reversed(self.image_size)),
                                  interpolation=cv.INTER_NEAREST)

            # Read instance and convert to center heatmap and offset map
            instance_path = frame_path["instance"]

            # ToDo: Sometimes some boundaries are kept as such and not transformed to an instance.
            #  Please consider that this is not a bug in the code but it's due to incomplete
            #  instance boundaries in the extra annotations provided by the SB dataset, which have
            #  low quality and are sometimes not annotated properly. They do not always complete the
            #  instance boundaries.
            # Read as grayscale as the values go from 0 to 1
            if instance_path.endswith("mat"):
                instance_boundaries = np.array(
                    self.dataset_sb_train._get_instance_boundaries(instance_path))
                instance = self.get_instances(instance_boundaries)
            else:
                instance = np.array(Image.open(instance_path))
            instance = cv.resize(instance,
                                  list(reversed(self.image_size)),
                                  interpolation=cv.INTER_NEAREST)

            # Compute instance IDs for thing classes in the Cityscapes domain.
            # ToDo: Createthis from instances
            thing_mask = self._make_thing_mask(semantic, as_bool=True)

            # The instance mask is initialized with zeros that do not have further meaning, as such
            # we remove them from the thing mask. ToDo: Can we do this cleaner?
            # ToDo: This will not exactly correspond to the instance parts due to misalignments
            #  between semantics and instances but this doesn't really matter
            thing_mask[instance == 0] = 0
            thing_mask[semantic == 0] = 0
            instance_msk = thing_mask.copy()

            # Fuse semantics map with instance map where we use 1000 as multiplier for semantics
            instance_pascal = self.fuse_sem_inst(semantic, instance)

            # Generate semantic_weights map by instance mask size
            # semantic_weights = np.ones_like(instance_pascal, dtype=np.uint8)
            # semantic_weights[semantic == 255] = 0

            # Set the semantic weights by instance mask size
            # full_res_h, full_res_w = image_size[1], image_size[0]
            # small_instance_area = self.small_instance_area_full_res * (height / full_res_h) * (
            #        width / full_res_w)

            # inst_id, inst_area = np.unique(instance_pascal, return_counts=True)

            # for instance_id, instance_area in zip(inst_id, inst_area):
            #    # Skip stuff pixels
            #    if instance_id == 0:
            #        continue
            #    if instance_area < small_instance_area:
            #        semantic_weights[instance_pascal == instance_id] = self.small_instance_weight

            # Compute center heatmap and (x,y) offsets to the center for each instance
            # offset, center = self.get_offset_center(instance_pascal, self.sigma, self.gaussian)
            # center = np.squeeze(center)

            # Generate pixel-wise loss weights
            # Unlike Panoptic-DeepLab, we do not consider the is_crowd label. Following them, we
            #  ignore stuff in the offset prediction.
            # center_weights = np.ones_like(center, dtype=np.uint8)
            # center_weights[0][semantic == 255] = 0
            # instance_msk_int = instance_msk.astype(np.uint8)
            # offset_weights = np.expand_dims(instance_msk_int, axis=0)

            output.update({
                "semantic": semantic,
                #"semantic_weights": semantic_weights,
                # "center": center,
                #"center_weights": center_weights,
                #"offset": offset,
                #"offset_weights": offset_weights,
                "instance": instance_pascal.astype(np.int32),
            })

            # if self.mode == "train":
            #     # Rasterize point annotations ToDo: Finalize this and put into the output dictionary
            #     clicks_fg = frame_path["points_fg"]
            #     clicks_bg = frame_path["points_bg"]
            #     clicks = clicks_fg + clicks_bg
            #     clicks = [(d["cls"], [(d["x"], d["y"])]) for d in clicks]
            #     clicks = self.rasterize_clicks(clicks, width, height)
            #     point_annotations = np.array(clicks)
            #     point_annotations = cv.resize(point_annotations,
            #                          list(reversed(self.image_size)),
            #                          interpolation=cv.INTER_NEAREST)
            #     output.update({
            #         "point": point_annotations
            #     })

        # For saving predictions in the Cityscapes format
        semantic_path = frame_path["semantic"]
        instance_path = frame_path["instance"]
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
        if self.label_mode == "pascal_voc":
            return self.CLASS_COLOR
        raise NotImplementedError(f"Unsupported label mode: {self.label_mode}")

    def compute_panoptic_label_in_gt_format(self, pred_semantic: np.array,
                                            pred_instance: np.array) -> Tuple[np.array, np.array]:
        semantic = pred_semantic.astype(np.uint8)
        panoptic = pred_instance.astype(np.uint16)

        return semantic, panoptic


class PascalVOCDataModule(pl.LightningDataModule):

    def __init__(self, cfg_dataset: Dict[str, Any], num_classes: int, batch_size: int,
                 num_workers: int,
                 transform_train: List[Callable], transform_test: List[Callable], label_mode: str,
                 train_sample_indices: Optional[List[int]] = None,
                 test_sample_indices: Optional[List[int]] = None,
                 train_load_test_image: bool = False,
                 train_set: str = "train",
                 test_set: str = "val",
                 add_sb: bool = True
                 ):
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
        self.add_sb = add_sb

        self.pascalvoc_train: Optional[PascalVOC] = None
        self.pascalvoc_test: Optional[PascalVOC] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.pascalvoc_train = PascalVOC(self.train_set, self.cfg_dataset,
                                              transform=self.transform_train,
                                              return_only_rgb=False,
                                              label_mode=self.label_mode,
                                              subset=self.train_sample_indices,
                                              load_test_image=self.train_load_test_image,
                                              add_sb=self.add_sb)
            assert self.pascalvoc_train.num_classes == self.num_classes
        if stage == "validate" or stage is None:
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.pascalvoc_test = PascalVOC(self.test_set, self.cfg_dataset,
                                             transform=self.transform_test,
                                             return_only_rgb=False,
                                             label_mode=self.label_mode,
                                             subset=self.test_sample_indices,
                                             add_sb=self.add_sb)
            assert self.pascalvoc_test.num_classes == self.num_classes

        if stage == "predict" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.pascalvoc_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.pascalvoc_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=False, drop_last=False)
