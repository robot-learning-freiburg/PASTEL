from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ToTensor:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["rgb"] = TF.to_tensor(data["rgb"])
        if "semantic" in data:
            data["semantic"] = data["semantic"].astype(np.int32)
            data["semantic"] = TF.to_tensor(data["semantic"])
        if "instance" in data:
            data["instance"] = data["instance"].astype(np.int32)
            data["instance"] = TF.to_tensor(data["instance"])
        if "boundary" in data:
            data["boundary"] = data["boundary"].astype(np.int32)
            data["boundary"] = TF.to_tensor(data["boundary"])
        if "center" in data:
            data["center"] = TF.to_tensor(data["center"])
        return data


class Resize:

    def __init__(self, size: List[int]):
        self.size = size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["rgb_original"] = data["rgb"]
        data["rgb"] = TF.resize(data["rgb"], self.size, interpolation=InterpolationMode.BILINEAR)
        if "semantic" in data:
            assert data["rgb_original"].shape[1:] == data["semantic"].shape[1:]
            data["semantic"] = TF.resize(data["semantic"], self.size,
                                         interpolation=InterpolationMode.NEAREST)
        if "instance" in data:
            assert data["rgb_original"].shape[1:] == data["instance"].shape[1:]
            data["instance"] = TF.resize(data["instance"], self.size,
                                         interpolation=InterpolationMode.NEAREST)
        if "boundary" in data:
            assert data["rgb_original"].shape[1:] == data["boundary"].shape[1:]
            data["boundary"] = TF.resize(data["boundary"], self.size,
                                         interpolation=InterpolationMode.NEAREST)
        if "center" in data:
            assert data["rgb_original"].shape[1:] == data["center"].shape[1:]
            data["center"] = TF.resize(data["center"], self.size,
                                       interpolation=InterpolationMode.NEAREST)
        return data


class RandomResizedCrop:

    def __init__(self, size: List[int], scale: Tuple[float, float]):
        self.size = size
        self.scale = scale

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "semantic" in data:
            assert data["rgb"].shape[1:] == data["semantic"].shape[1:]
        if "instance" in data:
            assert data["rgb"].shape[1:] == data["instance"].shape[1:]
        if "boundary" in data:
            assert data["rgb"].shape[1:] == data["boundary"].shape[1:]
        if "center" in data:
            assert data["center"].shape[1:] == data["rgb"].shape[1:]
        height, width = data["rgb"].shape[1:]

        rand_scale = np.random.uniform(self.scale[0], self.scale[1])
        crop_height = int(rand_scale * height)  # (H, W)
        crop_width = int(rand_scale * width)  # (H, W)
        crop_top = np.random.randint(0, height - crop_height + 1)
        crop_left = np.random.randint(0, width - crop_width + 1)

        data["rgb_original"] = data["rgb"]
        data["rgb"] = TF.resized_crop(data["rgb"], crop_top, crop_left, crop_height, crop_width,
                                      self.size,
                                      interpolation=InterpolationMode.BILINEAR)
        if "semantic" in data:
            data["semantic"] = TF.resized_crop(data["semantic"], crop_top, crop_left, crop_height,
                                               crop_width, self.size,
                                               interpolation=InterpolationMode.NEAREST)
        if "instance" in data:
            data["instance"] = TF.resized_crop(data["instance"], crop_top, crop_left, crop_height,
                                               crop_width, self.size,
                                               interpolation=InterpolationMode.NEAREST)
        if "boundary" in data:
            data["boundary"] = TF.resized_crop(data["boundary"], crop_top, crop_left, crop_height,
                                               crop_width, self.size,
                                               interpolation=InterpolationMode.NEAREST)
        if "center" in data:
            data["center"] = TF.resized_crop(data["center"], crop_top, crop_left, crop_height,
                                             crop_width, self.size,
                                             interpolation=InterpolationMode.BILINEAR)
        return data


class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["rgb"] = TF.normalize(data["rgb"], self.mean, self.std)
        data["rgb_mean"] = self.mean
        data["rgb_std"] = self.std
        return data


class RandomAffine:
    def __init__(self, degrees: Tuple[float, float], translate: Tuple[float, float],
                 scale: Tuple[float, float],
                 shear: Tuple[float, float, float, float], ignore_index: int = 255):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.ignore_index = ignore_index

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        assert data["rgb"].shape[1:] == data["semantic"].shape[1:] == data["instance"].shape[1:]
        img_width, img_height = data["rgb"].shape[1:]

        degree = np.random.uniform(self.degrees[0], self.degrees[1])
        translate = [
            int(np.random.uniform(-img_width * self.translate[0], img_width * self.translate[0])),
            int(np.random.uniform(-img_height * self.translate[1], img_height * self.translate[1]))]
        scale = np.random.uniform(self.scale[0], self.scale[1])
        shear = [np.random.uniform(self.shear[0], self.shear[1]),
                 np.random.uniform(self.shear[2], self.shear[3])]

        data["rgb"] = TF.affine(data["rgb"], degree, translate, scale, shear,
                                interpolation=InterpolationMode.BILINEAR)
        if "semantic" in data:
            data["semantic"] = TF.affine(data["semantic"], degree, translate, scale, shear,
                                         InterpolationMode.NEAREST, fill=self.ignore_index)
        if "instance" in data:
            data["instance"] = TF.affine(data["instance"], degree, translate, scale, shear,
                                         interpolation=InterpolationMode.NEAREST)
        if "boundary" in data:
            data["boundary"] = TF.affine(data["boundary"], degree, translate, scale, shear,
                                         InterpolationMode.NEAREST, fill=self.ignore_index)
        if "center" in data:
            data["center"] = TF.affine(data["center"], degree, translate, scale, shear,
                                       interpolation=InterpolationMode.BILINEAR)
        return data


class RandomHorizontalFlip:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip = np.random.choice([True, False])
        if flip:
            data["rgb"] = TF.hflip(data["rgb"])
            if "semantic" in data:
                data["semantic"] = TF.hflip(data["semantic"])
            if "instance" in data:
                data["instance"] = TF.hflip(data["instance"])
            if "boundary" in data:
                data["boundary"] = TF.hflip(data["boundary"])
            if "center" in data:
                data["center"] = TF.hflip(data["center"])
        return data


class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["rgb"] = self.color_jitter(data["rgb"])
        return data


class GaussianBlur:
    def __init__(self, kernel_size, sigma):
        self.gaussian_blur = T.GaussianBlur(kernel_size, sigma)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["rgb"] = self.gaussian_blur(data["rgb"])
        return data


class MaskPostProcess:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "semantic" in data:
            data["semantic"] = data["semantic"][0]  # (1, H, W) -> (H, W)
        if "instance" in data:
            data["instance"] = data["instance"][0]  # (1, H, W) -> (H, W)
        if "boundary" in data:
            data["boundary"] = data["boundary"][0]  # (1, H, W) -> (H, W)
        if "center" in data:
            data["center"] = data["center"][0]  # (1, H, W) -> (H, W)
        # if "other_sample" in data:
        #     del data["other_sample"]
        return data


class CopyPaste:

    def __init__(self, scale: Tuple[float, float], min_size: int,
                 initialize_semantic: bool = False, ignore_index: int = 255):
        self.scale = scale
        self.min_size = min_size
        self.initialize_semantic = initialize_semantic
        self.ignore_index = ignore_index

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        assert data["other_sample"]["rgb"].shape[0] == 1
        assert data["rgb"].shape[0] == 1

        source_rgb = data["rgb"][0]  # (3, H, W)
        source_semantic = data["semantic"][0]  # (H, W)
        source_instance = data["instance"][0]  # (H, W)
        target_rgb = torch.clone(data["other_sample"]["rgb"][0].detach())  # (3, H, W)
        if self.initialize_semantic:
            target_semantic = torch.clone(data["other_sample"]["semantic"][0].detach())  # (H, W)
        else:
            target_semantic = self.ignore_index * torch.ones_like(source_semantic)  # (H, W)

        source_instance_ids, instance_size = torch.unique(source_instance, return_counts=True)
        source_instance_ids = source_instance_ids[
            instance_size > self.min_size]  # remove small instances before sampling
        source_instance_ids = source_instance_ids[source_instance_ids != 0]  # remove background

        if len(source_instance_ids) > 0:
            instances_to_copy = np.random.choice(source_instance_ids.cpu(),
                                                 np.random.randint(1, len(source_instance_ids) + 1),
                                                 replace=False)
        else:
            instances_to_copy = []

        for i_instance_id in instances_to_copy:
            i_mask = source_instance == i_instance_id  # (H, W)

            # Calculate instance position and size
            i_pos = torch.nonzero(i_mask, as_tuple=False).min(dim=0)[0]  # (2,) Tensor
            i_top, i_left = i_pos[0], i_pos[1]  # (top, left)
            i_height, i_width = torch.nonzero(i_mask, as_tuple=False).max(dim=0)[
                                    0] - i_pos + 1  # (height, width)

            # Crop instance out
            i_rgb_crop = TF.crop(source_rgb, i_top, i_left, i_height, i_width)
            i_instance_crop = TF.crop(source_instance, i_top, i_left, i_height, i_width)

            # Scale instance randomly
            scale = np.random.uniform(self.scale[0], self.scale[1])
            i_height = int(i_height * scale)
            i_width = int(i_width * scale)
            i_rgb_crop = TF.resize(i_rgb_crop, [i_height, i_width])
            i_instance_crop = TF.resize(i_instance_crop.unsqueeze(0), [i_height, i_width],
                                        interpolation=InterpolationMode.NEAREST).squeeze(0)

            # Paste instance
            i_paste_top = np.random.randint(0, target_rgb.shape[1] - i_height)
            i_paste_left = np.random.randint(0, target_rgb.shape[2] - i_width)

            i_mask_crop = i_instance_crop == i_instance_id
            if i_mask_crop.sum() < self.min_size:
                continue
            i_mask_paste = torch.zeros(target_semantic.shape).bool()  # (H, W)
            i_mask_paste[i_paste_top:i_paste_top + i_height,
            i_paste_left:i_paste_left + i_width] = i_mask_crop

            target_rgb[i_mask_paste.repeat(3, 1, 1)] = i_rgb_crop[i_mask_crop.repeat(3, 1, 1)]

            i_semantic_id = i_instance_id // 1000

            # semantic_mask = semantic == i_semantic_id
            # num_instances_in_semantic = torch.unique(data["instance"][semantic_mask]).shape[0]
            # i_instance_id_new = i_semantic_id * 1000 + num_instances_in_semantic

            target_semantic[i_mask_paste] = i_semantic_id
            # data["instance"][i_mask_paste] = i_instance_id_new

        data["rgb_mix"] = target_rgb.unsqueeze(0)
        data["semantic_mix"] = target_semantic.unsqueeze(0)

        return data
