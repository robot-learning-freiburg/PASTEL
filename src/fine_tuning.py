from math import ceil
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)
from torch import nn


class FineTuner(pl.LightningModule):
    def __init__(self, vit_model: str, num_classes: int, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None,
                 test_multi_scales: Optional[List[int]] = None,
                 test_multi_scales_stride_divider: List[int] = None,
                 window_block_indexes: List[int] = (), window_size: int = 0):
        super().__init__()
        self.vit_model = vit_model
        self.num_classes = num_classes
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.test_multi_scales = test_multi_scales
        self.test_multi_scales_stride_divider = test_multi_scales_stride_divider

        if vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True, window_block_indexes=window_block_indexes,
                                         window_size=window_size)
            self.pretraining = "dinov2"
        elif vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True, window_block_indexes=window_block_indexes,
                                         window_size=window_size)
            self.pretraining = "dinov2"
        elif vit_model == 'vitl14':
            self.encoder = dinov2_vitl14(pretrained=True, window_block_indexes=window_block_indexes,
                                         window_size=window_size)
            self.pretraining = "dinov2"
        elif vit_model == 'vitg14':
            self.encoder = dinov2_vitg14(pretrained=True, window_block_indexes=window_block_indexes,
                                         window_size=window_size)
            self.pretraining = "dinov2"
        else:
            raise ValueError(f'Unknown model {vit_model}')

        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size
        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise
        for param in self.encoder.parameters():  # freeze backbone
            param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder_dinov1(self, img: torch.Tensor, feature_key: str = 'x'):
        assert feature_key in ['x']  # Currently only support features at the last layer

        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        x = self.encoder.forward_features(img)
        x = x[:, 1:, :]  # (B, Patches, feat_dim)
        x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
        x = x.reshape((x.shape[0], self.feat_dim, patches_h, patches_w))  # (B, feat_dim, H, W)
        if self.upsample_factor is not None:
            x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                          align_corners=False)  # (B, feat_dim, H, W)
        return x

    def forward_encoder_dinov2(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
        with torch.no_grad():
            block_outputs = self.encoder.forward_features(
                img,
                return_attention_features=return_attention_features,
                return_blocks=self.blocks)
            if self.blocks is None:
                block_outputs = [block_outputs]
            outs = []
            for x in block_outputs:
                x = x[feature_key]
                if feature_key == 'attn':
                    return x  # (B, num_heads, Patches+1, Patches+1)
                if feature_key in ['q', 'k', 'v']:
                    # (B, Patches+1, num_heads, feat_dim // num_heads)
                    x = x.permute((0, 2, 1, 3)).contiguous()
                    x = x.reshape((x.shape[0], -1, self.feat_dim))  # (B, Patches+1, feat_dim)
                outs.append(x)
            x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * self.num_blocks)

            x = x[:, 1:, :]  # (B, Patches, feat_dim)
            x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
            x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                           patches_w))  # (B, feat_dim, H, W)
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                              align_corners=False)  # (B, feat_dim, H, W)
        return x

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        if self.pretraining == "dinov2":
            return self.forward_encoder_dinov2(img, feature_key)
        elif self.pretraining == "dinov1":
            return self.forward_encoder_dinov1(img, feature_key)

    def multi_scale_test_augmentation(self, rgb: torch.Tensor, apply_softmax=False,
                                      boundary_margin: Optional[int] = None) -> torch.Tensor:
        # Splitting and upscaling at multiple scales
        all_preds = []  # List to store predictions at all scales to fuse later
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]

        for scale, stride_divider in zip(self.test_multi_scales, self.test_multi_scales_stride_divider):
            image_h_split, image_w_split = ceil(rgb_h / scale), ceil(rgb_w / scale)
            stride_h, stride_w = ceil(image_h_split / stride_divider), ceil(image_w_split / stride_divider)

            end_h = (rgb_h // stride_h) * stride_h + image_h_split
            new_h = end_h - ((end_h - rgb_h) // stride_h) * stride_h
            padding_h = ceil((new_h - rgb_h) / 2)  # divide by 2 to pad equally on both sides

            end_w = (rgb_w // stride_w) * stride_w + image_w_split
            new_w = end_w - ((end_w - rgb_w) // stride_w) * stride_w
            padding_w = ceil((new_w - rgb_w) / 2)  # divide by 2 to pad equally on both sides

            patches = F.unfold(rgb,
                               kernel_size=(image_h_split, image_w_split),
                               stride=(stride_h, stride_w),
                               padding=(padding_h, padding_w))  # (B, C * image_h_split * image_w_split, num_patches)
            patches_shape = list(patches.shape)
            patches_shape[1] = self.num_classes * image_h_split * image_w_split
            pred_patches = torch.zeros(patches_shape, device=rgb.device)

            for patch_i in range(patches.shape[-1]):
                patch = patches[:, :, patch_i]  # (B, C * image_h_split * image_w_split)
                patch = patch.reshape(batch_size, -1, image_h_split, image_w_split)  # (B, C, H, W)
                patch_upscaled = nn.functional.interpolate(patch,
                                                           size=[rgb_h, rgb_w], mode='bilinear',
                                                           align_corners=False)
                pred_i = self(patch_upscaled)  # (B, num_classes, H, W)
                if apply_softmax:
                    pred_i = torch.softmax(pred_i, dim=1)  # (B, num_classes, H, W)
                if boundary_margin is not None:  # Note: only works for boundary estimation
                    pred_i[:, :, :boundary_margin, :] = 1
                    pred_i[:, :, -boundary_margin:, :] = 1
                    pred_i[:, :, :, :boundary_margin] = 1
                    pred_i[:, :, :, -boundary_margin:] = 1
                pred_i = nn.functional.interpolate(pred_i, size=[image_h_split, image_w_split],
                                                   mode='area')  # (B, num_classes, image_h_split, image_w_split)
                pred_patches[:, :, patch_i] = pred_i.reshape(batch_size, -1)

            pred_scale = F.fold(pred_patches,
                                output_size=(rgb_h, rgb_w),
                                kernel_size=(image_h_split, image_w_split),
                                stride=(stride_h, stride_w),
                                padding=(padding_h, padding_w))

            ones = torch.ones((batch_size, 1, rgb_h, rgb_w), device=rgb.device)
            ones_patches = F.unfold(ones,
                                    kernel_size=(image_h_split, image_w_split),
                                    stride=(stride_h, stride_w),
                                    padding=(padding_h, padding_w))
            ones_scale = F.fold(ones_patches,
                                output_size=(rgb_h, rgb_w),
                                kernel_size=(image_h_split, image_w_split),
                                stride=(stride_h, stride_w),
                                padding=(padding_h, padding_w))

            pred_scale = pred_scale / ones_scale

            all_preds.append(pred_scale)

        # Concatenate and fuse scales
        pred = torch.stack(all_preds, dim=1)  # (B, S, num_classes, H, W)
        pred = pred.mean(dim=1)  # (B, num_classes, H, W)
        return pred
