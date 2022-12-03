# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import math
import warnings
from itertools import repeat

import torch
import torch.nn.functional as F
from torch import nn

from .ptq import QAct, QConv2d, QLinear

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

class Mlp_woq(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Linear(in_features,
                           hidden_features,
                           bias=True,)
        self.act = act_layer()
        # self.qact1 = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A,
        #                   observer_str=cfg.OBSERVER_A,
        #                   quantizer_str=cfg.QUANTIZER_A)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = nn.Linear(hidden_features,
                            out_features,
                            bias=True)
        # self.qact2 = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A,
        #                   observer_str=cfg.OBSERVER_A,
        #                   quantizer_str=cfg.QUANTIZER_A)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.qact1(x)
        x= self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed_woq(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans,
                            embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size,
                            bias=True)
        if norm_layer:
            # self.qact_before_norm = QAct(
            #     quant=quant,
            #     calibrate=calibrate,
            #     bit_type=cfg.BIT_TYPE_A,
            #     calibration_mode=cfg.CALIBRATION_MODE_A,
            #     observer_str=cfg.OBSERVER_A,
            #     quantizer_str=cfg.QUANTIZER_A)
            self.norm = norm_layer(embed_dim)
            # self.qact = QAct(quant=quant,
            #                  calibrate=calibrate,
            #                  bit_type=cfg.BIT_TYPE_A,
            #                  calibration_mode=cfg.CALIBRATION_MODE_A,
            #                  observer_str=cfg.OBSERVER_A,
            #                  quantizer_str=cfg.QUANTIZER_A)
        else:
            # self.qact_before_norm = nn.Identity()
            self.norm = nn.Identity()
            # self.qact = QAct(quant=quant,
            #                  calibrate=calibrate,
            #                  bit_type=cfg.BIT_TYPE_A,
            #                  calibration_mode=cfg.CALIBRATION_MODE_A,
            #                  observer_str=cfg.OBSERVER_A,
            #                  quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        if isinstance(self.norm, nn.Identity):
            x = self.norm(x)
        else:
            x = self.norm(x, self.qact_before_norm.quantizer,
                          self.qact.quantizer)
        # x = self.qact(x)
        return x


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[
                -1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
