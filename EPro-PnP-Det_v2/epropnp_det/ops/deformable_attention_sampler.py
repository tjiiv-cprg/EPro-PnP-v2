# Modified from https://github.com/tjiiv-cprg/EPro-PnP

"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.cnn.bricks.registry import ATTENTION


@ATTENTION.register_module()
class DeformableAttentionSampler(nn.Module):

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_points=32,
                 stride=4,
                 ffn_cfg=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.1,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(DeformableAttentionSampler, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.stride = stride
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg

        self.sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_points * 2)
        self.out_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.layer_norms = nn.ModuleList(
            [build_norm_layer(norm_cfg, self.embed_dims)[1] for _ in range(2)])
        self.ffn = build_feedforward_network(self.ffn_cfg, dict(type='FFN'))

        self.init_weights()

    def init_weights(self):
        xavier_init(self.sampling_offsets, gain=2.5, distribution='uniform')
        for m in [self.layer_norms, self.ffn]:
            if hasattr(m, 'init_weights'):
                m.init_weights()
        self._is_init = True

    def forward(self, query, obj_emb, key, value, img_shapes, img_transform,
                obj_xy_point, scales, obj_img_ind):
        """
        Args:
            query: shape (num_obj, num_head, 1, head_emb_dim)
            obj_emb: shape (num_obj, embed_dim)
            key: shape (num_img, embed_dim, h, w)
            value: shape (num_img, embed_dim, h, w)
            img_shapes: shape (num_img, 2)
            img_transform: shape (num_img, 3, 3)
            obj_xy_point: shape (num_obj, 2)
            scales: shape (num_obj, )
            obj_img_ind: shape (num_obj, )

        Returns:
            tuple[tensor]:
                output (num_obj_sample, embed_dim)
                v_samples (num_obj_sample, num_head, head_emb_dim, num_point)
                mask_samples (num_obj_sample, num_head, 1, num_point)
                x2d_samples (num_obj_sample, num_head, 2, num_point)
        """
        num_obj_samples = query.size(0)
        num_img, _, h_out, w_out = key.size()
        head_emb_dim = self.embed_dims // self.num_heads

        offsets = self.sampling_offsets(obj_emb).reshape(
            num_obj_samples, self.num_heads, self.num_points, 2)
        # (num_obj_sample, num_head, num_point, 2)
        sampling_location = obj_xy_point[:, None, None] + offsets * scales[:, None, None, None]

        obj_inv_trans = img_transform.inverse()[obj_img_ind]  # (num_obj_sample, 3, 3)
        # (num_obj_sample, num_head, 3, num_point) =
        #     (num_obj_sample, 1, 3, 3) @ (num_obj_sample, num_head, 3, num_point)
        x2d_samples = obj_inv_trans[:, None] @ F.pad(
            sampling_location, [0, 1], mode='constant', value=1.0).transpose(-1, -2)
        # if num_obj_samples > 0:
        #     x2d_samples = torch.linalg.solve(
        #         img_transform[obj_img_ind, None],
        #         F.pad(sampling_location, [0, 1], mode='constant', value=1.0).transpose(-1, -2))
        # else:
        #     x2d_samples = sampling_location.reshape(num_obj_samples, self.num_heads, 3, self.num_points)

        x2d_samples = x2d_samples[..., :2, :] / x2d_samples[..., 2:3, :]

        wh_img = key.new_tensor([w_out, h_out]) * self.stride
        sampling_grid = sampling_location * (2 / wh_img) - 1
        img_ind_grid = (obj_img_ind.to(torch.float32) + 0.5) * (2 / num_img) - 1.0
        img_sampling_grid = torch.cat(
            (sampling_grid.transpose(1, 0).reshape(self.num_heads, num_obj_samples, self.num_points, 1, 2),
             img_ind_grid[None, :, None, None, None].expand(self.num_heads, -1, self.num_points, 1, 1)),
            dim=-1)  # (num_head, num_obj_sample, num_point, 1, 3) in [x, y, img_ind]
        # (num_head, head_emb_dim, num_obj_sample, num_point, 1) ->
        # (num_obj_sample, num_head, head_emb_dim, num_point)
        k_samples = F.grid_sample(
            key.reshape(
                num_img, self.num_heads, head_emb_dim, h_out, w_out
            ).permute(1, 2, 0, 3, 4),  # (num_head, head_emb_dim, num_img, h_out, w_out)
            img_sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).squeeze(-1).permute(2, 0, 1, 3)
        v_samples = F.grid_sample(
            value.reshape(
                num_img, self.num_heads, head_emb_dim, h_out, w_out
            ).permute(1, 2, 0, 3, 4),  # (num_head, head_emb_dim, num_img, h_out, w_out)
            img_sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).squeeze(-1).permute(2, 0, 1, 3)
        half_stride = self.stride / 2
        img_shapes_ = torch.flip(img_shapes, [-1])[obj_img_ind, None, None]
        mask_lt = ((sampling_location - half_stride) / self.stride).clamp(0, 1)
        mask_rb = ((img_shapes_ + half_stride - sampling_location) / self.stride).clamp(0, 1)
        # (num_obj_sample, num_head, 1, num_point)
        mask_samples = torch.minimum(mask_lt, mask_rb).prod(dim=-1).unsqueeze(-2)
        # (num_obj_sample, num_head, 1, num_point) = (num_obj_sample, num_head, 1, head_emb_dim)
        # @ (num_obj_sample, num_head, head_emb_dim, num_point)
        a_samples = query @ k_samples / np.sqrt(head_emb_dim)
        a_samples_softmax = a_samples.softmax(dim=-1) * mask_samples
        # (num_obj_sample, num_head, head_emb_dim, 1)
        # = (num_obj_sample, num_head, head_emb_dim, num_point)
        # @ (num_obj_sample, num_head, num_point, 1)
        output = v_samples @ a_samples_softmax.reshape(num_obj_samples, self.num_heads, self.num_points, 1)
        output = output.reshape(num_obj_samples, self.embed_dims)
        output = self.out_proj(output) + obj_emb
        output = self.layer_norms[0](output)
        output = self.ffn(output, output)
        output = self.layer_norms[1](output)
        return output, v_samples, mask_samples, x2d_samples
