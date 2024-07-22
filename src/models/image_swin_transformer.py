# Removed depth dimension (D) and related operations.
# Adjusted tensor shapes and operations to work with 2D data (H, W) instead of 3D data (D, H, W).
# Changed 3D convolutions to 2D convolutions.
# Updated window partitioning, merging, and attention mechanisms to work with 2D data.
# Adjusted PatchEmbed and PatchMerging layers for 2D data.
# Changed adaptive pooling from 3D to 2D.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange

import logging

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)

class Mlp(nn.Module):
    """Multilayer perceptron (MLP) class for implementing a basic feedforward neural network."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Defines the forward pass of the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Partitions input tensors into smaller windows.
    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size (tuple[int]): Size of the window
    Returns:
        windows: Partitioned tensor of shape (B*num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape  # Removed depth dimension (D)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)  # Adjusted view to 2D
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, window_size), C)  # Adjusted permutation to 2D
    return windows

def window_reverse(windows, window_size, B, H, W):
    """
    Reconstructs the original tensor from partitioned windows.
    Args:
        windows: Partitioned tensor of shape (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of the original tensor
        W (int): Width of the original tensor
    Returns:
        x: Reconstructed tensor of shape (B, H, W, C)
    """
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)  # Adjusted view to 2D
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # Adjusted permutation to 2D
    return x

def get_window_size(x_size, window_size, shift_size=None):
    """
    Adjusts the window size to fit the input dimensions.
    Args:
        x_size (tuple[int]): Size of the input tensor
        window_size (tuple[int]): Desired window size
        shift_size (tuple[int], optional): Shift size for cyclic shift
    Returns:
        tuple[int]: Adjusted window size
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # Adjusted to 2D

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # Removed depth dimension (D)
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # Adjusted meshgrid to 2D
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward pass of the WindowAttention module."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block for 2D data."""

    def __init__(self, dim, num_heads, window_size=(7, 7), shift_size=(0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must be in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        """First part of the forward pass."""
        B, H, W, C = x.shape  # Removed depth dimension (D)
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)  # Adjusted to 2D

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0  # Removed depth dimension (pad_d0)
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # Adjusted padding to 2D
        _, Hp, Wp, _ = x.shape  # Removed depth dimension (Dp)
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))  # Adjusted roll to 2D
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))  # Adjusted view to 2D
        shifted_x = window_reverse(attn_windows, window_size, B, Hp, Wp)  # B H' W' C  # Adjusted to 2D
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))  # Adjusted roll to 2D
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()  # Adjusted slicing to 2D
        return x

    def forward_part2(self, x):
        """Second part of the forward pass."""
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """Full forward pass of the SwinTransformerBlock module."""
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward pass for Patch Merging Layer."""
        B, H, W, C = x.shape  # Removed depth dimension (D)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))  # Adjusted padding to 2D

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C  # Adjusted slicing to 2D
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C  # Adjusted slicing to 2D
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C  # Adjusted slicing to 2D
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C  # Adjusted slicing to 2D
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  # Adjusted concatenation to 2D

        x = self.norm(x)
        x = self.reduction(x)

        return x

@lru_cache()
def compute_mask(H, W, window_size, shift_size, device):
    """Computes the attention mask for shifted window attention."""
    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 Hp Wp 1  # Adjusted shape to 2D
    cnt = 0
    for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            img_mask[:, h, w, :] = cnt  # Adjusted indexing to 2D
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1], 1  # Adjusted to 2D
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class BasicLayer(nn.Module):
    """Basic Swin Transformer layer for one stage."""

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """Forward pass for the BasicLayer."""
        B, H, W, C = x.shape  # Removed depth dimension (D)
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)  # Adjusted to 2D
        x = rearrange(x, 'b h w c -> b h w c')
        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]
        attn_mask = compute_mask(Hp, Wp, window_size, shift_size, x.device)  # Adjusted to 2D
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, H, W, -1)  # Removed depth dimension (D)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b h w c -> b c h w')  # Adjusted rearrange to 2D
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding layer."""

    def __init__(self, patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # Changed to Conv2d
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward pass for the PatchEmbed layer."""
        _, _, H, W = x.size()  # Removed depth dimension (D)
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))  # Adjusted padding to 2D
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))  # Adjusted padding to 2D

        x = self.proj(x)  # B C H W  # Adjusted projection to 2D
        if self.norm is not None:
            H, W = x.size(2), x.size(3)  # Removed depth dimension (D)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, H, W)  # Adjusted view to 2D

        return x

class SwinTransformer(nn.Module):
    """Swin Transformer backbone for 2D data."""

    def __init__(self,
                 num_classes=1000,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4, 4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers - 1))

        # Add a norm layer for each output
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Changed to AdaptiveAvgPool2d
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze certain stages in the model."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin2d."""
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # Delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # Delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # Bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2 * self.window_size[0] - 1, 2 * self.window_size[1] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in the backbone."""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward pass for the SwinTransformer model."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c h w -> n h w c')  # Adjusted rearrange to 2D
        x = self.norm(x)
        x = rearrange(x, 'n h w c -> n c h w')  # Adjusted rearrange to 2D
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        return x
    
    def get_img_feature(self, x):
        """Extracts image features from the input."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c h w -> n h w c')  # Adjusted rearrange to 2D
        x = self.norm(x)
        x = rearrange(x, 'n h w c -> n c h w')  # Adjusted rearrange to 2D
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def train(self, mode=True):
        """Convert the model into training mode while keeping layers frozen."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
