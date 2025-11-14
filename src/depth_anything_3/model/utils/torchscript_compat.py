# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TorchScript compatibility utilities.

Provides native PyTorch replacements for einops and other operations
that are not TorchScript-compatible.
"""

import torch


@torch.jit.script
def rearrange_5d_to_4d(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange from (b, s, c, h, w) to ((b*s), c, h, w).

    Replaces: einops.rearrange(x, "b s c h w -> (b s) c h w")

    Args:
        x: Input tensor of shape (B, S, C, H, W)

    Returns:
        Tensor of shape (B*S, C, H, W)
    """
    B, S, C, H, W = x.shape
    return x.view(B * S, C, H, W)


@torch.jit.script
def rearrange_4d_to_3d_tokens(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Rearrange from (bs, c, h, w) to (bs, h*w, c).

    Replaces: einops.rearrange for patch embeddings

    Args:
        x: Input tensor of shape (BS, C, H, W)
        patch_size: Size of patches (for reference, not used in this operation)

    Returns:
        Tensor of shape (BS, H*W, C)
    """
    BS, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(BS, H * W, C)


@torch.jit.script
def rearrange_tokens_to_4d(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Rearrange from (bs, n, c) to (bs, c, h, w).

    Replaces: einops.rearrange for unpacking tokens to spatial

    Args:
        x: Input tensor of shape (BS, N, C)
        H: Target height
        W: Target width

    Returns:
        Tensor of shape (BS, C, H, W)
    """
    BS, N, C = x.shape
    return x.permute(0, 2, 1).reshape(BS, C, H, W)


@torch.jit.script
def rearrange_4d_to_5d(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Rearrange from ((b*s), c, h, w) to (b, s, c, h, w).

    Replaces: einops.rearrange(x, "(b s) c h w -> b s c h w", b=B, s=S)

    Args:
        x: Input tensor of shape (B*S, C, H, W)
        B: Batch size
        S: Number of views/frames

    Returns:
        Tensor of shape (B, S, C, H, W)
    """
    BS, C, H, W = x.shape
    return x.view(B, S, C, H, W)


@torch.jit.script
def rearrange_3d_to_4d_batch_view(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Rearrange from ((b*s), n, c) to (b, s, n, c).

    Replaces: einops.rearrange(x, "(b s) n c -> b s n c", b=B, s=S)

    Args:
        x: Input tensor of shape (B*S, N, C)
        B: Batch size
        S: Number of views/frames

    Returns:
        Tensor of shape (B, S, N, C)
    """
    BS, N, C = x.shape
    return x.view(B, S, N, C)


@torch.jit.script
def rearrange_4d_to_3d_flatten_views(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange from (b, s, n, c) to (b, (s*n), c).

    Replaces: einops.rearrange(x, "b s n c -> b (s n) c")

    Args:
        x: Input tensor of shape (B, S, N, C)

    Returns:
        Tensor of shape (B, S*N, C)
    """
    B, S, N, C = x.shape
    return x.reshape(B, S * N, C)


@torch.jit.script
def rearrange_3d_to_4d_unflatten_views(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """
    Rearrange from (b, (s*n), c) to (b, s, n, c).

    Replaces: einops.rearrange(x, "b (s n) c -> b s n c", b=B, s=S)

    Args:
        x: Input tensor of shape (B, S*N, C)
        B: Batch size
        S: Number of views

    Returns:
        Tensor of shape (B, S, N, C)
    """
    B_actual, SN, C = x.shape
    N = SN // S
    return x.reshape(B, S, N, C)


@torch.jit.script
def rearrange_4d_to_3d_batch_view_flatten(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange from (b, s, n, c) to ((b*s), n, c).

    Replaces: einops.rearrange(x, "b s n c -> (b s) n c")

    Args:
        x: Input tensor of shape (B, S, N, C)

    Returns:
        Tensor of shape (B*S, N, C)
    """
    B, S, N, C = x.shape
    return x.reshape(B * S, N, C)


class DictOutput:
    """
    TorchScript-compatible dictionary-like output container.

    Replaces addict.Dict with a simpler structure that TorchScript can handle.
    This is a workaround since TorchScript has limited dict support.
    """

    def __init__(self):
        # Use attributes instead of dict for TorchScript compatibility
        self.depth: torch.Tensor = torch.empty(0)
        self.depth_conf: torch.Tensor = torch.empty(0)
        self.ray: torch.Tensor = torch.empty(0)
        self.ray_conf: torch.Tensor = torch.empty(0)
        self.sky: torch.Tensor = torch.empty(0)
        self.extrinsics: torch.Tensor = torch.empty(0)
        self.intrinsics: torch.Tensor = torch.empty(0)
        self.has_depth: bool = False
        self.has_depth_conf: bool = False
        self.has_ray: bool = False
        self.has_ray_conf: bool = False
        self.has_sky: bool = False
        self.has_extrinsics: bool = False
        self.has_intrinsics: bool = False


def convert_dict_to_output(d: dict) -> DictOutput:
    """
    Convert a standard dictionary to DictOutput.

    Args:
        d: Input dictionary with string keys and tensor values

    Returns:
        DictOutput object with attributes set from dict
    """
    output = DictOutput()

    if 'depth' in d:
        output.depth = d['depth']
        output.has_depth = True
    if 'depth_conf' in d:
        output.depth_conf = d['depth_conf']
        output.has_depth_conf = True
    if 'ray' in d:
        output.ray = d['ray']
        output.has_ray = True
    if 'ray_conf' in d:
        output.ray_conf = d['ray_conf']
        output.has_ray_conf = True
    if 'sky' in d:
        output.sky = d['sky']
        output.has_sky = True
    if 'extrinsics' in d:
        output.extrinsics = d['extrinsics']
        output.has_extrinsics = True
    if 'intrinsics' in d:
        output.intrinsics = d['intrinsics']
        output.has_intrinsics = True

    return output


def convert_output_to_dict(output: DictOutput) -> dict:
    """
    Convert DictOutput back to standard dictionary.

    Args:
        output: DictOutput object

    Returns:
        Standard Python dictionary
    """
    d = {}

    if output.has_depth:
        d['depth'] = output.depth
    if output.has_depth_conf:
        d['depth_conf'] = output.depth_conf
    if output.has_ray:
        d['ray'] = output.ray
    if output.has_ray_conf:
        d['ray_conf'] = output.ray_conf
    if output.has_sky:
        d['sky'] = output.sky
    if output.has_extrinsics:
        d['extrinsics'] = output.extrinsics
    if output.has_intrinsics:
        d['intrinsics'] = output.intrinsics

    return d
