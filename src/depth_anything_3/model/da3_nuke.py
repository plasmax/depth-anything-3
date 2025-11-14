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
TorchScript-compatible wrapper for Depth Anything v3 for Nuke integration.

This module provides a TorchScript-compatible interface to DA3, with the following modifications:
- Replaces einops with native PyTorch operations
- Removes addict.Dict and OmegaConf dependencies
- Handles Nuke's (1, 6, H, W) input format
- Preserves camera parameter support for multi-view depth estimation
- Uses explicit type annotations for TorchScript compilation
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class DepthAnything3Nuke(nn.Module):
    """
    TorchScript-compatible wrapper for Depth Anything v3.

    Designed for Nuke's Inference node architecture which requires:
    - Input format: (1, 6, H, W) for two RGB images concatenated along channel dim
    - Camera parameters: extrinsics (1, 2, 4, 4) and intrinsics (1, 2, 3, 3)
    - Output: depth map (1, 2, H, W) for two views

    Args:
        backbone: DinoV2 vision transformer encoder
        head: DPT or DualDPT depth prediction head
        cam_enc: Camera encoder (optional)
        cam_dec: Camera decoder (optional)
        patch_size: Size of image patches (default: 14)
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        cam_enc: Optional[nn.Module] = None,
        cam_dec: Optional[nn.Module] = None,
        patch_size: int = 14,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.cam_enc = cam_enc
        self.cam_dec = cam_dec
        self.patch_size = patch_size
        self.has_camera = cam_enc is not None and cam_dec is not None

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for depth prediction.

        Args:
            x: Input images in Nuke format (1, 6, H, W) or standard (B, S, 3, H, W)
            extrinsics: Camera extrinsics (1, 2, 4, 4) or (B, S, 4, 4)
            intrinsics: Camera intrinsics (1, 2, 3, 3) or (B, S, 3, 3)

        Returns:
            depth: Predicted depth maps (1, 2, H, W) or (B, S, H, W)
        """
        # Detect and convert Nuke format (1, 6, H, W) to (1, 2, 3, H, W)
        if x.dim() == 4 and x.shape[1] == 6:
            x = self._nuke_format_to_multiview(x)

        # Ensure we have 5D tensor (B, S, 3, H, W)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, S, 3, H, W), got shape {x.shape}")

        B, S, C, H, W = x.shape

        # Process camera tokens if camera parameters are provided
        cam_token: Optional[torch.Tensor] = None
        if self.has_camera and extrinsics is not None and intrinsics is not None:
            # Ensure extrinsics and intrinsics are properly shaped
            if extrinsics.dim() == 3:
                extrinsics = extrinsics.unsqueeze(0)  # (S, 4, 4) -> (1, S, 4, 4)
            if intrinsics.dim() == 3:
                intrinsics = intrinsics.unsqueeze(0)  # (S, 3, 3) -> (1, S, 3, 3)

            # Use autocast disabled for camera encoding (as in original)
            with torch.cuda.amp.autocast(enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, (H, W))

        # Extract features from backbone
        feats = self._extract_features(x, cam_token)

        # Process through depth head
        with torch.cuda.amp.autocast(enabled=False):
            output = self._process_depth_head(feats, H, W)

            # Process camera estimation if decoder is available
            if self.has_camera and self.cam_dec is not None:
                output = self._process_camera_estimation(feats, H, W, output)

        # Extract depth tensor from output dictionary
        depth = output.get('depth', output.get('main', None))
        if depth is None:
            # Fallback: try to get first value from dict
            depth = list(output.values())[0]

        # Return depth in format matching input (B, S, H, W)
        return depth

    def _nuke_format_to_multiview(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert Nuke's (1, 6, H, W) format to (1, 2, 3, H, W).

        Args:
            x: Input in Nuke format (1, 6, H, W)

        Returns:
            x: Reshaped to (1, 2, 3, H, W)
        """
        B, C6, H, W = x.shape
        # Split 6 channels into 2 views of 3 channels each
        # Channels 0-2: first view RGB, Channels 3-5: second view RGB
        x = x.view(B, 2, 3, H, W)
        return x

    def _extract_features(
        self,
        x: torch.Tensor,
        cam_token: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Extract features from backbone without using get_intermediate_layers.

        Args:
            x: Input images (B, S, 3, H, W)
            cam_token: Optional camera tokens

        Returns:
            List of feature tensors from intermediate layers
        """
        # Call backbone's get_intermediate_layers method
        # This returns: tuple(zip(outputs, camera_tokens)), aux_outputs
        # We need to adapt this to avoid tuple unpacking issues
        intermediate_results = self.backbone.get_intermediate_layers(
            x,
            n=[9, 10, 11, 12],  # Example: last 4 layers, adjust based on model
            export_feat_layers=[],
            cam_token=cam_token,
        )

        # intermediate_results is (features_and_tokens, aux_outputs)
        features_and_tokens = intermediate_results[0]

        # Extract just the features from each tuple
        feats = []
        for item in features_and_tokens:
            # item is (features, camera_token)
            feats.append((item[0], item[1]))

        return feats

    def _process_depth_head(
        self,
        feats: list,
        H: int,
        W: int,
    ) -> dict:
        """
        Process features through depth head.

        Args:
            feats: List of feature tensors
            H: Original image height
            W: Original image width

        Returns:
            Dictionary with depth predictions
        """
        output = self.head(feats, H, W, patch_start_idx=0)

        # Convert addict.Dict to standard dict if needed
        if not isinstance(output, dict):
            # If it's an addict.Dict, convert to standard dict
            output = dict(output)

        return output

    def _process_camera_estimation(
        self,
        feats: list,
        H: int,
        W: int,
        output: dict,
    ) -> dict:
        """
        Process camera pose estimation if camera decoder is available.

        Args:
            feats: List of feature tensors
            H: Image height
            W: Image width
            output: Current output dictionary

        Returns:
            Updated output dictionary with camera parameters
        """
        if self.cam_dec is None:
            return output

        # Get camera tokens from last feature layer
        # feats[-1] is (features, camera_tokens)
        camera_tokens = feats[-1][1]

        # Decode camera pose
        pose_enc = self.cam_dec(camera_tokens)

        # Note: pose_encoding_to_extri_intri conversion would go here
        # For now, we'll skip this as it requires additional utility functions
        # output['extrinsics'] = ...
        # output['intrinsics'] = ...

        return output


def create_da3_nuke_wrapper(
    model_size: str = "small",
    pretrained_path: Optional[str] = None,
    use_camera: bool = True,
) -> DepthAnything3Nuke:
    """
    Factory function to create a TorchScript-compatible DA3 wrapper.

    This is a placeholder that would need to be completed with actual model loading.
    For now, it demonstrates the structure needed.

    Args:
        model_size: Model size variant ("small", "base", "large", "giant")
        pretrained_path: Path to pretrained checkpoint
        use_camera: Whether to include camera encoder/decoder

    Returns:
        DepthAnything3Nuke wrapper instance
    """
    # TODO: Implement actual model loading
    # This would involve:
    # 1. Loading the appropriate backbone configuration
    # 2. Loading the head configuration
    # 3. Optionally loading camera modules
    # 4. Loading pretrained weights

    raise NotImplementedError(
        "Model loading not yet implemented. "
        "Use direct instantiation with pre-loaded modules for now."
    )
