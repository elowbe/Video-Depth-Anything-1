# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from tqdm import tqdm
import numpy as np

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize
from .inference_utils import configure_inference, inference_context

from utils.util import compute_scale_and_shift

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        metric=False,
        decoder_batch_size=4,
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.metric = metric
        self.decoder_batch_size = decoder_batch_size

    def forward(self, x):
        features = self.forward_features(x)
        return self.forward_depth(features, x.shape)

    def forward_features(self, x):
        return self.pretrained.get_intermediate_layers(
            x.flatten(0, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )

    def forward_depth(self, features, input_shape):
        B, T, _, H, W = input_shape
        patch_h, patch_w = H // 14, W // 14
        depth = self.head(
            features,
            patch_h,
            patch_w,
            T,
            micro_batch_size=self.decoder_batch_size,
        )[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))

    @staticmethod
    def _preprocess_frames(frames, resize):
        """Resize and normalize a window into one contiguous NCHW tensor."""
        width, height = resize.get_size(frames[0].shape[1], frames[0].shape[0])
        result = np.empty((len(frames), 3, height, width), dtype=np.float32)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        inv_std = 1.0 / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

        for index, frame in enumerate(frames):
            # Resizing uint8 first avoids a large full-resolution FP32
            # allocation for every frame. Quantization error is below 1/255.
            image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            image = image.astype(np.float32)
            image *= 1.0 / 255.0
            image -= mean
            image *= inv_std
            result[index] = image.transpose(2, 0, 1)

        return torch.from_numpy(result).unsqueeze(0)

    @staticmethod
    def _merge_features(cached_features, new_features):
        if cached_features is None:
            return new_features
        return tuple(
            (
                torch.cat((cached_tokens, new_tokens), dim=0),
                torch.cat((cached_cls, new_cls), dim=0),
            )
            for (cached_tokens, cached_cls), (new_tokens, new_cls)
            in zip(cached_features, new_features)
        )

    @staticmethod
    def _resize_depths(depths, height, width):
        if depths.shape[-2:] == (height, width):
            return depths
        resized = np.empty((len(depths), height, width), dtype=np.float32)
        # Chunked CPU interpolation preserves the original align_corners=True
        # output convention without returning the whole low-resolution result
        # to the accelerator.
        for start in range(0, len(depths), 16):
            end = min(start + 16, len(depths))
            chunk = torch.from_numpy(depths[start:end]).unsqueeze(1)
            chunk = F.interpolate(
                chunk,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
            resized[start:end] = chunk[:, 0].numpy()
        return resized

    def infer_video_depth(
        self,
        frames,
        target_fps,
        input_size=518,
        device='cuda',
        fp32=False,
        cache_encoder=True,
    ):
        device = torch.device(device)
        parameter = next(self.parameters())
        expected_dtype = torch.float16 if device.type == "mps" and not fp32 else torch.float32
        if parameter.device != device or (
            device.type == "mps" and parameter.dtype != expected_dtype
        ):
            configure_inference(self, device, fp32=fp32)

        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        resize = Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        )

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list_aligned = None
        aligned_count = 0
        ref_align = None
        num_windows = len(range(0, org_video_len, frame_step))
        aligned_capacity = INFER_LEN + (num_windows - 1) * frame_step
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]
        blend_weights = np.linspace(0.0, 1.0, INTERP_LEN, dtype=np.float32)
        blend_weights = blend_weights[:, None, None]

        cached_features = None
        cached_inputs = None
        model_dtype = next(self.parameters()).dtype
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            start = 0 if frame_id == 0 else OVERLAP
            input_frames = [
                frame_list[frame_id + i]
                for i in range(start, INFER_LEN)
            ]
            cur_input = self._preprocess_frames(input_frames, resize).to(
                device=device,
                dtype=model_dtype,
            )
            if not cache_encoder and cached_inputs is not None:
                cur_input = torch.cat((cached_inputs, cur_input), dim=1)

            inference_mode, autocast = inference_context(device, fp32=fp32)
            with inference_mode, autocast:
                new_features = self.forward_features(cur_input)
                features = self._merge_features(cached_features, new_features)
                _, _, channels, height, width = cur_input.shape
                depth = self.forward_depth(
                    features,
                    (1, INFER_LEN, channels, height, width),
                )

            window_depths = depth[0].cpu().float().numpy()

            # Align each window immediately. The original implementation kept
            # every redundant 32-frame prediction until the end, which can
            # consume several extra GB and trigger memory compression.
            if depth_list_aligned is None:
                depth_list_aligned = np.empty(
                    (aligned_capacity, window_depths.shape[-2], window_depths.shape[-1]),
                    dtype=np.float32,
                )
                depth_list_aligned[:INFER_LEN] = window_depths
                aligned_count = INFER_LEN
                ref_align = window_depths[kf_align_list].copy()
            else:
                if self.metric:
                    scale, shift = 1.0, 0.0
                else:
                    scale, shift = compute_scale_and_shift(
                        window_depths[:align_len],
                        ref_align,
                    )

                post_depths = window_depths[align_len:OVERLAP] * scale + shift
                np.maximum(post_depths, 0, out=post_depths)
                previous = depth_list_aligned[
                    aligned_count - INTERP_LEN:aligned_count
                ]
                previous *= 1.0 - blend_weights
                previous += post_depths * blend_weights

                new_depths = window_depths[OVERLAP:INFER_LEN] * scale + shift
                np.maximum(new_depths, 0, out=new_depths)
                depth_list_aligned[
                    aligned_count:aligned_count + frame_step
                ] = new_depths
                aligned_count += frame_step

                ref_align[1:] = (
                    window_depths[kf_align_list[1:]] * scale + shift
                )
                np.maximum(ref_align[1:], 0, out=ref_align[1:])

            if cache_encoder:
                cached_features = tuple(
                    (tokens[KEYFRAMES], cls_token[KEYFRAMES])
                    for tokens, cls_token in features
                )
            else:
                cached_features = None
                cached_inputs = cur_input[:, KEYFRAMES].detach()

        del frame_list
        del cached_features, cached_inputs, features, new_features, depth, window_depths
        if device.type == "mps":
            torch.mps.empty_cache()

        depths = depth_list_aligned[:org_video_len]
        depths = self._resize_depths(depths, frame_height, frame_width)
        return depths, target_fps
