import argparse
import time

import numpy as np
import torch

from utils.dc_utils import read_video_frames
from video_depth_anything.inference_utils import (
    configure_inference,
    get_default_device,
    synchronize,
)
from video_depth_anything.video_depth import VideoDepthAnything


def main():
    parser = argparse.ArgumentParser(
        description="Compare VDA-Small quality and fast presets."
    )
    parser.add_argument("--input-video", default="test_videos/bathhouse.mp4")
    parser.add_argument("--max-len", type=int, default=66)
    parser.add_argument("--max-res", type=int, default=1280)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    args = parser.parse_args()

    device = get_default_device() if args.device == "auto" else torch.device(args.device)
    frames, fps = read_video_frames(
        args.input_video,
        args.max_len,
        target_fps=-1,
        max_res=args.max_res,
    )

    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        decoder_batch_size=8 if device.type == "mps" else 4,
    )
    model.load_state_dict(
        torch.load(
            "checkpoints/video_depth_anything_vits.pth",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    model = configure_inference(model, device)

    def benchmark(input_size, temporal_stride):
        # Warm Metal/CUDA kernels without including compilation in the timing.
        model.infer_video_depth(
            frames[: min(22, len(frames))],
            fps,
            input_size=input_size,
            device=device,
            temporal_stride=temporal_stride,
        )
        synchronize(device)
        start = time.perf_counter()
        depths, _ = model.infer_video_depth(
            frames,
            fps,
            input_size=input_size,
            device=device,
            temporal_stride=temporal_stride,
        )
        synchronize(device)
        return time.perf_counter() - start, depths

    quality_time, quality = benchmark(input_size=518, temporal_stride=1)
    fast_time, fast = benchmark(input_size=420, temporal_stride=2)

    # Relative depth is affine-ambiguous, so align the fast result before
    # reporting its difference from quality mode.
    quality_sample = quality[:, ::4, ::4].astype(np.float64)
    fast_sample = fast[:, ::4, ::4].astype(np.float64)
    design = np.stack(
        (fast_sample.ravel(), np.ones(fast_sample.size)),
        axis=1,
    )
    scale, shift = np.linalg.lstsq(
        design,
        quality_sample.ravel(),
        rcond=None,
    )[0]
    aligned = fast_sample * scale + shift
    span = np.percentile(quality_sample, 99) - np.percentile(quality_sample, 1)
    normalized_rmse = np.sqrt(np.mean((aligned - quality_sample) ** 2)) / span
    correlation = np.corrcoef(
        quality_sample.ravel(),
        fast_sample.ravel(),
    )[0, 1]

    print(f"Device: {device.type}; frames: {len(frames)}")
    print(f"Quality (518/stride 1): {quality_time:.3f}s")
    print(f"Fast    (420/stride 2): {fast_time:.3f}s")
    print(f"Speedup: {quality_time / fast_time:.2f}x")
    print(f"Fast/quality correlation: {correlation:.5f}")
    print(f"Affine-aligned normalized RMSE: {normalized_rmse:.5f}")


if __name__ == "__main__":
    main()
