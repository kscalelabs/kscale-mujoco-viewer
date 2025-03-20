"""Utility functions for saving data."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)


def save_video(frames: list[np.ndarray], save_path: Optional[str | Path] = None, fps: int = 30) -> None:
    """Save captured frames as video (MP4) or GIF.

    Args:
        frames: List of frames to save.
        save_path: Path to save the video.
        fps: Frames per second for the video.

    Raises:
        ValueError: If no frames to save or unsupported file extension.
        RuntimeError: If issues with saving video, especially MP4 format issues.
    """
    if save_path is None:
        raise ValueError("No path to save video")

    # Use provided path or default
    path = Path(save_path)

    if len(frames) == 0:
        raise ValueError("No frames to save")

    match path.suffix.lower():
        case ".mp4":
            try:
                import imageio.v2 as imageio
            except ImportError:
                raise RuntimeError(
                    "Failed to save video - note that saving .mp4 videos with imageio usually "
                    "requires the FFMPEG backend, which can be installed using `pip install "
                    "'imageio[ffmpeg]'`. Note that this also requires FFMPEG to be installed in "
                    "your system."
                )

            try:
                with imageio.get_writer(path, mode="I", fps=fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)  # type: ignore[attr-defined]

                    logger.info("Saved mp4 video with %d frames to: %s", len(frames), path)
            except Exception as e:
                raise RuntimeError(
                    "Failed to save video - note that saving .mp4 videos with imageio usually "
                    "requires the FFMPEG backend, which can be installed using `pip install "
                    "'imageio[ffmpeg]'`. Note that this also requires FFMPEG to be installed in "
                    "your system."
                ) from e

        case ".gif":
            images = [PIL.Image.fromarray(frame) for frame in frames]
            images[0].save(
                path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            logger.info("Saved GIF with %d frames to %s", len(frames), path)

        case _:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Expected .mp4 or .gif")
