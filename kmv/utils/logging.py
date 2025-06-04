"""Logging utilities."""

import logging
from pathlib import Path

import numpy as np
import PIL.Image

logger = logging.getLogger(__name__)


class VideoWriter:
    """Utility for streamed recording of frames to an MP4 file."""

    def __init__(self, save_path: str | Path, fps: int = 30) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() != ".mp4":
            raise ValueError(
                "VideoWriter currently supports only .mp4 files for incremental "
                "saving. Use `save_video` for other formats."
            )

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Failed to initialize video writer – saving .mp4 videos with "
                "imageio requires the FFMPEG backend. Install it with "
                "`pip install 'imageio[ffmpeg]'` and ensure ffmpeg is available "
                "on your system."
            ) from exc

        self._writer = imageio.get_writer(path, mode="I", fps=fps)
        self._path = path

    def append(self, frame: np.ndarray) -> None:
        """Add a single RGB frame (H×W×3, uint8) to the video on disk."""
        self._writer.append_data(frame)

    def close(self) -> None:
        """Finalize the file and release resources."""
        if self._writer is not None:
            self._writer.close()
            logger.info("Saved mp4 video to: %s", self._path)
            del self._writer


def save_video(frames: list[np.ndarray], save_path: str | Path, fps: int = 30) -> None:
    """Save captured frames as video (MP4) or GIF.

    Args:
        frames: List of frames to save.
        save_path: Path to save the video.
        fps: Frames per second for the video.

    Raises:
        ValueError: If no frames to save or unsupported file extension.
        RuntimeError: If issues with saving video, especially MP4 format issues.
    """
    (path := Path(save_path)).parent.mkdir(parents=True, exist_ok=True)

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
                        writer.append_data(frame)

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


def save_logs(logs: list[dict[str, np.ndarray]], save_path: str | Path) -> None:
    """Save stacked arrays from logs to CSV files.

    Args:
        logs: List of dictionaries containing arrays to save
        save_path: Directory to save the CSV files
    """
    (path := Path(save_path)).mkdir(parents=True, exist_ok=True)

    all_keys: set[str] = set()
    for log in logs:
        all_keys.update(log.keys())

    for key in all_keys:
        arrays = []
        for log in logs:
            if key in log:
                arrays.append(log[key])

        if arrays:
            stacked = np.stack(arrays)
            np.savetxt(path / f"{key}.tsv", stacked, delimiter="\t")
