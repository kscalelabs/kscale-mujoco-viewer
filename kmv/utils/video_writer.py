"""Video writer for saving frames incrementally with overlay information."""

import logging
import os
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

logger = logging.getLogger(__name__)


class StreamingVideoWriter:
    """A class to write video frames incrementally with text overlays."""
    
    def __init__(
        self, 
        save_path: Union[str, Path], 
        fps: int = 30,
        frame_width: int = 640, 
        frame_height: int = 480,
        show_frame_number: bool = True,
        show_sim_time: bool = True,
        quality: Optional[int] = 8,  # Quality setting (0-10), higher is better
    ) -> None:
        """Initialize the video writer.
        
        Args:
            save_path: Path to save the video
            fps: Frames per second
            frame_width: Width of the frames (for info only)
            frame_height: Height of the frames (for info only)
            show_frame_number: Whether to show the frame number on frames
            show_sim_time: Whether to show simulation time on frames
            quality: Video quality (0-10, 10 being highest), affects bitrate
        """
        self.save_path = Path(save_path)
        self.fps = fps
        self.frame_count = 0
        self.show_frame_number = show_frame_number
        self.show_sim_time = show_sim_time
        self.quality = quality
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_path.parent, exist_ok=True)
        
        # Set up the writer based on file extension
        ext = self.save_path.suffix.lower()
        
        if ext == ".mp4":
            try:
                import imageio.v2 as imageio
            except ImportError:
                raise RuntimeError(
                    "Failed to initialize video writer - saving .mp4 videos requires "
                    "imageio with FFMPEG backend, which can be installed using "
                    "`pip install 'imageio[ffmpeg]'`. FFMPEG must also be installed."
                )
            
            # Configure FFMPEG writer for MP4
            # Calculate bitrate based on quality (higher quality = higher bitrate)
            bitrate = None
            if self.quality is not None:
                bitrate = str(int(1000000 * (self.quality + 1) / 5))  # 200k-2.2M range
            
            # Use standard H.264 codec which works well across platforms
            try:
                self.writer = imageio.get_writer(
                    self.save_path,
                    format='FFMPEG',
                    mode='I',
                    fps=self.fps,
                    codec='h264',  # H.264 is well-supported across platforms
                    pixelformat='yuv420p',  # Standard pixel format for compatibility
                    bitrate=bitrate,
                    macro_block_size=16  # Ensures dimensions are compatible with codec
                )
                logger.info("Initialized MP4 writer with H.264 codec at %d FPS", fps)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize MP4 writer: {e}")
            
        elif ext == ".gif":
            try:
                import imageio.v2 as imageio
                self.writer = imageio.get_writer(
                    self.save_path,
                    format='GIF',
                    mode='I',
                    fps=self.fps,
                    loop=0  # 0 means loop indefinitely
                )
                logger.info("Initialized GIF writer at %d FPS", fps)
            except ImportError:
                raise RuntimeError(
                    "Failed to initialize GIF writer - saving GIFs requires imageio. "
                    "Please install it using `pip install imageio`."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize GIF writer: {e}")
            
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Expected .mp4 or .gif")
    
    def _add_text_overlay(self, frame: np.ndarray, sim_time: float) -> np.ndarray:
        """Add text overlays to the frame.
        
        Args:
            frame: The frame to modify (RGB format)
            sim_time: Current simulation time
            
        Returns:
            Frame with overlaid text
        """
        # Convert numpy array to PIL Image
        pil_image = PIL.Image.fromarray(frame)
        draw = PIL.ImageDraw.Draw(pil_image)
        
        # Try to get a font, fall back to default if not available
        font = None
        try:
            # Try to use a system font that's likely to be available
            # Different systems have different fonts available
            fonts_to_try = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
                '/Library/Fonts/Arial.ttf',  # Mac
                'C:\\Windows\\Fonts\\Arial.ttf',  # Windows
                None  # Fallback to default
            ]
            
            for font_path in fonts_to_try:
                try:
                    if font_path:
                        font = PIL.ImageFont.truetype(font_path, size=20)
                        break
                except Exception:
                    continue
        except Exception:
            # If anything goes wrong, proceed without a custom font
            pass
        
        # Define text to display
        texts = []
        if self.show_frame_number:
            texts.append(f"Frame: {self.frame_count}")
        if self.show_sim_time:
            texts.append(f"Time: {sim_time:.3f}s")
        
        # Draw text with a shadow effect for better visibility
        for i, text in enumerate(texts):
            x, y = 10, 10 + i * 30
            # Draw shadow
            draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
            # Draw text
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
        # Convert back to numpy array
        return np.array(pil_image)
    
    def add_frame(self, frame: np.ndarray, sim_time: float = 0.0) -> None:
        """Add a frame to the video with overlaid information.
        
        Args:
            frame: The frame to add (RGB format)
            sim_time: Current simulation time
        """
        if not hasattr(self, 'writer') or self.writer is None:
            logger.warning("Attempt to add frame to closed writer")
            return
            
        # Add text overlay if enabled
        if self.show_frame_number or self.show_sim_time:
            frame = self._add_text_overlay(frame, sim_time)
        
        # Write the frame
        self.writer.append_data(frame)
        self.frame_count += 1
        
        # Log progress periodically
        if self.frame_count % 100 == 0:
            logger.info("Captured %d frames", self.frame_count)
    
    def close(self) -> None:
        """Close the video writer and finalize the video."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            logger.info("Saved video with %d frames to: %s", self.frame_count, self.save_path)
            self.writer = None 