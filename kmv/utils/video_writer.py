"""Video writer for saving frames incrementally with overlay information."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

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
        self.gif_quality = 10  # Always use maximum quality for GIFs
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_path.parent, exist_ok=True)
        
        # Set up file paths based on input
        self.is_gif = self.save_path.suffix.lower() == ".gif"
        
        # If the target is GIF, we'll create an MP4 first and convert later
        if self.is_gif:
            self.temp_mp4_path = self.save_path.with_suffix('.temp.mp4')
            self.writer_path = self.temp_mp4_path
        else:
            self.writer_path = self.save_path
        
        # Initialize MP4 writer
        try:
            import imageio.v2 as imageio
        except ImportError:
            raise RuntimeError(
                "Failed to initialize video writer - saving videos requires "
                "imageio with FFMPEG backend, which can be installed using "
                "`pip install 'imageio[ffmpeg]'`. FFMPEG must also be installed."
            )
        
        # Calculate bitrate based on quality (higher quality = higher bitrate)
        bitrate = None
        if self.quality is not None:
            bitrate = str(int(1000000 * (self.quality + 1) / 5))  # 200k-2.2M range
        
        # Set up the writer with MP4 format
        try:
            self.writer = imageio.get_writer(
                self.writer_path,
                format='FFMPEG',
                mode='I',
                fps=self.fps,
                codec='h264',  # H.264 is well-supported across platforms
                pixelformat='yuv420p',  # Standard pixel format for compatibility
                bitrate=bitrate,
                macro_block_size=16  # Ensures dimensions are compatible with codec
            )
            logger.info(f"Initialized video writer for {self.save_path} (fps={fps})")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize video writer: {e}")
    
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
            logger.info(f"Captured {self.frame_count} frames")
    
    def _convert_mp4_to_gif(self) -> None:
        """Convert the temporary MP4 file to GIF format using ffmpeg with palette generation.
        
        This uses a two-step process:
        1. Generate an optimized color palette from the video
        2. Create the GIF using that palette for better quality and smaller file size
        """
        try:
            # Check if MP4 file exists
            mp4_path = str(self.temp_mp4_path)
            if not os.path.exists(mp4_path):
                logger.error(f"MP4 file does not exist at path: {mp4_path}")
                return
            
            logger.info(f"MP4 file exists with size: {os.path.getsize(mp4_path)} bytes")
            
            # Use ffmpeg directly via subprocess for better quality and smaller file size
            gif_path = str(self.save_path)
            logger.info(f"Converting MP4 to GIF using ffmpeg with palette: {mp4_path} -> {gif_path}")
            
            # Create a temporary palette file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_palette:
                palette_path = temp_palette.name
            
            try:
                # Using maximum quality settings for GIFs
                # Colors: 256 colors for maximum quality
                colors = 256
                
                # Width: 640px for highest quality
                width = 640
                
                # FPS: Use original FPS for highest quality
                gif_fps = self.fps
                
                # Dithering: Use sierra2_4a for highest quality
                dither_method = "sierra2_4a"
                
                logger.info(f"GIF generation with high quality: colors={colors}, width={width}px, fps={gif_fps}, dither={dither_method}")
                
                # Step 1: Generate palette with custom color count
                palette_cmd = [
                    "ffmpeg",
                    "-i", mp4_path,
                    "-vf", f"fps={gif_fps},scale={width}:-1:flags=lanczos,palettegen=max_colors={colors}:stats_mode=diff",
                    "-y",  # Overwrite output file if it exists
                    palette_path
                ]
                
                logger.info(f"Step 1: Running palette generation: {' '.join(palette_cmd)}")
                palette_result = subprocess.run(palette_cmd, capture_output=True, text=True)
                
                if palette_result.returncode != 0:
                    logger.error(f"Palette generation failed with code {palette_result.returncode}")
                    logger.error(f"stdout: {palette_result.stdout}")
                    logger.error(f"stderr: {palette_result.stderr}")
                    logger.info(f"Keeping MP4 file at: {mp4_path} (GIF conversion failed)")
                    return
                    
                logger.info("Palette generated successfully")
                
                # Step 2: Create GIF using the palette with custom dithering
                gif_cmd = [
                    "ffmpeg",
                    "-i", mp4_path,
                    "-i", palette_path,
                    "-filter_complex", 
                    f"fps={gif_fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither={dither_method}:diff_mode=rectangle",
                    "-loop", "0",  # Loop infinitely
                    "-y",  # Overwrite output file if it exists
                    gif_path
                ]
                
                logger.info(f"Step 2: Creating GIF with palette: {' '.join(gif_cmd)}")
                gif_result = subprocess.run(gif_cmd, capture_output=True, text=True)
                
                if gif_result.returncode == 0:
                    logger.info("GIF creation successful")
                    # Check if GIF was created
                    if os.path.exists(gif_path):
                        logger.info(f"GIF file created successfully with size: {os.path.getsize(gif_path)} bytes")
                    else:
                        logger.error(f"Failed to create GIF file at: {gif_path}")
                else:
                    logger.error(f"GIF creation failed with code {gif_result.returncode}")
                    logger.error(f"stdout: {gif_result.stdout}")
                    logger.error(f"stderr: {gif_result.stderr}")
                    logger.info(f"Keeping MP4 file at: {mp4_path} (GIF conversion failed)")
                    return
                    
            finally:
                # Clean up the palette file
                if os.path.exists(palette_path):
                    os.remove(palette_path)
                    logger.info(f"Removed temporary palette file: {palette_path}")
            
            # Remove the temporary MP4 file if conversion succeeded
            os.remove(mp4_path)
            logger.info(f"Successfully converted to GIF: {gif_path}")
            
        except Exception as e:
            logger.error(f"Failed to convert MP4 to GIF: {e}")
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            logger.info(f"MP4 file is still available at: {self.temp_mp4_path}")
    
    def close(self) -> None:
        """Close the video writer and finalize the video."""
        if hasattr(self, 'writer') and self.writer is not None:
            logger.info(f"Closing StreamingVideoWriter with {self.frame_count} frames")
            self.writer.close()
            logger.info(f"Writer closed, video saved to: {self.writer_path}")
            
            # Convert to GIF if needed
            if self.is_gif:
                logger.info(f"Starting GIF conversion process for {self.temp_mp4_path}")
                self._convert_mp4_to_gif()
            
            self.writer = None
            logger.info("StreamingVideoWriter fully closed") 