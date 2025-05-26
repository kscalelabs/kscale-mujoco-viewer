"""CLI entry-point:  python -m kmv scene.xml"""

import argparse
import colorlogging
import logging
from pathlib import Path
from .viewer import launch


def main() -> None:
    # Set up logging for the CLI
    colorlogging.configure()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="KMV MuJoCo viewer")
    parser.add_argument("xml", help="Path to MuJoCo XML scene file")
    args = parser.parse_args()

    try:
        xml_text = Path(args.xml).read_text()
        logger.info(f"Successfully loaded XML file: {args.xml}")
        logger.info(f"XML content length: {len(xml_text)} characters")
        logger.info("Launching viewer...")
        launch(xml_text)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()