"""CLI entry-point:  python -m kmv scene.xml"""

import argparse
from pathlib import Path
from .viewer import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="KMV MuJoCo viewer")
    parser.add_argument("xml", help="Path to MuJoCo XML scene file")
    args = parser.parse_args()

    try:
        xml_text = Path(args.xml).read_text()
        print(f"Successfully loaded XML file: {args.xml}")
        print(f"XML content length: {len(xml_text)} characters")
        print("Launching viewer...")
        launch(xml_text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()