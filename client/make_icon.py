from pathlib import Path

from PIL import Image


def main():
    assets = Path(__file__).parent / "assets"
    image = Image.open(assets / "1.png").convert("RGBA")
    image.save(
        assets / "reaground.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


if __name__ == "__main__":
    main()
