"""Reduce the file size of a saved PNG figure.

Downsamples the image to 150 dpi (if it was saved with a higher dpi) and stores
it with a 256-colour palette. Used by the latent channel plots.

Usage:  python scripts/utils/reduce_png.py <file.png> [...]
"""
import sys

from PIL import Image

TARGET_DPI = 150


def reduce_png(path, target_dpi=TARGET_DPI):
    im = Image.open(path)
    dpi = im.info.get("dpi", (target_dpi, target_dpi))[0]
    im = im.convert("RGBA")
    if im.getchannel("A").getextrema() == (255, 255):  # no transparency
        im = im.convert("RGB")
    if round(dpi) > target_dpi:
        scale = target_dpi / dpi
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BOX)  # area average
        dpi = target_dpi
    if im.mode == "RGB":
        im = im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    else:
        im = im.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    im.save(path, optimize=True, dpi=(dpi, dpi))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        reduce_png(p)
