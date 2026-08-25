#!/usr/bin/env python3
"""Regenerate the sidebar-logo block in _static/custom.css.

The book theme shows the site logo in the top bar. We hide it there and paint it
large above the left-hand menu instead, using a CSS background image on
`.myst-primary-sidebar-nav::before`.

Both a light and a dark variant are inlined as data URIs rather than referenced by
path: the MyST build renames static files with a content hash
(/build/<name>-<hash>.png), so a plain url(...) would break whenever the image
changes.

The dark variant is taken from _static/img/sbu_dse_emblem_dark.png when that file
exists. Otherwise it is derived from the light artwork and written there, so it can
be inspected and replaced with a hand-drawn version at any time.

Usage:
    python tools/embed_logo.py                 # use (or derive) the standard files
    python tools/embed_logo.py light.png       # different light source
    python tools/embed_logo.py light.png dark.png

Everything above the "---- sidebar logo" marker in custom.css is preserved.
"""

import base64
import io
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CSS = ROOT / "_static" / "custom.css"
LIGHT = ROOT / "_static" / "img" / "sbu_dse_emblem.png"
DARK = ROOT / "_static" / "img" / "sbu_dse_emblem_dark.png"
MARKER = "\n/* ---- sidebar logo"

RENDER_WIDTH = 700  # 2x the widest the sidebar gets, for retina displays
INK_THRESHOLD = 235  # grey level below which a pixel counts as artwork
MIN_INK_FRACTION = 0.01  # ignore rows/columns holding only faint background noise
# A pixel this pale and this close to neutral is paper, not drawing. The emblem has
# faint off-white speckle across the whole canvas; inverting it without this test
# turns the background into visible mottling.
PAPER_LIGHTNESS = 0.90
PAPER_CHROMA = 0.03
# Lightness range the inverted artwork is mapped into: former white lands on a dark
# grey close to the theme background rather than pure black, and the former black
# outlines stop just short of pure white.
DARK_FLOOR, DARK_CEIL = 0.11, 0.97


def rgb_to_hsl(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    lightness = (mx + mn) / 2
    sat = np.where(d < 1e-9, 0.0, d / (1 - np.abs(2 * lightness - 1) + 1e-9))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hue = np.zeros_like(lightness)
    m = (d > 1e-9) & (mx == r); hue[m] = ((g - b)[m] / d[m]) % 6
    m = (d > 1e-9) & (mx == g); hue[m] = ((b - r)[m] / d[m]) + 2
    m = (d > 1e-9) & (mx == b); hue[m] = ((r - g)[m] / d[m]) + 4
    return hue * 60, np.clip(sat, 0, 1), lightness


def hsl_to_rgb(hue: np.ndarray, sat: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    c = (1 - np.abs(2 * lightness - 1)) * sat
    hp = (hue / 60.0) % 6
    x = c * (1 - np.abs(hp % 2 - 1))
    m = lightness - c / 2
    z = np.zeros_like(c)
    out = np.zeros(c.shape + (3,))
    done = np.zeros(c.shape, bool)
    for cond, parts in zip(
        [hp < 1, hp < 2, hp < 3, hp < 4, hp < 5, hp <= 6],
        [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)],
    ):
        sel = cond & ~done
        out[sel] = np.stack(parts, -1)[sel]
        done |= sel
    return np.clip(out + m[..., None], 0, 1)


def derive_dark(img: Image.Image) -> Image.Image:
    """Invert lightness while preserving hue, and drop the paper background.

    A plain negative would swing the hues (the blue bus would turn orange), so the
    inversion happens in HSL and touches lightness only. Paper-white pixels become
    transparent so the emblem sits on whatever the dark theme paints behind it.
    """
    a = np.asarray(img.convert("RGB")).astype(float) / 255
    hue, sat, lightness = rgb_to_hsl(a)
    chroma = a.max(-1) - a.min(-1)
    paper = (lightness > PAPER_LIGHTNESS) & (chroma < PAPER_CHROMA)

    rgb = hsl_to_rgb(hue, sat, DARK_FLOOR + (DARK_CEIL - DARK_FLOOR) * (1 - lightness))
    rgba = np.concatenate([rgb, np.where(paper, 0.0, 1.0)[..., None]], -1)
    return Image.fromarray((rgba * 255).round().astype("uint8"), "RGBA")


def artwork_box(img: Image.Image, pad: int = 10) -> tuple[int, int, int, int]:
    """Bounding box of the real artwork, ignoring faint background noise.

    A plain getbbox() is not enough: the emblem has near-white speckle across the
    whole canvas, which makes every edge look like content.
    """
    grey = np.asarray(img.convert("L")).astype(int)
    ink = grey < INK_THRESHOLD
    rows = np.where(ink.mean(axis=1) > MIN_INK_FRACTION)[0]
    cols = np.where(ink.mean(axis=0) > MIN_INK_FRACTION)[0]
    if not len(rows) or not len(cols):
        return (0, 0, img.width, img.height)
    return (
        max(0, cols.min() - pad),
        max(0, rows.min() - pad),
        min(img.width, cols.max() + pad + 1),
        min(img.height, rows.max() + pad + 1),
    )


def data_uri(img: Image.Image, box: tuple[int, int, int, int]) -> str:
    cropped = img.crop(box)
    scaled = cropped.resize(
        (RENDER_WIDTH, round(cropped.height * RENDER_WIDTH / cropped.width)),
        Image.LANCZOS,
    )
    buf = io.BytesIO()
    scaled.save(buf, "WEBP", quality=82, method=6)  # WebP keeps the alpha channel
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    light_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LIGHT
    dark_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DARK

    light = Image.open(light_path).convert("RGB")
    # Crop both variants identically so they line up pixel for pixel.
    box = artwork_box(light)

    if dark_path.exists():
        dark = Image.open(dark_path).convert("RGBA")
        origin = f"read {dark_path.name}"
    else:
        dark = derive_dark(light)
        dark.save(dark_path)
        origin = f"derived, saved as {dark_path.name}"

    css = re.split(re.escape(MARKER), CSS.read_text())[0].rstrip() + "\n"
    w, h = box[2] - box[0], box[3] - box[1]
    CSS.write_text(
        css + BLOCK.format(w=w, h=h, light=data_uri(light, box), dark=data_uri(dark, box))
    )

    print(f"light: {light_path.name} cropped to {w}x{h}")
    print(f"dark : {origin}")
    print(f"wrote {CSS.relative_to(ROOT)} ({CSS.stat().st_size/1024:.1f} KB)")


BLOCK = '''
/* ---- sidebar logo ------------------------------------------------------
   Generated by tools/embed_logo.py — re-run that script instead of editing
   the data URIs below by hand.

   The book theme shows the logo in the top bar. We hide it there and paint it
   large above the left-hand menu instead. The images are inlined as data URIs
   because the build renames static files with a content hash, so a plain
   url(...) path would silently break whenever the artwork changes.

   Dark mode in this theme is class-based (Tailwind compiles dark: to
   :is(.dark *)), so the dark rule keys off .dark rather than a
   prefers-color-scheme media query.
   ------------------------------------------------------------------------ */

.myst-home-link-logo {{
  display: none;
}}

.myst-primary-sidebar-nav::before {{
  content: "";
  display: block;
  width: calc(100% - 0.5rem);
  aspect-ratio: {w} / {h};
  margin: 0 0.25rem 1.25rem;
  background-image: url("{light}");
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center top;
  border-radius: 10px;
}}

/* dark variant: white outlines, transparent background, hues preserved */
.dark .myst-primary-sidebar-nav::before {{
  background-image: url("{dark}");
}}
'''


if __name__ == "__main__":
    main()
