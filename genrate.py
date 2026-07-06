import os
import json
import gi
import cairo
import cv2
import numpy as np

from PIL import Image

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")

from gi.repository import Pango, PangoCairo

# ============================================================
# Files
# ============================================================

TEMPLATE = "template.jpg"
OUTPUT_DIR = "output_cards"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Dataset
# ============================================================

with open("dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

with open("config.json", encoding="utf-8") as f:
    POS = json.load(f)

# ============================================================
# Load template
# ============================================================

def load_surface(path):

    img = Image.open(path).convert("RGBA")

    w, h = img.size

    data = bytearray(img.tobytes("raw", "BGRA"))

    surface = cairo.ImageSurface.create_for_data(
        data,
        cairo.FORMAT_ARGB32,
        w,
        h
    )

    return surface, data


# ============================================================
# Draw Arabic Text
# ============================================================
def draw_text(
        ctx,
        text,
        x,
        y,
        font="Noto Sans Arabic",
        size=25,
        color=(0.08, 0.08, 0.08),
        letter_spacing = 0,
        is_bold = False
):

    layout = PangoCairo.create_layout(ctx)

    layout.set_text(str(text), -1)

    desc = Pango.FontDescription()

    desc.set_family(font)

    desc.set_size(size * Pango.SCALE)

    if is_bold:
        desc.set_weight(Pango.Weight.BOLD)

    layout.set_font_description(desc)

    if letter_spacing > 0:
        attrs = Pango.AttrList()
        spacing_attr = Pango.attr_letter_spacing_new(int(letter_spacing * Pango.SCALE))
        attrs.insert(spacing_attr)
        layout.set_attributes(attrs)

    layout.set_auto_dir(True)

    layout.set_alignment(Pango.Alignment.RIGHT)

    layout.set_width(-1)

    layout.set_single_paragraph_mode(False)

    layout.set_wrap(Pango.WrapMode.WORD_CHAR)

    layout.set_justify(False)

    layout.set_ellipsize(Pango.EllipsizeMode.NONE)

    _, logical = layout.get_pixel_extents()

    ctx.set_source_rgb(*color)

    ctx.move_to(
        x - logical.width,
        y
    )

    PangoCairo.show_layout(ctx, layout)


# ============================================================
# Generate Cards
# ============================================================

for i, person in enumerate(dataset):

    surface, _ = load_surface(TEMPLATE)

    ctx = cairo.Context(surface)

        # -------------------------------
    # First Name
    # -------------------------------

    draw_text(
        ctx,
        person["name_first"],
        *POS["name_first"],
        font="Simplified Arabic",
        size=31,
        is_bold=True
    )

    # -------------------------------
    # Rest of Name
    # -------------------------------

    draw_text(
        ctx,
        person["name_rest"],
        *POS["name_rest"],
        font="Kufi",
        size=31,
        is_bold=True

    )

    # -------------------------------
    # Street
    # -------------------------------

    draw_text(
        ctx,
        person["street"],
        *POS["street"],
        font="Simplified Arabic",
        size=31,
        is_bold=True

    )

    # -------------------------------
    # Rest of Address
    # -------------------------------

    draw_text(
        ctx,
        person["address_rest"],
        *POS["address_rest"],
        font="Noto Sans Arabic",
        size=31,
        is_bold=True

    )

    # -------------------------------
    # National ID
    # -------------------------------
    draw_text(
        ctx,
        person["national_id"],
        *POS["nid"],
        font="DejaVu Sans",
        size=36,
        letter_spacing=17,
        is_bold=True

    )

    # ============================================================
    # Save temporary PNG
    # ============================================================

    tmp = os.path.join(
        OUTPUT_DIR,
        f"tmp_{i}.png"
    )

    surface.write_to_png(tmp)

    # ============================================================
    # OpenCV Post-processing
    # ============================================================

    img = cv2.imread(tmp)

    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0.3
    )

    noise = np.random.normal(
        0,
        2,
        img.shape
    )

    img = img.astype(np.float32)

    img += noise

    img = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    output = os.path.join(
        OUTPUT_DIR,
        f"card_{i}.jpg"
    )

    cv2.imwrite(
        output,
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    os.remove(tmp)

    print(f"Saved: {output}")

print("Done.")