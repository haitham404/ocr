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

FRONT_TEMPLATE = "template_front.jpg"
BACK_TEMPLATE = "template_back.jpg"

OUTPUT_DIR = "output_cards"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Dataset
# ============================================================

with open("front_dataset.json", encoding="utf-8") as f:
    front_dataset = json.load(f)

with open("back_dataset.json", encoding="utf-8") as f:
    back_dataset = json.load(f)

with open("front_config.json", encoding="utf-8") as f:
    FRONT_POS = json.load(f)

with open("back_config.json", encoding="utf-8") as f:
    BACK_POS = json.load(f)

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
def draw_back_text(
        ctx,
        text,
        x,
        y,
        font="Noto Sans Arabic",
        size=25,
        color=(0.08, 0.08, 0.08),
        letter_spacing=0,
        is_bold=False,
        align="right",
        width=700
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
        attrs.insert(
            Pango.attr_letter_spacing_new(
                int(letter_spacing * Pango.SCALE)
            )
        )
        layout.set_attributes(attrs)

    layout.set_auto_dir(True)
    layout.set_width(width * Pango.SCALE)

    if align == "left":
        layout.set_alignment(Pango.Alignment.LEFT)
    elif align == "center":
        layout.set_alignment(Pango.Alignment.CENTER)
    else:
        layout.set_alignment(Pango.Alignment.RIGHT)

    layout.set_wrap(Pango.WrapMode.WORD_CHAR)
    layout.set_justify(False)
    layout.set_ellipsize(Pango.EllipsizeMode.NONE)

    ctx.set_source_rgb(*color)
    ctx.move_to(x, y)

    PangoCairo.show_layout(ctx, layout)

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



def post_process(surface, output_path):

    tmp = output_path + ".png"

    surface.write_to_png(tmp)

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

    cv2.imwrite(
        output_path,
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    os.remove(tmp)


# ============================================================
# Generate Cards
# ============================================================

for i, (person, back) in enumerate(zip(front_dataset, back_dataset)):

    card_dir = os.path.join(
        OUTPUT_DIR,
        f"card_{i:03d}"
    )

    os.makedirs(card_dir, exist_ok=True)

    # ========================================================
    # FRONT
    # ========================================================

    surface, _ = load_surface(FRONT_TEMPLATE)

    ctx = cairo.Context(surface)

    # -------------------------------
    # First Name
    # -------------------------------

    draw_text(
        ctx,
        person["name_first"],
        *FRONT_POS["name_first"],
        font="Simplified Arabic",
        size=31,
        is_bold=True
    )

    # -------------------------------
    # Rest Name
    # -------------------------------

    draw_text(
        ctx,
        person["name_rest"],
        *FRONT_POS["name_rest"],
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
        *FRONT_POS["street"],
        font="Simplified Arabic",
        size=31,
        is_bold=True
    )

    # -------------------------------
    # Address
    # -------------------------------

    draw_text(
        ctx,
        person["address_rest"],
        *FRONT_POS["address_rest"],
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
        *FRONT_POS["nid"],
        font="DejaVu Sans",
        size=36,
        letter_spacing=17,
        is_bold=True
    )

    post_process(
        surface,
        os.path.join(card_dir, "front.jpg")
    )

    # ========================================================
    # BACK
    # ========================================================

    surface, _ = load_surface(BACK_TEMPLATE)

    ctx = cairo.Context(surface)

    # -------------------------------
    # National ID
    # -------------------------------

    draw_back_text(
        ctx,
        back["national_id"],
        *BACK_POS["national_id"],
        font="DejaVu Sans",
        size=34,
        letter_spacing=17,
        is_bold=True
    )

    # -------------------------------
    # Occupation
    # -------------------------------

    draw_back_text(
        ctx,
        back["occupation"],
        *BACK_POS["occupation"],
        font="Simplified Arabic",
        size=30,
        is_bold=True
    )

    # # -------------------------------
    # # Issue Date
    # # -------------------------------

    # draw_text(
    #     ctx,
    #     back["issue_date"],
    #     *BACK_POS["issue_date"],
    #     font="DejaVu Sans",
    #     size=28,
    #     is_bold=True
    # )

        # -------------------------------
        # Expiry Label
        # -------------------------------
    # التاريخ أولاً (يمين)
    draw_back_text(
        ctx,
        back["expiry_date"],
        *BACK_POS["expiry_date"],
        font="DejaVu Sans",
        size=28,
        is_bold=True,
        align="right"
    )

    # ثم عبارة البطاقة سارية حتى
    draw_back_text(
        ctx,
        "البطاقة سارية حتى",
        *BACK_POS["expiry_label"],
        font="Simplified Arabic",
        size=30,
        is_bold=True,
        align="right"
    )
    # -------------------------------
    # Religion
    # -------------------------------

    draw_back_text(
        ctx,
        back["religion"],
        *BACK_POS["religion"],
        font="Simplified Arabic",
        size=28,
        is_bold=True
    )

    # -------------------------------
    # Gender
    # -------------------------------

    draw_back_text(
        ctx,
        back["gender"],
        *BACK_POS["gender"],
        font="Simplified Arabic",
        size=28,
        is_bold=True
    )

    # -------------------------------
    # Marital Status
    # -------------------------------

    draw_back_text(
        ctx,
        back["marital_status"],
        *BACK_POS["marital_status"],
        font="Simplified Arabic",
        size=28,
        is_bold=True
    )

    post_process(
        surface,
        os.path.join(card_dir, "back.jpg")
    )

    print(f"Saved card {i:03d}")