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

FRONT_TEMPLATE = "template_front.png"
BACK_TEMPLATE = "template_back.jpg"

OUTPUT_DIR = "output_cards"
OUTPUT_DIR_DATASET = "Dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_DATASET, exist_ok=True)

# ============================================================
# Dataset
# ============================================================

with open("dataset.json", encoding="utf-8") as f:
    front_dataset = json.load(f)

with open("dataset.json", encoding="utf-8") as f:
    back_dataset = json.load(f)

with open("config_new.json", encoding="utf-8") as f:
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
# Draw Arabic Text (Modified to return labels)
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

    _, logical_rect = layout.get_pixel_extents()
    lx = logical_rect.x
    lw = logical_rect.width
    lh = logical_rect.height

    if align == "center":
        draw_x = x - (lx + lw / 2)
    elif align == "right":
        draw_x = x - (lx + lw)
    else:
        draw_x = x - lx

    ctx.set_source_rgb(*color)
    ctx.move_to(draw_x, y)

    PangoCairo.show_layout(ctx, layout)

    return {
        "transcription": str(text),
        "points": [
            [int(draw_x), int(y)],
            [int(draw_x + lw), int(y)],
            [int(draw_x + lw), int(y + lh)],
            [int(draw_x), int(y + lh)]
        ]
    }

# ============================================================
# Draw Arabic Text (Modified to return labels)
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
    lw = logical.width
    lh = logical.height

    draw_x = x - lw

    ctx.set_source_rgb(*color)

    ctx.move_to(
        draw_x,
        y
    )

    PangoCairo.show_layout(ctx, layout)

    return {
        "transcription": str(text),
        "points": [
            [int(draw_x), int(y)],
            [int(draw_x + lw), int(y)],
            [int(draw_x + lw), int(y + lh)],
            [int(draw_x), int(y + lh)]
        ]
    }

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

all_labels = [] # الحصالة بتاعة المربعات

cnt = 0
for i, (person, back) in enumerate(zip(front_dataset, back_dataset)):
    # cnt+=1
    # if(cnt == 20):
    #     break
    
    folder_name = f"card_{i:03d}"
    card_dir = os.path.join(
        OUTPUT_DIR_DATASET,
        folder_name
    )

    os.makedirs(card_dir, exist_ok=True)

    # ========================================================
    # FRONT
    # ========================================================

    surface, _ = load_surface(FRONT_TEMPLATE)

    ctx = cairo.Context(surface)
    
    front_boxes = [] # حصالة مربعات الوش

    # -------------------------------
    # First Name
    # -------------------------------

    front_boxes.append(draw_text(
        ctx,
        person["name_first"],
        *FRONT_POS["name_first"],
        font="Simplified Arabic",
        size=23,
        is_bold=True
    ))

    # -------------------------------
    # Rest Name
    # -------------------------------

    front_boxes.append(draw_text(
        ctx,
        person["name_rest"],
        *FRONT_POS["name_rest"],
        font="Kufi",
        size=23,
        is_bold=True
    ))

    # -------------------------------
    # Street
    # -------------------------------

    front_boxes.append(draw_text(
        ctx,
        person["street"],
        *FRONT_POS["street"],
        font="Simplified Arabic",
        size=23,
        is_bold=True
    ))

    # -------------------------------
    # Address
    # -------------------------------

    front_boxes.append(draw_text(
        ctx,
        person["address_rest"],
        *FRONT_POS["address_rest"],
        font="Noto Sans Arabic",
        size=22,
        is_bold=True
    ))

    # -------------------------------
    # National ID
    # -------------------------------

    front_boxes.append(draw_text(
        ctx,
        person["national_id"],
        *FRONT_POS["nid"],
        font="DejaVu Sans",
        size=25,
        letter_spacing=17,
        is_bold=True
    ))
    
    # -------------------------------
    # Brith Date
    # -------------------------------

    front_boxes.append(draw_text(
        ctx,
        person["birth_date"],
        *FRONT_POS["bdate"],
        font="DejaVu Sans",
        size=28,
        letter_spacing=7,
        is_bold=True
    ))

    post_process(
        surface,
        os.path.join(card_dir, "front.jpg")
    )
    
    # حفظ سطر الوش في ملف الـ Label
    all_labels.append(f"{folder_name}/front.jpg\t{json.dumps(front_boxes, ensure_ascii=False)}\n")

    # ========================================================
    # BACK
    # ========================================================

    surface, _ = load_surface(BACK_TEMPLATE)

    ctx = cairo.Context(surface)
    
    back_boxes = [] # حصالة مربعات الضهر

    # -------------------------------
    # National ID
    # -------------------------------

    back_boxes.append(draw_text(
        ctx,
        back["national_id"],
        *BACK_POS["national_id"],
        font="DejaVu Sans",
        size=29,
        letter_spacing=10,
        is_bold=True
    ))

    # -------------------------------
    # Occupation
    # -------------------------------

    back_boxes.append(draw_text(
        ctx,
        back["occupation"],
        *BACK_POS["occupation"],
        font="Simplified Arabic",
        size=29,
        is_bold=True
    ))

    # -------------------------------
    # Issue Date
    # -------------------------------

    back_boxes.append(draw_text(
        ctx,
        back["issue_date"],
        *BACK_POS["issue_date"],
        font="DejaVu Sans",
        size=29,
        is_bold=True
    ))

    # -------------------------------
    # Expiry Date
    # -------------------------------
    back_boxes.append(draw_text(
        ctx,
        back["expiry_date"],
        *BACK_POS["expiry_date"],
        font="DejaVu Sans",
        size=29,
        is_bold=True
    ))

    # -------------------------------
    # Religion
    # -------------------------------

    back_boxes.append(draw_text(
        ctx,
        back["religion"],
        *BACK_POS["religion"],
        font="Simplified Arabic",
        size=29,
        is_bold=True
    ))

    # -------------------------------
    # Gender
    # -------------------------------

    back_boxes.append(draw_text(
        ctx,
        back["gender"],
        *BACK_POS["gender"],
        font="Simplified Arabic",
        size=29,
        is_bold=True
    ))

    # -------------------------------
    # Marital Status
    # -------------------------------

    back_boxes.append(draw_text(
        ctx,
        back["marital_status"],
        *BACK_POS["marital_status"],
        font="Simplified Arabic",
        size=29,
        is_bold=True
    ))

    post_process(
        surface,
        os.path.join(card_dir, "back.jpg")
    )
    
    # حفظ سطر الضهر في ملف الـ Label
    all_labels.append(f"{folder_name}/back.jpg\t{json.dumps(back_boxes, ensure_ascii=False)}\n")

    print(f"Saved card {i:03d}")

# ============================================================
# إنشاء ملف المربعات لـ PaddleOCR
# ============================================================
with open(os.path.join(OUTPUT_DIR_DATASET, "Label.txt"), "w", encoding="utf-8") as f:
    f.writelines(all_labels)

print("تم إنشاء ملف Label.txt بنجاح!")