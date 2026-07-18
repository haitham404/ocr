import json
from pathlib import Path

import cv2


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("Dataset")
OUTPUT_DIR = Path("rec_dataset")


# ============================================================
# CROP TEXT REGION
# ============================================================

def crop_text_region(image, points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))

    x2 = min(image.shape[1], max(xs))
    y2 = min(image.shape[0], max(ys))

    return image[y1:y2, x1:x2]


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name):
    input_label_file = DATASET_DIR / f"{split_name}.txt"
    output_label_file = OUTPUT_DIR / f"{split_name}.txt"

    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    lines = input_label_file.read_text(
        encoding="utf-8"
    ).splitlines()

    print(f"\nProcessing {split_name.upper()}")
    print("=" * 60)

    for line in lines:
        if not line.strip():
            continue

        image_path, annotations_json = line.split("\t", 1)

        full_image_path = DATASET_DIR / image_path
        image = cv2.imread(str(full_image_path))

        if image is None:
            print(f"[ERROR] Cannot read: {full_image_path}")
            continue

        annotations = json.loads(annotations_json)

        image_path_obj = Path(image_path)

        card_name = image_path_obj.parent.name
        side = image_path_obj.stem

        # rec_dataset/train/card_000/front/
        crop_dir = split_dir / card_name / side
        crop_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {card_name}/{side}")

        for index, annotation in enumerate(annotations):
            text = annotation["transcription"].strip()

            if not text or text == "###":
                continue

            points = annotation["points"]

            crop = crop_text_region(
                image=image,
                points=points,
            )

            if crop.size == 0:
                print(
                    f"[WARNING] Empty crop: "
                    f"{card_name}/{side} -> {text}"
                )
                continue

            # ====================================================
            # SAVE CROP
            # ====================================================

            crop_name = f"{index:02d}.jpg"
            crop_path = crop_dir / crop_name

            success = cv2.imwrite(
                str(crop_path),
                crop,
            )

            if not success:
                print(f"[ERROR] Failed to save: {crop_path}")
                continue

            relative_path = crop_path.relative_to(OUTPUT_DIR)

            samples.append(
                (
                    relative_path.as_posix(),
                    text,
                )
            )

    # ============================================================
    # SAVE LABEL FILE
    # ============================================================

    with output_label_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for image_path, text in samples:
            file.write(f"{image_path}\t{text}\n")

    print("-" * 60)
    print(f"{split_name.upper()} samples: {len(samples)}")

    return len(samples)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_count = process_split("train")
    val_count = process_split("val")

    total = train_count + val_count

    print("\n")
    print("=" * 60)
    print("RECOGNITION DATASET GENERATED")
    print("=" * 60)

    print(f"Train samples : {train_count}")
    print(f"Val samples   : {val_count}")
    print(f"Total samples : {total}")

    print("=" * 60)


if __name__ == "__main__":
    main()