from pathlib import Path

import cv2


DATASET_DIR = Path("Dataset")
OUTPUT_DIR = Path("fixed_crops")

NUM_CARDS = 10


ROIS = {
    "first_name": (820, 140, 1030, 235),

    "full_name": (560, 210, 1030, 295),

    "address_1": (500, 285, 1030, 370),

    "address_2": (550, 350, 1030, 450),

    "national_id": (380, 470, 1030, 565),

    "birth_date": (0, 450, 400, 555),
}


def crop_rois(image, card_name):
    card_output_dir = OUTPUT_DIR / card_name
    card_output_dir.mkdir(parents=True, exist_ok=True)

    for field_name, (x1, y1, x2, y2) in ROIS.items():
        crop = image[y1:y2, x1:x2]

        output_path = card_output_dir / f"{field_name}.jpg"

        cv2.imwrite(str(output_path), crop)

        print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    cards = sorted(DATASET_DIR.glob("card_*"))[:NUM_CARDS]

    print(f"Testing {len(cards)} cards")

    for card_dir in cards:
        image_path = card_dir / "front.jpg"

        print(f"\nProcessing: {image_path}")

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        crop_rois(
            image=image,
            card_name=card_dir.name,
        )


if __name__ == "__main__":
    main()