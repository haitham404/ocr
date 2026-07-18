import os
import time

from paddleocr import TextRecognition


# =====================================================
# CONFIG
# =====================================================

IMAGE_DIR = "fixed_crops"
OUTPUT_DIR = "results_rec_only"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD RECOGNITION MODEL ONLY
# =====================================================

recognizer = TextRecognition(
    model_name="arabic_PP-OCRv5_mobile_rec",
    device="cpu",
)


# =====================================================
# LOOP OVER CARDS
# =====================================================

cards = sorted(os.listdir(IMAGE_DIR))

for card in cards:

    card_dir = os.path.join(IMAGE_DIR, card)

    if not os.path.isdir(card_dir):
        continue

    # Output folder for each card
    card_output_dir = os.path.join(OUTPUT_DIR, card)

    os.makedirs(
        card_output_dir,
        exist_ok=True,
    )

    crops = sorted(os.listdir(card_dir))

    for crop_name in crops:

        if not crop_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            card_dir,
            crop_name,
        )

        field_name = os.path.splitext(crop_name)[0]

        print("=" * 70)
        print(f"Card  : {card}")
        print(f"Field : {field_name}")

        # =====================================================
        # RECOGNITION
        # =====================================================

        t0 = time.perf_counter()

        results = recognizer.predict(
            input=image_path,
            batch_size=1,
        )

        latency = time.perf_counter() - t0

        for result in results:

            res = result.json["res"]

            text = res["rec_text"]
            score = float(res["rec_score"])

            print(f"Text    : {text}")
            print(f"Score   : {score:.3f}")
            print(f"Latency : {latency:.3f} sec")

            # =====================================================
            # SAVE TXT
            # =====================================================

            txt_path = os.path.join(
                card_output_dir,
                f"{field_name}.txt",
            )

            with open(
                txt_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)

            print(f"Saved: {txt_path}")


print("\nRecognition finished.")