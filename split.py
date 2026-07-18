from pathlib import Path
from collections import defaultdict
import random

label = Path("Dataset/Label.txt")

lines = label.read_text(encoding="utf-8").splitlines()

# Group front/back by person (card_xxx)
cards = defaultdict(list)

for line in lines:
    image_path = line.split("\t", 1)[0]

    # card_005/back.jpg -> card_005
    card_id = Path(image_path).parts[0]

    cards[card_id].append(line)

# Shuffle persons, NOT images
card_ids = list(cards.keys())

random.seed(42)
random.shuffle(card_ids)

split = int(len(card_ids) * 0.8)

train_cards = card_ids[:split]
val_cards = card_ids[split:]

# Restore all front/back images for each person
train = [
    line
    for card_id in train_cards
    for line in cards[card_id]
]

val = [
    line
    for card_id in val_cards
    for line in cards[card_id]
]

Path("Dataset/train.txt").write_text(
    "\n".join(train) + "\n",
    encoding="utf-8"
)

Path("Dataset/val.txt").write_text(
    "\n".join(val) + "\n",
    encoding="utf-8"
)

print("Total persons:", len(cards))
print("Train persons:", len(train_cards))
print("Val persons:", len(val_cards))

print("Train images:", len(train))
print("Val images:", len(val))

print("\nTrain cards:")
print(train_cards)

print("\nVal cards:")
print(val_cards)