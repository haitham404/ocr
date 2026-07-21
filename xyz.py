import cv2

IMAGE_PATH = "Dataset/card_000/front.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

h, w = image.shape[:2]

# Select ROI
x, y, width, height = cv2.selectROI(
    "Select ROI",
    image,
    showCrosshair=True,
    fromCenter=False,
)

cv2.destroyAllWindows()

x2 = x + width
y2 = y + height

print("=" * 50)
print("Pixel Coordinates")
print(f"x1 = {x}")
print(f"y1 = {y}")
print(f"x2 = {x2}")
print(f"y2 = {y2}")

print("\nNormalized Coordinates")
print(f"x1 = {x / w:.4f}")
print(f"y1 = {y / h:.4f}")
print(f"x2 = {x2 / w:.4f}")
print(f"y2 = {y2 / h:.4f}")

crop = image[y:y2, x:x2]

cv2.imshow("Crop", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()