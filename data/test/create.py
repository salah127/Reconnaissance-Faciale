import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced_float = enhanced.astype(float)
    min_val = np.percentile(enhanced_float, 5)
    max_val = np.percentile(enhanced_float, 95)
    normalized = np.clip((enhanced_float - min_val) / (max_val - min_val), 0, 1)
    return (normalized * 255).astype(np.uint8)

# === Paramètres ===
input_image_path = os.path.join(os.path.dirname(__file__), "test7.png")
target_size = (64, 64)

if not os.path.isfile(input_image_path):
    print(f"❌ Fichier introuvable : {input_image_path}")
    exit(1)

img_color = cv2.imdecode(np.fromfile(input_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if img_color is None:
    print(f"❌ Impossible de lire l'image : {input_image_path}")
    exit(1)

img_resized = cv2.resize(img_color, target_size)
img_processed = preprocess_image(img_resized)
img_final = img_processed.astype("float32") / 255.0
img_final = img_final.reshape(1, 64, 64, 1)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=None,
    brightness_range=None,
    horizontal_flip=True,
    fill_mode='constant',
    cval=1.0
)

augmented_img = datagen.random_transform(img_final[0])

if np.mean(augmented_img) < 0.2:
    augmented_img = np.clip(augmented_img * 1.5, 0, 1)

output_path = os.path.join(os.path.dirname(__file__), "augmented5.jpg")
cv2.imwrite(output_path, (augmented_img.squeeze() * 255).astype(np.uint8))
print(f"✅ Image augmentée sauvegardée sous : {output_path}")