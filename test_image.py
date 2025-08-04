import cv2
import numpy as np
import tensorflow as tf
import json

MODEL_PATH = "best_model_user_final.keras"
SEUIL_PATH = "seuil_user.json"
IMAGE_PATH = "./data/test/test11.png"  #utilisateur
# IMAGE_PATH = "./data/test/test9.png" #non utilisateur

# Charger modèle et seuil
model = tf.keras.models.load_model(MODEL_PATH)
with open(SEUIL_PATH, "r") as f:
    seuil = json.load(f)["seuil_optimal"]

# Charger et préparer l'image
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"❌ Erreur : impossible de lire l'image {IMAGE_PATH}")
    exit(1)
img_resized = cv2.resize(img, (64, 64))
img_normalized = img_resized.astype("float32") / 255.0
img_input = img_normalized.reshape(1, 64, 64, 1)

# Prédiction
score = model.predict(img_input, verbose=0)[0][0]
is_user = score > seuil
print(f" Score prédictif : {score:.3f}")
print(" Résultat :", "UTILISATEUR reconnu " if is_user else "Autre personne")