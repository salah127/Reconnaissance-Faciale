import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convertir en niveaux de gris
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Normalisation adaptative
    enhanced_float = enhanced.astype(float)
    min_val = np.percentile(enhanced_float, 5)  # Ignorer les valeurs extrêmes
    max_val = np.percentile(enhanced_float, 95)
    normalized = np.clip((enhanced_float - min_val) / (max_val - min_val), 0, 1)
    
    return (normalized * 255).astype(np.uint8)

# === Paramètres ===
input_folder = os.path.join(os.path.dirname(__file__), "d_goat")
target_size = (64, 64)
desired_total = 50

# Paramètres de traitement d'image
MIN_BRIGHTNESS = 5  # Luminosité minimale moyenne
CONTRAST_CLIP_LIMIT = 3.0  # Limite pour l'égalisation adaptative d'histogramme

# === Étape 1 : Charger les 3 images personnelles ===
images = []

for fname in os.listdir(input_folder):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            path = os.path.join(input_folder, fname)
            img_color = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_color is None:
                print(f"❌ Impossible de lire l'image : {fname}")
                continue

            img_resized = cv2.resize(img_color, target_size)
            img_processed = preprocess_image(img_resized)
            img_final = img_processed.astype("float32") / 255.0
            
            # Vérification simple de la qualité
            if np.mean(img_final) < 0.1:
                print(f"⚠️ Image ignorée (trop sombre) : {fname}")
                continue
                
            images.append(img_final)
            print(f"✅ Image chargée : {fname}")
            
        except Exception as e:
            print(f"❌ Erreur : {fname} - {str(e)}")

if not images:
    print("❌ Aucune image valide trouvée dans le dossier my_faces")
    exit(1)

images = np.array(images).reshape(-1, 64, 64, 1)
print(f"\n✅ Total des images chargées : {len(images)}")

# === Étape 2 : Créer un générateur de données augmentées ===
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=None,           # Pas de rescale ici car on le fait manuellement
    brightness_range=None,   # On va gérer la luminosité nous-mêmes
    horizontal_flip=True,
    fill_mode='constant',
    cval=1.0                # Valeur blanche pour le remplissage (1.0 car on est déjà normalisé)
)

# === Étape 3 : Générer les images augmentées ===
augmented = []
print("\n🔄 Génération des images augmentées...")

num_augmented_per_image = desired_total // len(images)
for idx, image in enumerate(images):
    # Générer des variations pour chaque image
    for i in range(num_augmented_per_image):
        # Générer une variation
        augmented_img = datagen.random_transform(image)
        
        # S'assurer que l'image n'est pas trop sombre
        if np.mean(augmented_img) < 0.2:
            # Ajuster la luminosité si nécessaire
            augmented_img = np.clip(augmented_img * 1.5, 0, 1)
        
        augmented.append(augmented_img)
        
        # # Sauvegarder quelques exemples pour vérification
        # if idx == 0 and i < 3:  # Sauvegarder 3 exemples de la première image
        #     debug_path = os.path.join(os.path.dirname(__file__), f"debug_augmented_{i+1}.jpg")
        #     cv2.imwrite(debug_path, (augmented_img.squeeze() * 255).astype(np.uint8))

augmented = np.array(augmented)
print("✅ Dataset final :", augmented.shape)

# Vérifier la distribution des valeurs
print(f"📊 Statistiques du dataset:")
print(f"    Min: {augmented.min():.3f}")
print(f"    Max: {augmented.max():.3f}")
print(f"    Moyenne: {augmented.mean():.3f}")
print(f"    Écart-type: {augmented.std():.3f}")

# === Étape 4 : Sauvegarde
output_file = os.path.join(os.path.dirname(__file__), "dgoat_faces_augmented.npy")
np.save(output_file, augmented)
print(f"💾 Fichier sauvegardé : {output_file}")
