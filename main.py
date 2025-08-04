"""
Système de reconnaissance faciale binaire
=========================================
Script principal pour entrainer un modèle qui distingue une personne spécifique des autres.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os
from datetime import datetime

# Chargement et préparation des données
olivetti_faces = np.load("./data/olivetti_faces.npy")
my_faces = np.load("./data/dgoat_faces_augmented.npy")

if olivetti_faces.max() > 1.0:
    olivetti_faces = olivetti_faces / 255.0
if my_faces.max() > 1.0:
    my_faces = my_faces / 255.0

if len(olivetti_faces.shape) == 3:
    olivetti_faces = olivetti_faces.reshape(-1, 64, 64, 1)
if len(my_faces.shape) == 3:
    my_faces = my_faces.reshape(-1, 64, 64, 1)

print(f"olivetti_faces: {olivetti_faces.shape}, min={olivetti_faces.min():.3f}, max={olivetti_faces.max():.3f}")
print(f"my_faces: {my_faces.shape}, min={my_faces.min():.3f}, max={my_faces.max():.3f}")

y_olivetti = np.zeros(len(olivetti_faces), dtype=np.uint8)
y_my_faces = np.ones(len(my_faces), dtype=np.uint8)

x_all = np.concatenate([olivetti_faces, my_faces], axis=0)
y_all = np.concatenate([y_olivetti, y_my_faces], axis=0)

print(f"Images Olivetti: {olivetti_faces.shape}")
print(f"Mes images: {my_faces.shape}")
print(f"Total: {x_all.shape}")
print(f"Distribution: {np.sum(y_all)} positifs, {len(y_all)-np.sum(y_all)} négatifs")

plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(my_faces[i].squeeze(), cmap='gray')
    plt.title('User')
    plt.axis('off')
    plt.subplot(2, 5, i + 6)
    plt.imshow(olivetti_faces[i].squeeze(), cmap='gray')
    plt.title('Other')
    plt.axis('off')
plt.suptitle('5 user (haut), 5 autres (bas)')
plt.tight_layout()
plt.show()

# Test d'apprentissage sur mini-dataset équilibré
mini_user = my_faces[:20]
mini_other = olivetti_faces[:20]
mini_x = np.concatenate([mini_user, mini_other], axis=0)
mini_y = np.array([1] * 20 + [0] * 20)
idx = np.random.permutation(40)
mini_x = mini_x[idx]
mini_y = mini_y[idx]

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

mini_model = create_model()
mini_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
mini_model.fit(mini_x, mini_y, epochs=20, batch_size=4, verbose=2)
mini_pred = mini_model.predict(mini_x)
mini_pred_bin = (mini_pred > 0.5).astype(int).flatten()
acc = np.mean(mini_pred_bin == mini_y)
print(f"Mini-set accuracy: {acc:.2f}")
if acc < 0.8:
    print("Le modèle n'arrive pas à sur-apprendre le mini-set. Vérifiez les images et labels.")
else:
    print("Le modèle sur-apprend le mini-set. Les données semblent correctes.")

# Division du dataset
x_temp, x_test, y_temp, y_test = train_test_split(
    x_all, y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)
print(f"Train: {x_train.shape} ({np.sum(y_train)} positifs)")
print(f"Validation: {x_val.shape} ({np.sum(y_val)} positifs)")
print(f"Test: {x_test.shape} ({np.sum(y_test)} positifs)")

# Création du modèle
model = create_model()

# Compilation du modèle
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()

# Callbacks
log_dir = os.path.join("logs4", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
callbacks = [
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model_user_final.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Calcul des class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

print(f"Distribution y_train: {np.bincount(y_train.astype(int))}")
print(f"Distribution y_val: {np.bincount(y_val.astype(int))}")
print(f"Distribution y_test: {np.bincount(y_test.astype(int))}")

# Entrainement du modèle
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation du modèle
my_faces_pred = model.predict(my_faces)
mean_my = np.mean(my_faces_pred)
std_my = np.std(my_faces_pred)

other_indices = [10, 20, 100, 300]
other_faces = olivetti_faces[other_indices]
other_faces_pred = model.predict(other_faces)
mean_other = np.mean(other_faces_pred)
std_other = np.std(other_faces_pred)

print("Statistiques des prédictions :")
print(f"Mes images: moyenne={mean_my:.3f}, std={std_my:.3f}")
print(f"Autres: moyenne={mean_other:.3f}, std={std_other:.3f}")

test_img_user = my_faces[0].reshape(1, 64, 64, 1)
test_img_other = olivetti_faces[0].reshape(1, 64, 64, 1)
pred_user = model.predict(test_img_user)[0][0]
pred_other = model.predict(test_img_other)[0][0]
print(f"Prédiction image utilisateur: {pred_user:.3f}")
print(f"Prédiction image autre: {pred_other:.3f}")

seuil_base = (mean_my + mean_other) / 2
if abs(mean_my - mean_other) < 0.1:
    seuil_optimal = min(mean_my - std_my / 2, 0.3)
else:
    seuil_optimal = max(0.3, min(0.7, seuil_base))
print(f"Seuil optimal calculé: {seuil_optimal:.3f}")

seuils = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
meilleur_f1 = 0
meilleur_seuil = seuil_optimal

for seuil in seuils:
    my_correct = np.sum(my_faces_pred > seuil)
    other_correct = np.sum(other_faces_pred <= seuil)
    precision = my_correct / (my_correct + (len(other_faces_pred) - other_correct))
    recall = my_correct / len(my_faces_pred)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    print(f"Seuil {seuil:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    if f1 > meilleur_f1:
        meilleur_f1 = f1
        meilleur_seuil = seuil

# Sauvegarde du modèle et des informations
model.save('modele_user.keras')
seuil_info = {
    'seuil_optimal': float(meilleur_seuil),
    'moyenne_mes_images': float(mean_my),
    'moyenne_autres': float(mean_other),
    'std_mes_images': float(std_my),
    'std_autres': float(std_other),
    'f1_score': float(meilleur_f1)
}
with open('seuil_user.json', 'w') as f:
    json.dump(seuil_info, f, indent=2)

# Résumé final
final_my_correct = np.sum(my_faces_pred > meilleur_seuil)
final_other_correct = np.sum(other_faces_pred <= meilleur_seuil)
precision_finale = final_my_correct / (final_my_correct + (len(other_faces_pred) - final_other_correct))
recall_final = final_my_correct / len(my_faces_pred)
print("Performances :")
print(f"Précision globale: {(final_my_correct + final_other_correct) / (len(my_faces_pred) + len(other_faces_pred)):.3f}")
print(f"Recall (détection utilisateur): {recall_final:.3f}")
print(f"Précision: {precision_finale:.3f}")
print(f"F1-Score: {meilleur_f1:.3f}")
print("Fichiers sauvegardés :")
print("Modèle: modele_user.keras")
print("Seuil: seuil_user.json")
print(f"Logs: {log_dir}")
print(f"Seuil optimal retenu: {meilleur_seuil:.3f}")

print("Système de reconnaissance faciale prêt à l'utilisation.")
print("Pour utiliser le modèle :")
print("1. Charger le modèle : model = tf.keras.models.load_model('modele_user.keras')")
print("2. Charger le seuil : seuil = json.load(open('seuil_user.json'))['seuil_optimal']")
print("3. Prédire : model.predict(image) > seuil")
