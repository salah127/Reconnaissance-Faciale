"""
SYSTÈME DE RECONNAISSANCE FACIALE BINAIRE
==========================================

Objectif: Développer un système de reconnaissance faciale avec CNN qui:
- Analyse des images en entrée (64x64 pixels en niveaux de gris)
- Retourne un résultat binaire: 1 si l'image correspond à l'utilisateur, 0 sinon
- Utilise l'augmentation de données pour équilibrer le dataset déséquilibré
- Produit UN SEUL modèle final optimisé avec seuil adaptatif

Architecture: CNN avec couches convolutionnelles + pooling + dense
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# === ÉTAPE 1: CHARGER LES DONNÉES ===
print(" Chargement des données...")

data = np.load("./data/my_faces_augmented.npy")
print(f" Données chargées: {data.shape}")
print(f"   Range: {data.min():.3f} - {data.max():.3f}")

# === ÉTAPE 2: CRÉER LES TARGETS À PARTIR DES DONNÉES ===
print("\n ÉTAPE 2: Création des labels automatiques...")

USER_ID = 0  # Choisir quelle personne reconnaître (0 à 39)

def create_binary_labels(total_images=400, images_per_person=10, target_person_id=0):
    """Crée les labels binaires basés sur la structure du dataset"""
    y_binary = np.zeros(total_images)
    start_idx = target_person_id * images_per_person
    end_idx = start_idx + images_per_person
    y_binary[start_idx:end_idx] = 1
    return y_binary.astype(int)

# Créer les labels
y_binary = create_binary_labels(target_person_id=USER_ID)

print(f" Labels créés automatiquement:")
print(f"   Personne {USER_ID}: images {USER_ID*10} à {USER_ID*10+9} = Utilisateur (1)")
print(f"   Total images utilisateur: {np.sum(y_binary)} ({np.sum(y_binary)/len(y_binary)*100:.1f}%)")
print(f"   Total images autres: {len(y_binary) - np.sum(y_binary)} ({(len(y_binary) - np.sum(y_binary))/len(y_binary)*100:.1f}%)")
print(f"  DÉSÉQUILIBRE: 2.5% utilisateur vs 97.5% autres!")

# === ÉTAPE 3: VISUALISER QUELQUES EXEMPLES ===
print("\n Visualisation des données...")

def show_user_vs_others(data, labels, user_id=0):
    """Montre quelques images de l'utilisateur vs autres"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Images de l'utilisateur (première ligne)
    user_indices = np.where(labels == 1)[0][:5]
    for i, idx in enumerate(user_indices):
        axes[0, i].imshow(data[idx], cmap='gray')
        axes[0, i].set_title(f'Utilisateur {user_id}\n(Image {idx})')
        axes[0, i].axis('off')
    
    # Images d'autres personnes (deuxième ligne)  
    other_indices = np.where(labels == 0)[0][:5]
    for i, idx in enumerate(other_indices):
        axes[1, i].imshow(data[idx], cmap='gray')
        person_id = idx // 10
        axes[1, i].set_title(f'Personne {person_id}\n(Image {idx})')
        axes[1, i].axis('off')
    
    plt.suptitle(f'UTILISATEUR {user_id} vs AUTRES PERSONNES', fontsize=16)
    plt.tight_layout()
    plt.show()

# Afficher les exemples
show_user_vs_others(data, y_binary, USER_ID)

print("\n Équilibrage des données...")




def equilibrer_donnees(data, labels, methode="sur_echantillonnage_intelligent"):
    """
    Équilibre les classes avec augmentation de données intelligente
    
    Args:
        data: images
        labels: étiquettes
        methode: "sous_echantillonnage", "sur_echantillonnage", ou "sur_echantillonnage_intelligent"
    
    Returns:
        data_eq, labels_eq: données équilibrées
    """
    from sklearn.utils import resample
    
    # Indices des classes
    indices_user = np.where(labels == 1)[0]  # 10 images
    indices_others = np.where(labels == 0)[0]  # 390 images
    
    print(f"   Avant équilibrage:")
    print(f"   - Utilisateur: {len(indices_user)} images")
    print(f"   - Autres: {len(indices_others)} images")
    
    if methode == "sous_echantillonnage":
        # Prendre plus d'images d'autres personnes pour avoir assez de données
        np.random.seed(42)
        indices_others_reduits = np.random.choice(indices_others, size=len(indices_user)*6, replace=False)
        
        # Combiner
        indices_finaux = np.concatenate([indices_user, indices_others_reduits])
        
    elif methode == "sur_echantillonnage_intelligent":
        # Stratégie intelligente: dupliquer P0 et prendre plus d'autres personnes
        np.random.seed(42)
        
        # Dupliquer les images P0 avec transformations légères
        def augmenter_image(img):
            """Applique des transformations légères à une image"""
            try:
                # Essayer avec scipy si disponible
                from scipy.ndimage import rotate
                angle = np.random.uniform(-10, 10)
                img_transformed = rotate(img, angle, reshape=False, cval=img.mean())
            except ImportError:
                # Alternative sans scipy : translation légère
                shift_x = np.random.randint(-3, 4)
                shift_y = np.random.randint(-3, 4)
                img_transformed = np.roll(img, shift_x, axis=0)
                img_transformed = np.roll(img_transformed, shift_y, axis=1)
            
            # Bruit léger
            noise = np.random.normal(0, 0.01, img_transformed.shape)  # Moins de bruit
            img_noise = np.clip(img_transformed + noise, 0, 1)
            
            # Variation de contraste légère
            contrast_factor = np.random.uniform(0.9, 1.1)
            img_final = np.clip(img_noise * contrast_factor, 0, 1)
            
            return img_final
        
        # Créer plus d'exemples de P0 avec augmentation
        data_user_augmented = []
        labels_user_augmented = []
        
        # Garder les originaux
        for idx in indices_user:
            data_user_augmented.append(data[idx])
            labels_user_augmented.append(1)
        
        # Ajouter des versions augmentées (4x plus)
        for _ in range(4):
            for idx in indices_user:
                img_augmented = augmenter_image(data[idx])
                data_user_augmented.append(img_augmented)
                labels_user_augmented.append(1)
        
        # Prendre un échantillon d'autres personnes (même quantité que P0 augmenté)
        nb_autres_needed = len(data_user_augmented)
        indices_others_selected = np.random.choice(indices_others, size=nb_autres_needed, replace=False)
        
        # Combiner toutes les données
        data_finale = data_user_augmented + [data[idx] for idx in indices_others_selected]
        labels_finale = labels_user_augmented + [0] * len(indices_others_selected)
        
        # Convertir en arrays
        data_eq = np.array(data_finale)
        labels_eq = np.array(labels_finale)
        
        # Mélanger
        shuffle_indices = np.random.permutation(len(data_eq))
        data_eq = data_eq[shuffle_indices]
        labels_eq = labels_eq[shuffle_indices]
        
        print(f"   Après équilibrage ({methode}):")
        print(f"   - Utilisateur: {np.sum(labels_eq)} images ({np.sum(labels_eq)/len(labels_eq)*100:.1f}%)")
        print(f"   - Autres: {len(labels_eq) - np.sum(labels_eq)} images ({(len(labels_eq) - np.sum(labels_eq))/len(labels_eq)*100:.1f}%)")
        print(f"   - Total: {len(data_eq)} images")
        
        return data_eq, labels_eq
        
    elif methode == "sur_echantillonnage":
        # Sur-échantillonnage simple avec resample
        data_user = data[indices_user]
        data_others = data[indices_others]
        labels_user = labels[indices_user]
        labels_others = labels[indices_others]
        
        # Sur-échantillonner les utilisateurs
        data_user_resampled, labels_user_resampled = resample(
            data_user, labels_user, 
            n_samples=len(indices_others), 
            random_state=42
        )
        
        # Combiner
        data_eq = np.concatenate([data_user_resampled, data_others])
        labels_eq = np.concatenate([labels_user_resampled, labels_others])
        
    # Mélanger pour les méthodes simples
    if methode != "sur_echantillonnage_intelligent":
        shuffle_indices = np.random.permutation(len(data_eq))
        data_eq = data_eq[shuffle_indices]
        labels_eq = labels_eq[shuffle_indices]
        
        print(f"   Après équilibrage ({methode}):")
        print(f"   - Utilisateur: {np.sum(labels_eq)} images ({np.sum(labels_eq)/len(labels_eq)*100:.1f}%)")
        print(f"   - Autres: {len(labels_eq) - np.sum(labels_eq)} images ({(len(labels_eq) - np.sum(labels_eq))/len(labels_eq)*100:.1f}%)")
    
    return data_eq, labels_eq





# Équilibrer avec augmentation intelligente
data_eq, y_eq = equilibrer_donnees(data, y_binary, methode="sur_echantillonnage_intelligent")

# === ÉTAPE 5: PRÉPARER LES DONNÉES ÉQUILIBRÉES ===
print("\n Préparation des données équilibrées...")

# Aplatir pour division
X_flat_eq = data_eq.reshape(data_eq.shape[0], -1)

# Division train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_flat_eq, y_eq, 
    test_size=0.3,  # Plus de test car moins de données
    random_state=42,
    stratify=y_eq
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=0.3, 
    random_state=42,
    stratify=y_temp
)

print(f" Données équilibrées préparées:")
print(f"   Train: {X_train.shape} - Utilisateur: {np.sum(y_train)}/{len(y_train)} ({np.sum(y_train)/len(y_train)*100:.1f}%)")
print(f"   Val: {X_val.shape} - Utilisateur: {np.sum(y_val)}/{len(y_val)} ({np.sum(y_val)/len(y_val)*100:.1f}%)")
print(f"   Test: {X_test.shape} - Utilisateur: {np.sum(y_test)}/{len(y_test)} ({np.sum(y_test)/len(y_test)*100:.1f}%)")

# === ÉTAPE 6: REFORMATER POUR LE CNN ===
print("\n🔄 ÉTAPE 6: Reformatage pour CNN...")

def prepare_data_for_cnn(X_train, X_val, X_test):
    """Transforme les données pour le CNN"""
    X_train_cnn = X_train.reshape(-1, 64, 64, 1)
    X_val_cnn = X_val.reshape(-1, 64, 64, 1)  
    X_test_cnn = X_test.reshape(-1, 64, 64, 1)
    return X_train_cnn, X_val_cnn, X_test_cnn

X_train_cnn, X_val_cnn, X_test_cnn = prepare_data_for_cnn(X_train, X_val, X_test)

print(f" Données reformatées pour CNN:")
print(f"   X_train_cnn: {X_train_cnn.shape}, range: {X_train_cnn.min():.3f}-{X_train_cnn.max():.3f}")

# === ÉTAPE 7: CRÉER ET COMPILER LE MODÈLE ===
print("\n ÉTAPE 7: Création du modèle...")

try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from keras.optimizers import Adam
    
    def create_improved_cnn():
        """Crée un CNN amélioré pour reconnaissance faciale"""
        model = Sequential([
            # Première couche avec normalisation
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Deuxième bloc
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Troisième bloc
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Couches denses
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Compilation avec optimiseur adaptatif
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Learning rate plus faible
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    # Créer le modèle amélioré
    model = create_improved_cnn()
    
    print(" Modèle créé et compilé!")
    print("\n Architecture du modèle:")
    model.summary()
    
    # === ÉTAPE 8: ENTRAÎNEMENT ÉQUILIBRÉ ===
    print("\n ÉTAPE 8: Entraînement du modèle équilibré...")
    
    # Vérification
    print(f" Vérification avant entraînement:")
    print(f"   Shape entrée: {X_train_cnn.shape}")
    print(f"   Labels distribution: {np.unique(y_train, return_counts=True)}")
    
    # Entraînement avec callbacks pour éviter le sur-apprentissage
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Plus de patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Amélioration minimale
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
        # Suppression du ModelCheckpoint - on garde seulement le modèle final
    ]
    
    # Calculer les poids des classes pour l'entraînement
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    print(f" Poids des classes: {class_weight_dict}")
    
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_val_cnn, y_val),
        epochs=100,  # Plus d'époques avec early stopping
        batch_size=16,  # Batch size adapté
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight_dict  # Utiliser les poids équilibrés
    )
    
    print(" Entraînement terminé!")
    
    # === ÉTAPE 9: ÉVALUATION COMPLÈTE AVEC CALCUL DE SEUIL OPTIMAL ===
    print("\n ÉTAPE 9: Évaluation complète...")
    
    # Test sur toutes les images de la personne 0 du dataset original
    print(f"\n Test sur TOUTES les images de la personne 0:")
    predictions_p0 = []
    for i in range(10):
        test_img = data[i].reshape(1, 64, 64, 1)
        pred = model.predict(test_img, verbose=0)[0][0]
        predictions_p0.append(pred)
        print(f"   Image {i}: {pred:.3f}")
    
    # Test sur autres personnes
    print(f"\n Test sur autres personnes:")
    predictions_autres = []
    test_autres = [10, 25, 50, 100, 200, 300]  # Plus d'exemples
    for i in test_autres:
        test_img = data[i].reshape(1, 64, 64, 1)
        pred = model.predict(test_img, verbose=0)[0][0]
        predictions_autres.append(pred)
        personne = i // 10
        print(f"   Image {i} (pers.{personne}): {pred:.3f}")
    
    # Calcul des statistiques
    moyenne_p0 = np.mean(predictions_p0)
    moyenne_autres = np.mean(predictions_autres)
    std_p0 = np.std(predictions_p0)
    std_autres = np.std(predictions_autres)
    
    print(f"\n STATISTIQUES DES PRÉDICTIONS:")
    print(f"   Personne 0: moyenne={moyenne_p0:.3f}, std={std_p0:.3f}")
    print(f"   Autres: moyenne={moyenne_autres:.3f}, std={std_autres:.3f}")
    
    # Calcul du seuil optimal plus intelligent
    seuil_milieu = (moyenne_p0 + moyenne_autres) / 2
    
    # Si les distributions se chevauchent beaucoup, utiliser un seuil plus bas
    if abs(moyenne_p0 - moyenne_autres) < 0.1:
        seuil_optimal = min(moyenne_p0 - std_p0/2, 0.3)  # Seuil plus bas
        print("  Distributions très proches, seuil abaissé")
    else:
        seuil_optimal = max(0.3, min(0.7, seuil_milieu))  # Seuil normal
    
    print(f"   Seuil calculé (milieu): {seuil_milieu:.3f}")
    print(f"   Seuil optimal choisi: {seuil_optimal:.3f}")
    
    # Test avec une plage de seuils plus large
    print(f"\n ÉVALUATION AVEC DIFFÉRENTS SEUILS:")
    seuils_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    meilleur_seuil = 0.3
    meilleure_precision = 0
    meilleur_f1 = 0
    
    for seuil in seuils_test:
        # Test P0
        correct_p0 = sum(1 for p in predictions_p0 if p > seuil)
        # Test autres
        correct_autres = sum(1 for p in predictions_autres if p <= seuil)
        
        # Calcul des métriques
        true_positives = correct_p0
        false_negatives = 10 - correct_p0
        true_negatives = correct_autres
        false_positives = len(predictions_autres) - correct_autres
        
        # Précision et rappel
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        accuracy = (true_positives + true_negatives) / (10 + len(predictions_autres))
        
        print(f"   Seuil {seuil}: P0={correct_p0}/10 ({correct_p0*10}%), Autres={correct_autres}/{len(predictions_autres)}, Acc={accuracy:.1%}, F1={f1_score:.3f}")
        
        # Critère amélioré: F1-score élevé ET au moins 60% de détection P0
        if correct_p0 >= 6 and f1_score > meilleur_f1:
            meilleur_f1 = f1_score
            meilleure_precision = accuracy
            meilleur_seuil = seuil
    
    print(f"\n MEILLEUR SEUIL TROUVÉ: {meilleur_seuil}")
    
    # Évaluation finale avec le meilleur seuil
    bonnes_detections = sum(1 for p in predictions_p0 if p > meilleur_seuil)
    bonnes_rejections = sum(1 for p in predictions_autres if p <= meilleur_seuil)
    
    print(f"\n RÉSULTATS AVEC SEUIL OPTIMAL ({meilleur_seuil}):")
    print(f"   Détection personne 0: {bonnes_detections}/10 ({bonnes_detections*10}%)")
    print(f"   Rejet autres: {bonnes_rejections}/{len(test_autres)} ({bonnes_rejections*100//len(test_autres)}%)")
    
    # Évaluation sur ensemble de test équilibré
    test_results = model.evaluate(X_test_cnn, y_test, verbose=0)
    test_accuracy = test_results[1]  # L'accuracy est le deuxième élément
    print(f"\n Précision sur test équilibré: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    # === ÉTAPE 10: SAUVEGARDE ===
    print("\n ÉTAPE 10: Sauvegarde...")
    
    # Nettoyer les fichiers temporaires s'ils existent
    import os
    if os.path.exists('best_model_temp.h5'):
        os.remove('best_model_temp.h5')
        print(" Fichier temporaire supprimé")
    
    # Sauvegarder le modèle final unique
    model.save(f'./test/modele_user_{USER_ID}.h5')
    print(f" Modèle unique sauvegardé: modele_user_{USER_ID}.h5")
    
    # Sauvegarde des informations de seuil
    import json
    seuil_info = {
        'seuil_optimal': float(meilleur_seuil),
        'moyenne_p0': float(moyenne_p0),
        'moyenne_autres': float(moyenne_autres),
        'std_p0': float(std_p0),
        'std_autres': float(std_autres),
        'precision_avec_seuil': float(meilleure_precision)
    }
    
    with open(f'seuil_user_{USER_ID}.json', 'w') as f:
        json.dump(seuil_info, f, indent=2)
    print(f" Informations de seuil sauvées: seuil_user_{USER_ID}.json")
    
    # Résumé final
    precision_globale = (bonnes_detections + bonnes_rejections) / (10 + len(test_autres)) * 100
    
  
    print(f" Système prêt pour la reconnaissance faciale binaire !")
    
except Exception as e:
    print(f" Erreur: {e}")
    import traceback
    traceback.print_exc()

print(f"\n Système de reconnaissance faciale binaire créé avec succès!")
print(f" Modèle unique disponible: modele_user_{USER_ID}.h5")
print(f" Testez avec: python test.py")