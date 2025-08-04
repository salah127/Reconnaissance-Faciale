"""
Script de reconnaissance faciale - Test avec modèle pré-entraîné
Utilise le modèle modele_user_0_equilibre.h5 pour reconnaître la personne 0
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
import os

def charger_modele(chemin_modele="modele_user_0_equilibre.h5"):
    """
    Charge le modèle pré-entraîné
    
    Args:
        chemin_modele: chemin vers le fichier .h5
    
    Returns:
        model: modèle Keras chargé
    """
    try:
        model = load_model(chemin_modele)
        print(f" Modèle chargé: {chemin_modele}")
        return model
    except FileNotFoundError:
        print(f" Modèle non trouvé: {chemin_modele}")
        print(" Assurez-vous d'avoir exécuté guide_complet.py d'abord")
        return None
    except Exception as e:
        print(f" Erreur lors du chargement: {e}")
        return None

def charger_seuil_optimal(user_id=0):
    """
    Charge le seuil optimal depuis le fichier JSON généré par guide_complet.py
    
    Args:
        user_id: ID de l'utilisateur
    
    Returns:
        float: seuil optimal ou 0.7 par défaut
    """
    try:
        import json
        with open(f'seuil_user_{user_id}.json', 'r') as f:
            seuil_info = json.load(f)
        
        seuil = seuil_info['seuil_optimal']
        print(f" Seuil optimal chargé: {seuil:.3f}")
        print(f"   (Basé sur: P0={seuil_info['moyenne_p0']:.3f}, Autres={seuil_info['moyenne_autres']:.3f})")
        return seuil
        
    except FileNotFoundError:
        print(f"  Fichier seuil non trouvé, utilisation du seuil par défaut: 0.7")
        return 0.7
    except Exception as e:
        print(f"  Erreur chargement seuil: {e}, utilisation du seuil par défaut: 0.7")
        return 0.7

def preparer_image(chemin_image):
    """
    Prépare une image pour la prédiction
    
    Args:
        chemin_image: chemin vers l'image à tester
    
    Returns:
        image_preparee: image au format attendu par le modèle
        image_originale: image pour affichage
    """
    try:
        # Charger l'image
        image = Image.open(chemin_image)
        
        # Convertir en niveaux de gris si couleur
        if image.mode != 'L':
            image = image.convert('L')
        
        # Redimensionner à 64x64
        image = image.resize((64, 64))
        
        # Convertir en array numpy
        image_array = np.array(image)
        
        # Normaliser (0-255 → 0-1)
        image_array = image_array / 255.0
        
        # Reformater pour le CNN: (1, 64, 64, 1)
        image_preparee = image_array.reshape(1, 64, 64, 1)
        
        return image_preparee, image_array
        
    except FileNotFoundError:
        print(f" Image non trouvée: {chemin_image}")
        return None, None
    except Exception as e:
        print(f" Erreur lors du traitement de l'image: {e}")
        return None, None

def predire_personne(model, image_preparee, seuil=None):  # Seuil automatique par défaut
    """
    Fait une prédiction sur l'image
    
    Args:
        model: modèle Keras
        image_preparee: image au bon format
        seuil: seuil de décision (si None, charge automatiquement)
    
    Returns:
        dict: résultats de la prédiction
    """
    # Charger le seuil optimal si non fourni
    if seuil is None:
        seuil = charger_seuil_optimal(user_id=0)
    
    # Prédiction
    probabilite = model.predict(image_preparee, verbose=0)[0][0]
    
    # Décision
    est_utilisateur = probabilite > seuil
    
    # Niveau de confiance
    if est_utilisateur:
        confiance = probabilite
        resultat = " C'EST LA PERSONNE 0!"
    else:
        confiance = 1 - probabilite
        resultat = " CE N'EST PAS LA PERSONNE 0"
    
    return {
        'probabilite': probabilite,
        'confiance': confiance,
        'est_utilisateur': est_utilisateur,
        'resultat': resultat,
        'seuil': seuil  # Ajout du seuil utilisé
    }

def tester_image(chemin_image, afficher=True):
    """
    Fonction principale pour tester une image
    
    Args:
        chemin_image: chemin vers l'image
        afficher: afficher l'image et les résultats
    
    Returns:
        dict: résultats complets
    """
    print(f"\n TEST DE RECONNAISSANCE")
    print(f" Image: {chemin_image}")
    print("-" * 50)
    
    # 1. Charger le modèle
    model = charger_modele()
    if model is None:
        return None
    
    # 2. Préparer l'image
    image_preparee, image_affichage = preparer_image(chemin_image)
    if image_preparee is None:
        return None
    
    # 3. Faire la prédiction
    resultats = predire_personne(model, image_preparee)
    
    # 4. Afficher les résultats
    print(f" {resultats['resultat']}")
    print(f" Probabilité: {resultats['probabilite']:.3f} ({resultats['probabilite']*100:.1f}%)")
    print(f" Confiance: {resultats['confiance']:.3f} ({resultats['confiance']*100:.1f}%)")
    
    # 5. Affichage graphique
    if afficher and image_affichage is not None:
        plt.figure(figsize=(8, 6))
        
        # Image
        plt.subplot(1, 2, 1)
        plt.imshow(image_affichage, cmap='gray')
        plt.title(f'Image testée\n{os.path.basename(chemin_image)}')
        plt.axis('off')
        
        # Résultats
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.8, resultats['resultat'], fontsize=16, weight='bold')
        plt.text(0.1, 0.6, f"Probabilité: {resultats['probabilite']:.3f}", fontsize=12)
        plt.text(0.1, 0.4, f"Confiance: {resultats['confiance']*100:.1f}%", fontsize=12)
        
        # Afficher le seuil utilisé
        seuil_utilise = resultats.get('seuil', 0.7)
        plt.text(0.1, 0.2, f"Seuil utilisé: {seuil_utilise:.3f}", fontsize=10)
        
        # Barre de probabilité
        couleur = 'green' if resultats['est_utilisateur'] else 'red'
        plt.barh(0.05, resultats['probabilite'], height=0.05, color=couleur, alpha=0.7)
        
        # Ligne du seuil
        plt.axvline(x=seuil_utilise, color='black', linestyle='--', alpha=0.7)
        plt.text(seuil_utilise, 0.01, f"Seuil", ha='center', fontsize=8)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Résultats de reconnaissance')
        
        plt.tight_layout()
        plt.show()
    
    return resultats

def tester_depuis_dataset(index_image=0):
    """
    Teste avec une image du dataset original
    
    Args:
        index_image: index de l'image dans le dataset (0-399)
    """
    try:
        # Charger le dataset
        data = np.load("data/olivetti_faces.npy")
        
        if index_image >= len(data):
            print(f" Index trop grand. Max: {len(data)-1}")
            return
        
        # Extraire l'image
        image = data[index_image]
        
        # Sauvegarder temporairement
        temp_path = "temp_test_image.png"
        plt.imsave(temp_path, image, cmap='gray')
        
        # Déterminer la vraie personne
        vraie_personne = index_image // 10
        
        print(f"\n TEST AVEC IMAGE DU DATASET")
        print(f" Index: {index_image}")
        print(f" Vraie personne: {vraie_personne}")
        print(f" Devrait être: {'PERSONNE 0' if vraie_personne == 0 else 'PAS PERSONNE 0'}")
        
        # Tester
        resultats = tester_image(temp_path, afficher=True)
        
        # Nettoyer
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return resultats
        
    except FileNotFoundError:
        print(" Dataset non trouvé: data/olivetti_faces.npy")
        return None

def tester_plusieurs_exemples():
    """Teste plusieurs images du dataset pour validation"""
    print("\n TESTS DE VALIDATION")
    print("=" * 60)
    
    # Tests avec images de la personne 0 (devraient être reconnus)
    print("\n TESTS AVEC PERSONNE 0 (devraient être reconnus):")
    for i in [0, 1, 5, 9]:  # Images de la personne 0
        print(f"\n--- Test image {i} ---")
        tester_depuis_dataset(i)
    
    # Tests avec autres personnes (ne devraient PAS être reconnus)
    print("\n TESTS AVEC AUTRES PERSONNES (ne devraient PAS être reconnus):")
    for i in [10, 25, 50, 100]:  # Images d'autres personnes
        print(f"\n--- Test image {i} ---")
        tester_depuis_dataset(i)


def diagnostic_rapide():
    """Diagnostic rapide pour trouver le meilleur seuil"""
    print("\n DIAGNOSTIC RAPIDE DU MODÈLE")
    print("=" * 50)
    
    try:
        # Charger données et modèle
        data = np.load("data/olivetti_faces.npy")
        model = charger_modele()
        if model is None:
            return
        
        print(" Test sur échantillon représentatif...")
        
        # Tester personne 0 (images 0-9)
        print("\n PERSONNE 0:")
        predictions_p0 = []
        for i in range(5):  # Tester 5 images de P0
            image = data[i].reshape(1, 64, 64, 1)
            pred = model.predict(image, verbose=0)[0][0]
            predictions_p0.append(pred)
            print(f"   Image {i}: {pred:.3f}")
        
        # Tester autres personnes
        print(f"\n AUTRES PERSONNES:")
        predictions_autres = []
        test_indices = [10, 25, 50, 100, 200]  # 5 autres personnes
        for i in test_indices:
            image = data[i].reshape(1, 64, 64, 1)
            pred = model.predict(image, verbose=0)[0][0]
            predictions_autres.append(pred)
            pers = i // 10
            print(f" Image {i} (pers.{pers}): {pred:.3f}")
        
        # Statistiques
        moyenne_p0 = np.mean(predictions_p0)
        moyenne_autres = np.mean(predictions_autres)
        
        print(f"\n STATISTIQUES:")
        print(f"   Moyenne P0: {moyenne_p0:.3f}")
        print(f"   Moyenne autres: {moyenne_autres:.3f}")
        
        # Recommandation de seuil
        seuil_optimal = (moyenne_p0 + moyenne_autres) / 2
        if seuil_optimal < 0.5:
            seuil_recommande = 0.5
        elif seuil_optimal > 0.8:
            seuil_recommande = 0.8
        else:
            seuil_recommande = round(seuil_optimal, 1)
        
        print(f"   Seuil optimal calculé: {seuil_optimal:.3f}")
        print(f"   Seuil recommandé: {seuil_recommande}")
        
        # Test avec différents seuils
        print(f"\n TEST AVEC DIFFÉRENTS SEUILS:")
        seuils_test = [0.5, 0.6, 0.7, 0.8]
        
        for seuil in seuils_test:
            correct_p0 = sum(1 for p in predictions_p0 if p > seuil)
            correct_autres = sum(1 for p in predictions_autres if p <= seuil)
            total_correct = correct_p0 + correct_autres
            precision = total_correct / (len(predictions_p0) + len(predictions_autres))
            
            print(f"   Seuil {seuil}: P0={correct_p0}/5, Autres={correct_autres}/5, Précision={precision:.1%}")
        
        return seuil_recommande
        
    except Exception as e:
        print(f" Erreur diagnostic: {e}")
        return 0.7

# === UTILISATION PRINCIPALE ===
if __name__ == "__main__":
    print(" RECONNAISSANCE FACIALE - PERSONNE 0")
    print("=" * 60)
    
    # Vérifier que le modèle existe
    if not os.path.exists("modele_user_0_equilibre.h5"):
        print(" Modèle 'modele_user_0_equilibre.h5' non trouvé!")
        print(" Exécutez d'abord 'guide_complet.py' pour créer le modèle")
        exit()
    
    print("\n OPTIONS DISPONIBLES:")
    print("1. Tester une image externe")
    print("2. Tester une image du dataset")
    print("3. Tests de validation automatiques")
    print("4. Diagnostic rapide du modèle")  # ← NOUVEAU
    
    choix = input("\n Votre choix (1/2/3/4): ").strip()
    
    if choix == "1":
        # Test image externe
        chemin = input(" Chemin vers l'image: ").strip()
        if chemin:
            tester_image(chemin)
        
    elif choix == "2":
        # Test image du dataset
        try:
            index = int(input(" Index de l'image (0-399): "))
            tester_depuis_dataset(index)
        except ValueError:
            print(" Veuillez entrer un nombre valide")
            
    elif choix == "3":
        # Tests automatiques
        tester_plusieurs_exemples()
        
    elif choix == "4":  # ← NOUVEAU
        # Diagnostic rapide
        seuil_recommande = diagnostic_rapide()
        print(f"\n CONSEIL: Changez le seuil dans predire_personne() à {seuil_recommande}")
        
    else:
        print(" Choix invalide")
        
    print("\n Test terminé!")
