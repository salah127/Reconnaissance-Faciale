"""
Métriques simples pour évaluer le modèle de reconnaissance faciale
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Évalue le modèle avec des métriques simples
    
    Args:
        y_true: vraies étiquettes (0 ou 1)
        y_pred: prédictions (0 ou 1) 
        y_pred_proba: probabilités de prédiction (optionnel)
    """
    
    print("=== ÉVALUATION DU MODÈLE ===")
    
    # Métriques de base
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Précision (Accuracy): {accuracy:.3f}")
    print(f"Précision (Precision): {precision:.3f}")
    print(f"Rappel (Recall): {recall:.3f}")
    print(f"Score F1: {f1:.3f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatrice de confusion:")
    print(f"Vrais Autres: {cm[0,0]}, Faux Utilisateur: {cm[0,1]}")
    print(f"Faux Autres: {cm[1,0]}, Vrais Utilisateur: {cm[1,1]}")
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(y_true, y_pred):
    """Affiche la matrice de confusion"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.colorbar()
    
    classes = ['Autres', 'Utilisateur']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Ajouter les valeurs dans les cases
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.show()

def simple_evaluation(model, X_test, y_test):
    """
    Évaluation simple d'un modèle Keras
    
    Args:
        model: modèle Keras entraîné
        X_test: données de test
        y_test: vraies étiquettes de test
    """
    
    # Prédictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Évaluation
    results = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Affichage graphique
    plot_confusion_matrix(y_test, y_pred)
    
    return results
