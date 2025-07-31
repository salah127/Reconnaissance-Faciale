"""
Modèle CNN simple pour la reconnaissance faciale binaire
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

def create_simple_cnn(input_shape=(64, 64, 1)):
    """
    Crée un CNN simple pour la classification binaire
    
    Args:
        input_shape: forme des images d'entrée
    
    Returns:
        model: modèle Keras compilé
    """
    
    model = Sequential([
        # Première couche convolutionnelle
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Deuxième couche convolutionnelle  
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Troisième couche convolutionnelle
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Aplatir et couches denses
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Sortie binaire
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_for_cnn(X_train, X_val, X_test):
    """
    Prépare les données pour le CNN (ajout dimension des canaux)
    
    Args:
        X_train, X_val, X_test: données aplaties
    
    Returns:
        Données reformatées pour le CNN
    """
    
    # Redimensionner pour le CNN (64, 64, 1)
    X_train_cnn = X_train.reshape(-1, 64, 64, 1)
    X_val_cnn = X_val.reshape(-1, 64, 64, 1)  
    X_test_cnn = X_test.reshape(-1, 64, 64, 1)
    
    return X_train_cnn, X_val_cnn, X_test_cnn

if __name__ == "__main__":
    # Exemple d'utilisation
    model = create_simple_cnn()
    print("Modèle CNN simple créé:")
    model.summary()
