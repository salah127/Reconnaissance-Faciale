# Perceptron Multi-couche utilisant la descente de gradient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Import the Sequential model and Dense layer
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization	# type: ignore
from keras.optimizers import Adam	# type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
# from keras.preprocessing.image import ImageDataGenerator # type: ignore
import mglearn



data=np.load("data/olivetti_faces.npy")
target=np.load("data/olivetti_faces_target.npy")

print("There are {} images in the dataset".format(len(data)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))
print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4]))


print("unique target number:",np.unique(target))

print("unique data number:",np.unique(data))



def show_40_distinct_people(images, unique_ids):
    #Creating 4X10 subplots in  18x9 figure size
    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    #For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr=axarr.flatten()
    
    #iterating over user ids
    for unique_id in unique_ids:
        image_index=unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")
    
    
show_40_distinct_people(data, np.unique(target))


def show_10_faces_of_n_subject(images, subject_ids):
    cols=10# each subject has 10 distinct face images
    rows=(len(subject_ids)*10)/cols #
    rows=int(rows)
    
    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))
    #axarr=axarr.flatten()
    
    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index=subject_id*10 + j
            axarr[i,j].imshow(images[image_index], cmap="gray")
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            axarr[i,j].set_title("face id:{}".format(subject_id))   
            
            
#You can playaround subject_ids to see other people faces
show_10_faces_of_n_subject(images=data, subject_ids=[0,5, 21, 24, 36])


#We reshape images for machine learnig  model
X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
print("X shape:",X.shape)

# === PRÉPARATION SIMPLE POUR RECONNAISSANCE BINAIRE ===

# 1. Choisir l'utilisateur cible (par exemple personne 0)
USER_ID = 0

# 2. Créer les labels binaires : 1 = utilisateur, 0 = autres
y_binary = (target == USER_ID).astype(int)
print(f"Utilisateur choisi: {USER_ID}")
print(f"Images de l'utilisateur: {np.sum(y_binary)}")
print(f"Images d'autres personnes: {len(y_binary) - np.sum(y_binary)}")

# 3. Division simple en 3 ensembles
# Train: 60%, Validation: 20%, Test: 20%
X_temp, X_test, y_temp, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nRépartition des données:")
print(f"Entraînement: {len(X_train)} images")
print(f"Validation: {len(X_val)} images") 
print(f"Test: {len(X_test)} images")

# 4. Normaliser les données
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

print("\nDonnées normalisées et prêtes pour l'entraînement!")

# Garder l'ancien code pour comparaison
X_old = X.copy()
X_train_old, X_test_old, y_train_old, y_test_old=train_test_split(X_old, target, test_size=0.3, stratify=target, random_state=0)
print("X_train shape:",X_train_old.shape)
print("y_train shape:{}".format(y_train_old.shape))




# Visualisation simple de la nouvelle distribution
plt.figure(figsize=(8, 5))
classes = ['Utilisateur', 'Autres']
counts = [np.sum(y_train), len(y_train) - np.sum(y_train)]
plt.bar(classes, counts, color=['blue', 'orange'])
plt.title('Distribution des classes (binaire)')
plt.ylabel('Nombre d\'échantillons')
plt.show()

# Ancienne visualisation (multiclasse)
y_frame=pd.DataFrame()
y_frame['subject ids']=y_train_old
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),title="Number of Samples for Each Classes (Original)")

mglearn.plots.plot_pca_illustration()

# === EXEMPLE D'UTILISATION SIMPLE ===

print("\n" + "="*50)
print("DONNÉES PRÊTES POUR L'ENTRAÎNEMENT!")
print("="*50)
print(f"✅ Utilisateur cible: {USER_ID}")
print(f"✅ Données d'entraînement: {len(X_train)} échantillons") 
print(f"✅ Données de validation: {len(X_val)} échantillons")
print(f"✅ Données de test: {len(X_test)} échantillons")
print(f"✅ Classes: Utilisateur ({np.sum(y_train)}) vs Autres ({len(y_train)-np.sum(y_train)})")

print("\nProchaines étapes:")
print("1. Créer un modèle CNN simple")
print("2. Entraîner le modèle")  
print("3. Évaluer les performances")

# Exemple simple de comment utiliser les données:
print(f"\nExemple d'utilisation:")
print(f"X_train.shape: {X_train.shape}")
print(f"y_train exemple: {y_train[:5]} (1=utilisateur, 0=autres)")

# Pour utiliser avec un CNN, reformater ainsi:
X_train_cnn = X_train.reshape(-1, 64, 64, 1)
print(f"Pour CNN: X_train_cnn.shape = {X_train_cnn.shape}")

print("="*50)