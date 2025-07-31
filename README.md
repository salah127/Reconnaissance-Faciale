# 🧠 Projet de Reconnaissance Faciale Binaire avec CNN

## 🎯 Objectif du TP

Développer un système de **reconnaissance faciale binaire** basé sur un **réseau de neurones convolutionnel (CNN)** qui :
- Analyse une image de visage,
- Retourne `1` si c’est **l’utilisateur**, `0` sinon (**personne lambda**).

Ce projet permet d’apprendre à :
- Construire et entraîner un CNN pour la classification binaire,
- Collecter et prétraiter un dataset personnalisé,
- Implémenter un système complet de reconnaissance faciale basé IA.

---

## 📁 Structure du Projet

face_authentication_cnn/
│
├── data/
│ ├── user/ # Images de l’utilisateur (toi)
│ ├── others/ # Images de personnes lambda (ex: CelebA, LFW)
│ ├── processed/ # Images redimensionnées et normalisées
│ └── split/ # Données divisées en train/val/test
│
├── notebooks/
│ └── exploration.ipynb # Analyse exploratoire, visualisation, etc.
│
├── models/
│ └── cnn_face_model.pt # Modèle entraîné (PyTorch) ou .h5 (TensorFlow)
│
├── utils/
│ ├── data_preprocessing.py # Redimensionnement, normalisation, split, augmentation
│ └── metrics.py # Métriques d'évaluation personnalisées
│
├── train.py # Script d'entraînement du modèle
├── evaluate.py # Évaluation sur l'ensemble de test
├── predict.py # Prédiction sur une image ou via webcam
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier


---

## 🔄 Workflow du Projet

### 1. 📸 Collecte des Données

- **Utilisateur** : prendre **au moins 50 photos** de ton visage dans différentes expressions, angles, arrière-plans et éclairages.
- **Autres visages** : utiliser un dataset public comme **CelebA** ou **LFW** pour récupérer un nombre équivalent d’images.

---

### 2. 🧼 Prétraitement

- Redimensionner toutes les images à **64x64** (ou 128x128) pixels.
- Normaliser les valeurs de pixels (entre 0 et 1).
- Sauvegarder les images traitées dans `data/processed/`.

> Script à utiliser : `utils/data_preprocessing.py`

---

### 3. 🧪 Découpage du Dataset

- Diviser les images en :
  - **70%** pour l'entraînement
  - **15%** pour la validation
  - **15%** pour le test
- Assurer un **équilibrage des classes** dans chaque ensemble.

---

### 4. 🧠 Conception du Modèle CNN

- Framework conseillé : **PyTorch** ou **TensorFlow/Keras**
- Architecture suggérée :

    Conv2D → ReLU → MaxPooling → Dropout
    Conv2D → ReLU → MaxPooling → Dropout
    Flatten → Dense → Dropout → Dense(1, activation='sigmoid')


---

### 5. 🏋️ Entraînement

- Fonction de perte : `binary_crossentropy`
- Optimiseur : `adam`
- Suivi des performances : accuracy, loss, F1-score
- Utiliser la validation pour éviter le surapprentissage

> Script : `train.py`

---

### 6. 📊 Évaluation

- Tester sur l’ensemble de test avec les métriques :
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Analyser les erreurs (faux positifs, faux négatifs)

> Script : `evaluate.py`

---

### 7. 🧪 Prédiction

- Charger une image (ou utiliser la webcam)
- Prédire si c’est l’utilisateur (`1`) ou une autre personne (`0`)

> Script : `predict.py`

---

## ⚙️ Installation des dépendances

```bash
pip install -r requirements.txt

---

## Commandes utiles

# Prétraitement des images
```bash
python utils/data_preprocessing.py

# Entraînement du modèle
```bash
python train.py

# Évaluation du modèle
```bash
python evaluate.py

# Prédiction avec une image
```bash
python predict.py --image path/to/image.jpg


