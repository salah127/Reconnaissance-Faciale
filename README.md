# Système de reconnaissance faciale binaire

## Objectif
Ce projet permet d'entraîner et d'utiliser un modèle de deep learning pour reconnaître automatiquement une personne spécifique parmi d'autres à partir d'images de visages.

## Structure du projet
- `main.py` : script principal d'entraînement et d'évaluation
- `test_image.py` : script pour tester une image avec le modèle entraîné
- `requirements.txt` : dépendances Python
- `seuil_user.json` : seuil optimal et statistiques du modèle
- `data/` : dossier contenant les données (exemples fournis)
- `fig_*.png` : figures générées automatiquement (courbes, matrices, etc.)

## Installation
1. Cloner le dépôt
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation rapide
### 1. Entraîner le modèle
Lancer le script principal :
```bash
python main.py
```
Le script va :
- Charger et préparer les données
- Entraîner le modèle
- Générer toutes les figures utiles
- Sauvegarder le modèle et le seuil optimal

### 2. Tester une image
Mettre une image de test dans `data/test/` (format PNG ou JPG, visage centré).
Lancer :
```bash
python test_image.py
```
Le script affiche le score et indique si l'image est reconnue comme l'utilisateur ou non.

## Fichiers importants à tester
- `modele_user.keras` et `best_model_user_final.keras` : modèles prêts à l'emploi
- `seuil_user.json` : seuil optimal pour la décision
- Quelques images de test dans `data/test/`

## Génération des figures
Toutes les figures (distribution des classes, courbes d'apprentissage, histogrammes, matrice de confusion, etc.) sont générées automatiquement lors de l'entraînement et sauvegardées dans le dossier du projet.

## Conseils
- Pour réentraîner sur vos propres images, placez-les dans `data/` et adaptez le script si besoin.
- Pour tester, utilisez des images en niveaux de gris, 64x64 ou laissez le script redimensionner.

## Contact
Pour toute question ou amélioration, contactez l'auteur du projet.
