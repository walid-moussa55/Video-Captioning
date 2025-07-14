# Video Captioning with ResNet50

## Présentation

Ce projet propose un système complet de génération automatique de descriptions textuelles (captions) pour des vidéos, basé sur l'extraction de caractéristiques visuelles avec ResNet50 et l'utilisation de modèles séquentiels avec attention. Il inclut :
- Un pipeline de préparation et d'entraînement sur le dataset MSR-VTT.
- Un modèle personnalisé de captioning vidéo (encoder-decoder avec attention).
- Une interface web interactive (Flask + HTML/JS) permettant de tester deux modèles :
  - **Custom Video Model** : génère une légende globale pour la vidéo.
  - **Frame Analysis Model** : génère une légende pour chaque segment/intervalle de la vidéo (basé sur ViT-GPT2).

## Structure du projet

- `video-captioning-resnet50.ipynb` : Notebook principal pour la préparation des données, l'extraction des frames, l'extraction des features avec ResNet50, la préparation des captions, l'entraînement du modèle sequence-to-sequence avec attention, et l'évaluation.
- `video-captioning-UI.ipynb` : Notebook contenant l'API Flask et l'interface utilisateur web pour uploader une vidéo et obtenir des captions avec le modèle custom ou le modèle pré-entraîné.

## Pipeline de traitement

### 1. Préparation des données
- Téléchargement et extraction du dataset MSR-VTT.
- Génération de deux fichiers :
  - `video_paths.txt` : mapping entre un ID numérique et le chemin de la vidéo.
  - `captions.txt` : mapping entre l'ID et chaque caption associée.

### 2. Extraction des frames
- Extraction des 40 premières frames de chaque vidéo valide.
- Sauvegarde des frames dans un dossier dédié.

### 3. Extraction des features
- Utilisation de ResNet50 (pré-entraîné ImageNet, sans la couche top, pooling='avg') pour extraire un vecteur de 2048 dimensions par frame.
- Sauvegarde des features de chaque vidéo sous forme de fichiers `.npy`.

### 4. Préparation des captions
- Nettoyage et formatage des captions (`<start> ... <end>`).
- Tokenization et création du vocabulaire (taille max 1500 tokens).

### 5. Entraînement du modèle
- Modèle encoder-decoder avec attention :
  - **Encoder** : Bidirectional LSTM sur les features extraites (40x2048).
  - **Decoder** : LSTM avec attention, génère la séquence de tokens.
- Entraînement avec early stopping, réduction du learning rate, et TensorBoard.

### 6. Sauvegarde et inférence
- Sauvegarde des modèles d'inférence (encoder, decoder, tokenizer).
- Fonction d'inférence pour générer une caption à partir d'une vidéo.

## Interface Web

- API Flask permettant d'uploader une vidéo et de choisir le modèle de génération.
- Deux modes :
  - **Custom Video Model** : utilise le modèle entraîné ResNet50+LSTM+Attention pour générer une caption globale.
  - **Frame Analysis Model** : utilise le modèle ViT-GPT2 pré-entraîné pour générer une caption par segment (via HuggingFace Transformers).
- Interface HTML moderne avec sélection du modèle, upload vidéo, affichage des résultats et gestion des erreurs.

## Dépendances principales

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Flask
- Pyngrok (pour exposer l'API en public)
- Transformers (HuggingFace)
- Numpy, Pandas, Matplotlib, Seaborn
- Joblib

## Lancement de l'interface

1. Installer les dépendances :
   ```sh
   pip install python-dotenv flask pyngrok tensorflow keras opencv-python transformers joblib
   ```

2. Lancer le notebook [video-captioning-UI.ipynb](d:/Video%20Caption/video-captioning-UI.ipynb) et exécuter toutes les cellules.

3. Accéder à l'URL publique générée par ngrok pour utiliser l'interface web.

## Références

- [MSR-VTT Dataset](https://www.robots.ox.ac.uk/~maxbain/msrvtt/)
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
- [ViT-GPT2 Image Captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

---

**Auteurs** :  
- [WAM (3lallah Grpoup)](https://github.com/walid-moussa55)
- [Mimoun Ouhda (3lallah Grpoup)](https://github.com/mimounouhd)
- [Yassin Boujnan (3lallah Grpoup)]()
