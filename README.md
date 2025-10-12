# Emotion Detection Project

This repository contains code and resources for an emotion detection system using deep learning and computer vision. The project includes data preprocessing, augmentation, model training (CNN and ResNet), and a Tkinter-based GUI for real-time emotion recognition from webcam input.

## Project Structure

- `app.py` — Main GUI application for emotion detection (Tkinter, OpenCV, PyTorch).
- `cnn.py` — Implementation of the EmotionCNN model.
- `data_augmentation.py` — Tools for augmenting image datasets (noise, rotation, flipping, brightness).
- `preprocess.py` — Scripts for preprocessing images (grayscale, resize to 48x48).
- `resnet.ipynb` — Jupyter notebook for training and evaluating CNN/ResNet models.
- `emotionCNN.pth` — Trained EmotionCNN model weights.
- `predictions.csv` — Output predictions for test images.
- `test_template.csv` — Template for test image IDs.
- `images/` — Emotion icon images for GUI display.

## Usage

### 1. Preprocess Dataset

Run `preprocess.py` to convert images to grayscale and resize to 48x48 pixels.

### 2. Data Augmentation

Use `data_augmentation.py` to create augmented versions of your dataset for improved model robustness.

### 3. Model Training

Train models using `resnet.ipynb`. You can choose between custom CNN (`cnn.py`) and ResNet architectures.

### 4. GUI Application

Launch the emotion detector GUI:

```sh
python app.py
```

The app will use your webcam to detect faces and predict emotions in real time.

### 5. Prediction Submission

Generate predictions for test images using the notebook and save results to `predictions.csv`.

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Notes

- Place your training images in the appropriate folder structure before preprocessing.
- Make sure emotion icon images are available in the `images/` directory.
- Model weights (`emotionCNN.pth`) should be present for the GUI to work in detection mode.

## License

MIT License

---

For details on each module, see the source files:
- [app.py](app.py)
- [cnn.py](cnn.py)
- [data_augmentation.py](data_augmentation.py)
- [preprocess.py](preprocess.py)