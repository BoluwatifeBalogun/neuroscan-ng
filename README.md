# NeuroScan NG

**Design and Implementation of a Web-Based Intelligent System for Early Detection of Alzheimer's Disease from MRI Images using Convolutional Neural Networks (CNNs).**
Case study: Federal Republic of Nigeria.

A Flask + SQLite + TensorFlow system. A VGG19 transfer-learning model classifies brain MRI images into four classes (Non-Demented, Very Mild, Mild, Moderate Demented) and the web app lets authorised users upload scans, view predictions with confidence scores, probability breakdowns and Grad-CAM heatmaps, and keep a scan history.

## Quick start (demo mode)

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Without the trained model the app runs in a clearly-labelled **demo mode**: every screen and flow works, predictions are deterministic placeholders.

## Training the real model (Google Colab recommended)

1. Download the Kaggle **Alzheimer's Dataset (4 class of Images)** (~6,400 MRI images) and note the folder containing the four class subfolders.
2. On a GPU runtime:

```bash
pip install tensorflow matplotlib
python train_model.py --data_dir /content/AlzheimerDataset --epochs 12 --fine_tune_epochs 6
```

3. The script performs the full methodology from the write-up: stratified 70/15/15 split, 224×224 preprocessing, augmentation, class weighting, frozen-base training, block5 fine-tuning, and held-out evaluation with **accuracy, precision, recall, sensitivity, specificity, F1-score and confusion matrix** (Objective 3).
4. Copy the produced files into the web app:

```
model/alzheimer_vgg19.keras
model/labels.json
model/metrics.json
model/confusion_matrix.png
model/training_curves.png
```

5. Install TensorFlow where the app runs (`pip install tensorflow`) and restart. The demo banner disappears, predictions come from the model, and the Model Performance page renders the real metrics.

## Structure

```
app.py            Flask app: auth, upload → preprocess → predict → result, history, metrics
train_model.py    VGG19 training pipeline (Objectives 1-3)
train_neuroscan_colab.ipynb  Ready-to-run Colab notebook (dataset download + training + artifact export)
templates/        Jinja pages (landing, auth, dashboard, result, history, performance, about)
static/           Green & white Nigeria design system (CSS + JS)
model/            Drop trained artefacts here
uploads/          Stored scan images
```

## Important

Every prediction is a **preliminary, decision-support result** requiring interpretation by a qualified medical professional. The system is a research artefact, not a diagnostic device.
