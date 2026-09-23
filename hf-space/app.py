"""
NeuroScan NG Classifier (Hugging Face Space)
Gradio front-end for the VGG19 Alzheimer's MRI classifier.

Runs the real trained model when alzheimer_vgg19.keras + labels.json sit
next to this file. Until then it serves clearly-labelled demo predictions
so the Space works end to end from day one.
"""

import hashlib
import io
import json
import os

import gradio as gr
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)
DEFAULT_LABELS = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
PRETTY = {
    "NonDemented": "Non-Demented",
    "VeryMildDemented": "Very Mild Demented",
    "MildDemented": "Mild Demented",
    "ModerateDemented": "Moderate Demented",
}

_model, _labels = None, None


def load_model():
    global _model, _labels
    if _model is not None:
        return True
    path = next((os.path.join(HERE, n) for n in
                 ("alzheimer_vgg19.keras", "alzheimer_vgg19.h5")
                 if os.path.exists(os.path.join(HERE, n))), None)
    if path is None:
        return False
    import tensorflow as tf
    _model = tf.keras.models.load_model(path)
    lp = os.path.join(HERE, "labels.json")
    _labels = json.load(open(lp)) if os.path.exists(lp) else DEFAULT_LABELS
    return True


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    return np.expand_dims(np.asarray(img, dtype=np.float32) / 255.0, 0)


def demo_probs(img: Image.Image):
    b = io.BytesIO(); img.convert("RGB").save(b, "PNG")
    digest = hashlib.sha256(b.getvalue()).digest()
    raw = np.array([int.from_bytes(digest[i:i + 4], "big") for i in range(0, 16, 4)],
                   dtype=np.float64)
    logits = (raw % 1000) / 140.0
    e = np.exp(logits - logits.max())
    return e / e.sum()


def looks_like_mri(img: Image.Image):
    arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    return np.abs(arr - arr.mean(axis=2, keepdims=True)).mean() < 12.0


def gradcam(img: Image.Image, class_idx: int):
    import tensorflow as tf
    batch = preprocess(img)
    base = _model.layers[0]
    grad_model = tf.keras.Model(base.input,
                                [base.get_layer("block5_conv4").output, base.output])
    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(batch)
        x = base_out
        for layer in _model.layers[1:]:
            x = layer(x)
        loss = x[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.einsum("bijc,bc->bij", conv_out, weights))[0].numpy()
    cam /= cam.max() + 1e-8
    heat = np.asarray(Image.fromarray(np.uint8(cam * 255))
                      .resize(IMG_SIZE, Image.BICUBIC), dtype=np.float32) / 255.0
    rgba = np.zeros((*IMG_SIZE, 4), dtype=np.uint8)
    rgba[..., 0] = np.uint8(255 * np.clip(2 * heat, 0, 1))
    rgba[..., 1] = np.uint8(255 * np.clip(2 * (1 - heat), 0, 1) * 0.85)
    rgba[..., 2] = 40
    rgba[..., 3] = np.uint8(165 * heat)
    bg = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS).convert("RGBA")
    return Image.alpha_composite(bg, Image.fromarray(rgba, "RGBA")).convert("RGB")


def classify(img: Image.Image):
    if img is None:
        raise gr.Error("Upload a brain MRI image first.")
    have_model = load_model()
    if have_model:
        probs = _model.predict(preprocess(img), verbose=0)[0].astype(float)
        labels = _labels
    else:
        probs = demo_probs(img)
        labels = DEFAULT_LABELS
    scores = {PRETTY.get(labels[i], labels[i]): float(probs[i]) for i in range(len(labels))}
    idx = int(np.argmax(probs))

    cam = gradcam(img, idx) if have_model else None

    notes = []
    if not have_model:
        notes.append("**Demo mode**: the trained model is not installed in this Space yet, "
                     "so this is an illustrative placeholder, not a model prediction.")
    if not looks_like_mri(img):
        notes.append("**Caution**: this image does not look like a typical grayscale MRI "
                     "slice; the classification is unlikely to be meaningful.")
    notes.append("This is a preliminary, decision-support result from a research system. "
                 "It must be interpreted by a qualified medical professional and is **not a diagnosis**.")
    return scores, cam, "\n\n".join(notes)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Brain MRI image (axial slice)"),
    outputs=[
        gr.Label(num_top_classes=4, label="Prediction"),
        gr.Image(label="Grad-CAM: regions driving the prediction"),
        gr.Markdown(label="Notes"),
    ],
    title="NeuroScan NG: Alzheimer's MRI Classifier",
    description=(
        "VGG19 transfer-learning classifier from the study *Design and Implementation of a "
        "Web-Based Intelligent System for Early Detection of Alzheimer's Disease from MRI "
        "Images using CNNs* (case study: Nigeria). Upload an axial brain MRI slice to get a "
        "four-class prediction with confidence scores. Full web application: https://neuroscan-ng.onrender.com"
    ),
    theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
