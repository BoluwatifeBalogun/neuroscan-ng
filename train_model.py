"""
NeuroScan NG VGG19 training pipeline (Objectives 1-3 of the study).

Run on Google Colab (GPU runtime) or any machine with TensorFlow:

    python train_model.py --data_dir /path/to/AlzheimerDataset --epochs 15

Expected data layout (Kaggle "Alzheimer's Dataset (4 class of Images)"):
    data_dir/
        MildDemented/       *.jpg
        ModerateDemented/   *.jpg
        NonDemented/        *.jpg
        VeryMildDemented/   *.jpg

Outputs (drop the whole folder contents into the web app's model/ dir):
    model/alzheimer_vgg19.keras   trained model
    model/labels.json             class order used by the model
    model/metrics.json            accuracy, precision, recall, sensitivity,
                                  specificity, F1, per-class breakdown
    model/confusion_matrix.png
    model/training_curves.png
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19

IMG_SIZE = (224, 224)
SEED = 42


def build_datasets(data_dir, batch_size):
    """Objective 1: collect and preprocess labelled MRI images.
    Stratified 70/15/15 split, resize 224x224, rescale to [0,1]."""
    train_full = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.30, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical")
    holdout = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.30, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical")

    class_names = train_full.class_names
    val_batches = tf.data.experimental.cardinality(holdout) // 2
    val_ds = holdout.take(val_batches)
    test_ds = holdout.skip(val_batches)

    rescale = layers.Rescaling(1.0 / 255)
    augment = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.10),
        layers.RandomFlip("horizontal"),
    ])

    autotune = tf.data.AUTOTUNE
    train_ds = (train_full
                .map(lambda x, y: (augment(rescale(x), training=True), y),
                     num_parallel_calls=autotune)
                .prefetch(autotune))
    val_ds = val_ds.map(lambda x, y: (rescale(x), y)).prefetch(autotune)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y)).prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names, train_full


def compute_class_weights(train_full, n_classes):
    counts = np.zeros(n_classes)
    for _, y in train_full.unbatch():
        counts[int(np.argmax(y))] += 1
    total = counts.sum()
    return {i: float(total / (n_classes * c)) for i, c in enumerate(counts)}


def build_model(n_classes):
    """Objective 2: VGG19 transfer learning. ImageNet convolutional base is
    frozen; a new classification head is trained for the 4 MRI classes."""
    base = VGG19(weights="imagenet", include_top=False,
                 input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation="softmax"),
    ], name="neuroscan_vgg19")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base


def fine_tune(model, base, lr=1e-5):
    """Phase 2: unfreeze block5 of VGG19 and continue at a low learning rate."""
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])


def evaluate(model, test_ds, class_names, out_dir):
    """Objective 3: accuracy, precision, recall, sensitivity, specificity,
    F1-score and confusion matrix on the held-out test set."""
    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class, macro = {}, {"precision": [], "recall": [], "specificity": [], "f1": []}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0          # = sensitivity
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }
        for k, v in (("precision", precision), ("recall", recall),
                     ("specificity", specificity), ("f1", f1)):
            macro[k].append(v)

    metrics = {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_precision": round(float(np.mean(macro["precision"])), 4),
        "macro_recall_sensitivity": round(float(np.mean(macro["recall"])), 4),
        "macro_specificity": round(float(np.mean(macro["specificity"])), 4),
        "macro_f1": round(float(np.mean(macro["f1"])), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "test_samples": int(len(y_true)),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(n), class_names, rotation=45, ha="right")
        ax.set_yticks(range(n), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "#0E1B14")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    except ImportError:
        pass
    return metrics


def plot_history(histories, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(acc, color="#046A38", label="train")
    a1.plot(val_acc, color="#0DA65D", label="validation")
    a1.set_title("Accuracy"); a1.legend()
    a2.plot(loss, color="#046A38", label="train")
    a2.plot(val_loss, color="#0DA65D", label="validation")
    a2.set_title("Loss"); a2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="model")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--fine_tune_epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_ds, val_ds, test_ds, class_names, train_full = build_datasets(
        args.data_dir, args.batch_size)
    print("Classes:", class_names)

    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(class_names, f)

    weights = compute_class_weights(train_full, len(class_names))
    model, base = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6),
    ]

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                   class_weight=weights, callbacks=callbacks)

    fine_tune(model, base)
    h2 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.fine_tune_epochs, class_weight=weights,
                   callbacks=callbacks)

    model.save(os.path.join(args.out_dir, "alzheimer_vgg19.keras"))

    # TensorFlow Lite export (float16): ~40 MB model that runs on small
    # hosts (e.g. Render free tier) via tflite-runtime, no full TF needed.
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        with open(os.path.join(args.out_dir, "alzheimer_vgg19.tflite"), "wb") as f:
            f.write(converter.convert())
        print("TFLite export: alzheimer_vgg19.tflite")
    except Exception as exc:
        print("TFLite export skipped:", exc)

    plot_history([h1, h2], args.out_dir)
    metrics = evaluate(model, test_ds, class_names, args.out_dir)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("confusion_matrix", "per_class")}, indent=2))
    print(f"\nDone. Copy everything in '{args.out_dir}/' into the web app's model/ folder.")


if __name__ == "__main__":
    main()
