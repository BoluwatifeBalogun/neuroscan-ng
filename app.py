"""
NeuroScan NG
Design and Implementation of a Web-Based Intelligent System for Early
Detection of Alzheimer's Disease from MRI Images using Convolutional
Neural Networks (CNNs).

Case Study: Federal Republic of Nigeria (nationwide deployment context).

Stack: Flask + SQLite + TensorFlow (VGG19 transfer learning).
The app runs in a clearly-labelled DEMO mode until trained model files
(model/alzheimer_vgg19.keras or .h5 + model/labels.json) are present.
"""

import hashlib
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime
from functools import wraps

import numpy as np
from flask import (Flask, abort, flash, g, redirect, render_template,
                   request, session, url_for)
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "neuroscan.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_UPLOAD_MB = 10
IMG_SIZE = (224, 224)

DEFAULT_LABELS = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

CLASS_INFO = {
    "NonDemented": {
        "title": "Non-Demented",
        "tone": "ok",
        "summary": "No structural pattern associated with Alzheimer's-related "
                   "atrophy was identified in this scan by the model.",
    },
    "VeryMildDemented": {
        "title": "Very Mild Demented",
        "tone": "watch",
        "summary": "The model identified subtle patterns consistent with very "
                   "mild Alzheimer's-related changes. Clinical correlation is advised.",
    },
    "MildDemented": {
        "title": "Mild Demented",
        "tone": "warn",
        "summary": "The model identified patterns consistent with mild "
                   "Alzheimer's-related structural change. Specialist review is advised.",
    },
    "ModerateDemented": {
        "title": "Moderate Demented",
        "tone": "alert",
        "summary": "The model identified patterns consistent with moderate "
                   "Alzheimer's-related structural change. Urgent specialist review is advised.",
    },
}

NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue",
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu",
    "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
    "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
    "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
    "Federal Capital Territory",
]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("NEUROSCAN_SECRET", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            institution TEXT,
            state TEXT,
            role TEXT NOT NULL DEFAULT 'clinician',
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            stored_name TEXT NOT NULL,
            original_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            mode TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    try:
        db.execute("ALTER TABLE scans ADD COLUMN atypical INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    db.commit()
    db.close()


# --------------------------------------------------------------------------
# Model engine (lazy-loaded; falls back to labelled demo mode)
# --------------------------------------------------------------------------

class PredictionEngine:
    """Loads the trained VGG19 model once. If TensorFlow or the model
    files are unavailable, serves deterministic, clearly-labelled demo
    predictions so the full system flow remains testable."""

    def __init__(self):
        self._model = None
        self._tflite = None
        self._labels = None
        self._checked = False
        self.mode = "demo"

    def _model_path(self):
        for name in ("alzheimer_vgg19.keras", "alzheimer_vgg19.h5"):
            p = os.path.join(MODEL_DIR, name)
            if os.path.exists(p):
                return p
        return None

    def _load_labels(self, labels_path):
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self._labels = json.load(f)
        else:
            self._labels = DEFAULT_LABELS

    def load(self):
        if self._checked:
            return
        self._checked = True
        labels_path = os.path.join(MODEL_DIR, "labels.json")

        # Preferred on small hosts: TFLite model via the lightweight runtime.
        tflite_path = os.path.join(MODEL_DIR, "alzheimer_vgg19.tflite")
        if os.path.exists(tflite_path):
            interpreter = None
            try:
                from tflite_runtime.interpreter import Interpreter
                interpreter = Interpreter(model_path=tflite_path)
            except Exception as exc:
                app.logger.warning("tflite-runtime unavailable: %s", exc)
                try:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=tflite_path)
                except Exception:
                    interpreter = None
            if interpreter is not None:
                try:
                    interpreter.allocate_tensors()
                    self._tflite = interpreter
                    self._load_labels(labels_path)
                    self.mode = "model"
                    return
                except Exception as exc:  # pragma: no cover
                    app.logger.warning("TFLite load failed: %s", exc)

        # Full Keras model (needed for Grad-CAM) where TensorFlow fits.
        path = self._model_path()
        if path is None:
            return
        try:
            import tensorflow as tf  # noqa: local import keeps demo mode light
            self._model = tf.keras.models.load_model(path)
            self._load_labels(labels_path)
            self.mode = "model"
        except Exception as exc:  # pragma: no cover
            app.logger.warning("Model load failed, staying in demo mode: %s", exc)
            self._model = None
            self.mode = "demo"

    @property
    def labels(self):
        return self._labels or DEFAULT_LABELS

    @staticmethod
    def looks_like_mri(image_bytes):
        """Soft plausibility check: brain MRI slices are effectively grayscale.
        Returns False for strongly coloured images (photos, screenshots) so the
        result can carry a caution. Never blocks; the report treats out-of-
        distribution input as a stated limitation, so the system warns instead."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
            arr = np.asarray(img, dtype=np.float32)
            channel_spread = np.abs(arr - arr.mean(axis=2, keepdims=True)).mean()
            return channel_spread < 12.0
        except Exception:
            return True

    @staticmethod
    def preprocess(image_bytes):
        """Validate + prepare an MRI image exactly as done at training time:
        RGB conversion, resize to 224x224, then VGG19 ImageNet preprocessing
        (BGR channel order, ImageNet channel means subtracted)."""
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
        arr = arr[..., ::-1] - np.array([103.939, 116.779, 123.68],
                                        dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    def _demo_probs(self, image_bytes):
        """Deterministic pseudo-probabilities derived from the image hash,
        so the same file always yields the same demo result."""
        digest = hashlib.sha256(image_bytes).digest()
        raw = np.array([int.from_bytes(digest[i:i + 4], "big") for i in range(0, 16, 4)],
                       dtype=np.float64)
        logits = (raw % 1000) / 140.0
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def predict(self, image_bytes):
        self.load()
        batch = self.preprocess(image_bytes)
        if self._tflite is not None:
            inp = self._tflite.get_input_details()[0]
            out = self._tflite.get_output_details()[0]
            self._tflite.set_tensor(inp["index"], batch.astype(np.float32))
            self._tflite.invoke()
            probs = self._tflite.get_tensor(out["index"])[0].astype(float)
            mode = "model"
        elif self._model is not None:
            probs = self._model.predict(batch, verbose=0)[0].astype(float)
            mode = "model"
        else:
            probs = self._demo_probs(image_bytes)
            mode = "demo"
        labels = self.labels
        order = int(np.argmax(probs))
        return {
            "mode": mode,
            "predicted_class": labels[order],
            "confidence": float(probs[order]),
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
            "class_index": order,
        }

    def gradcam(self, image_bytes, class_idx, out_path):
        """Grad-CAM explainability: writes a heatmap overlay showing which
        regions of the MRI most influenced the prediction. Only available
        when the real model is loaded. Returns True on success."""
        self.load()
        if self._model is None:
            return False
        try:
            import tensorflow as tf
            batch = self.preprocess(image_bytes)
            base = self._model.layers[0]
            last_conv = base.get_layer("block5_conv4")
            grad_model = tf.keras.Model(base.input, [last_conv.output, base.output])
            with tf.GradientTape() as tape:
                conv_out, base_out = grad_model(batch)
                x = base_out
                for layer in self._model.layers[1:]:
                    x = layer(x)
                loss = x[:, class_idx]
            grads = tape.gradient(loss, conv_out)
            weights = tf.reduce_mean(grads, axis=(1, 2))
            cam = tf.nn.relu(tf.einsum("bijc,bc->bij", conv_out, weights))[0].numpy()
            cam = cam / (cam.max() + 1e-8)

            heat = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BICUBIC)
            heat = np.asarray(heat, dtype=np.float32) / 255.0
            rgba = np.zeros((*IMG_SIZE, 4), dtype=np.uint8)
            rgba[..., 0] = np.uint8(255 * np.clip(2 * heat, 0, 1))          # red rises
            rgba[..., 1] = np.uint8(255 * np.clip(2 * (1 - heat), 0, 1) * 0.85)  # green falls
            rgba[..., 2] = 40
            rgba[..., 3] = np.uint8(165 * heat)
            overlay = Image.fromarray(rgba, "RGBA")

            bg = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            bg = bg.convert("RGBA")
            Image.alpha_composite(bg, overlay).convert("RGB").save(out_path)
            return True
        except Exception as exc:  # pragma: no cover
            app.logger.warning("Grad-CAM failed: %s", exc)
            return False


engine = PredictionEngine()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = os.urandom(16).hex()
    return session["csrf_token"]


def csrf_protect(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if request.method == "POST":
            token = request.form.get("csrf_token", "")
            if not token or token != session.get("csrf_token"):
                abort(400, description="Invalid or missing CSRF token.")
        return view(*args, **kwargs)
    return wrapped


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Sign in to continue.", "info")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


@app.context_processor
def inject_globals():
    engine.load()
    return {
        "csrf_token": get_csrf_token(),
        "current_user": current_user(),
        "engine_mode": engine.mode,
        "class_info": CLASS_INFO,
        "now": datetime.now(),
    }


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", states=NIGERIAN_STATES)


@app.route("/register", methods=["GET", "POST"])
@csrf_protect
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        institution = request.form.get("institution", "").strip()
        state = request.form.get("state", "").strip()
        password = request.form.get("password", "")
        if not full_name or not email or len(password) < 8:
            flash("Provide your name, a valid email, and a password of at least 8 characters.", "error")
            return render_template("register.html", states=NIGERIAN_STATES), 400
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (full_name, email, institution, state, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (full_name, email, institution, state,
                 generate_password_hash(password), datetime.utcnow().isoformat()),
            )
            db.commit()
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")
            return render_template("register.html", states=NIGERIAN_STATES), 409
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        session["user_id"] = user["id"]
        flash("Account created. Welcome to NeuroScan NG.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html", states=NIGERIAN_STATES)


@app.route("/login", methods=["GET", "POST"])
@csrf_protect
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Email or password is incorrect.", "error")
            return render_template("login.html"), 401
        session["user_id"] = user["id"]
        dest = request.args.get("next") or url_for("dashboard")
        return redirect(dest)
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Signed out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    uid = session["user_id"]
    recent = db.execute(
        "SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT 5", (uid,)
    ).fetchall()
    stats = db.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN predicted_class != 'NonDemented' THEN 1 ELSE 0 END) AS flagged, "
        "AVG(confidence) AS avg_conf, AVG(elapsed_ms) AS avg_ms "
        "FROM scans WHERE user_id = ?", (uid,)
    ).fetchone()
    return render_template("dashboard.html", recent=recent, stats=stats,
                           max_mb=MAX_UPLOAD_MB)


@app.route("/predict", methods=["POST"])
@login_required
@csrf_protect
def predict():
    file = request.files.get("mri")
    if file is None or file.filename == "":
        flash("Choose an MRI image to analyse.", "error")
        return redirect(url_for("dashboard"))
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        flash("Unsupported file type. Upload a JPG, PNG, BMP or WEBP image.", "error")
        return redirect(url_for("dashboard"))

    image_bytes = file.read()
    started = datetime.now()
    try:
        result = engine.predict(image_bytes)
    except (UnidentifiedImageError, OSError):
        flash("The file could not be read as an image. Upload a valid MRI scan.", "error")
        return redirect(url_for("dashboard"))
    elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)

    atypical = 0 if PredictionEngine.looks_like_mri(image_bytes) else 1
    stored = f"{uuid.uuid4().hex}.{ext}"
    with open(os.path.join(UPLOAD_DIR, stored), "wb") as f:
        f.write(image_bytes)
    if result["mode"] == "model":
        engine.gradcam(image_bytes, result["class_index"],
                       os.path.join(UPLOAD_DIR, stored + ".cam.png"))

    db = get_db()
    cur = db.execute(
        "INSERT INTO scans (user_id, stored_name, original_name, predicted_class, "
        "confidence, probabilities, mode, elapsed_ms, created_at, atypical) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session["user_id"], stored, secure_filename(file.filename),
         result["predicted_class"], result["confidence"],
         json.dumps(result["probabilities"]), result["mode"], elapsed_ms,
         datetime.utcnow().isoformat(), atypical),
    )
    db.commit()
    return redirect(url_for("result", scan_id=cur.lastrowid))


@app.route("/result/<int:scan_id>")
@login_required
def result(scan_id):
    scan = get_db().execute(
        "SELECT * FROM scans WHERE id = ? AND user_id = ?",
        (scan_id, session["user_id"])
    ).fetchone()
    if scan is None:
        abort(404)
    probs = json.loads(scan["probabilities"])
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    has_cam = os.path.exists(os.path.join(UPLOAD_DIR, scan["stored_name"] + ".cam.png"))
    return render_template("result.html", scan=scan, ordered=ordered, has_cam=has_cam)


@app.route("/scan-cam/<int:scan_id>")
@login_required
def scan_cam(scan_id):
    from flask import send_from_directory
    scan = get_db().execute(
        "SELECT * FROM scans WHERE id = ? AND user_id = ?",
        (scan_id, session["user_id"])
    ).fetchone()
    if scan is None:
        abort(404)
    cam_name = scan["stored_name"] + ".cam.png"
    if not os.path.exists(os.path.join(UPLOAD_DIR, cam_name)):
        abort(404)
    return send_from_directory(UPLOAD_DIR, cam_name)


@app.route("/scan-image/<int:scan_id>")
@login_required
def scan_image(scan_id):
    from flask import send_from_directory
    scan = get_db().execute(
        "SELECT * FROM scans WHERE id = ? AND user_id = ?",
        (scan_id, session["user_id"])
    ).fetchone()
    if scan is None:
        abort(404)
    return send_from_directory(UPLOAD_DIR, scan["stored_name"])


@app.route("/history")
@login_required
def history():
    scans = get_db().execute(
        "SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC",
        (session["user_id"],)
    ).fetchall()
    return render_template("history.html", scans=scans)


@app.route("/performance")
def performance():
    return render_template("performance.html", metrics=load_metrics())


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404,
                           message="This page does not exist."), 404


@app.errorhandler(413)
def too_large(_e):
    flash(f"File is larger than {MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("dashboard"))


init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
