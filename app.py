# app.py
# Full-featured Flask app with patient-name support and camera-compatible uploads.
#
# Requirements:
# Flask, tensorflow, pillow, numpy, matplotlib, reportlab
#
# Install:
# py -m pip install Flask tensorflow pillow numpy matplotlib reportlab

import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask_cors import CORS

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_REL = os.path.join('model', 'best_retinopathy_model.h5')
MODEL_PATH = os.path.join(BASE_DIR, MODEL_REL)

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 48 * 1024 * 1024  # 48 MB
IMAGE_SIZE = (150, 150)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# expose zip if you used it in templates (optional)
app.jinja_env.globals['zip'] = zip
CORS(app)

# ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)

# --- Load model safely ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place best_retinopathy_model.h5 inside the model/ folder.")

# load model (compile=False for compatibility)
model = load_model(MODEL_PATH, compile=False)

# explicit 5-class mapping
DEFAULT_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def sanitize_patient_name(name: str) -> str:
    """Return a filesystem-safe short representation for patient name (no spaces, limited chars)."""
    if not name:
        return ''
    # keep letters, numbers, underscore; replace spaces with underscore
    cleaned = "".join(c for c in name if c.isalnum() or c in (' ', '_')).strip()
    cleaned = cleaned.replace(" ", "_")
    # trim length
    return cleaned[:64]

def preprocess_image(filepath: str):
    img = Image.open(filepath).convert('RGB')
    img = ImageOps.fit(img, IMAGE_SIZE, Image.LANCZOS)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict_raw_array(arr: np.ndarray):
    """Return numpy predictions for arr (1,H,W,C)."""
    preds = model.predict(arr)
    return np.asarray(preds)

def interpret_predictions(preds: np.ndarray):
    """
    Given prediction array, return:
    - is_multiclass: bool
    - class_probs: 1D numpy array of probabilities (if binary returns 2 probs: [P(NoDR), P(DR)])
    - label: human label string
    - top_prob: float (probability of chosen label)
    """
    preds_arr = np.asarray(preds).ravel()
    if preds_arr.size == 0:
        return False, np.array([0.0]), "Unknown", 0.0

    # Binary: single probability, treat as probability of positive (DR)
    if preds_arr.size == 1:
        prob_pos = float(preds_arr[0])
        probs = np.array([1.0 - prob_pos, prob_pos])
        class_names = ["No DR", "Diabetic Retinopathy"]
        label = class_names[1] if prob_pos > 0.5 else class_names[0]
        top_prob = max(prob_pos, 1.0 - prob_pos)
        return False, probs, label, float(top_prob)

    # Multi-class: logits or probs vector
    probs = preds_arr.astype(np.float64)
    # If not normalized, softmax
    if probs.min() < 0 or probs.max() > 1.0001 or not np.isclose(probs.sum(), 1.0, atol=1e-2):
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    # If exactly 5 outputs, map to DR classes
    if probs.size == 5:
        class_names = DEFAULT_CLASS_NAMES
    else:
        class_names = [f"Class {i}" for i in range(probs.size)]

    top_idx = int(np.argmax(probs))
    label = class_names[top_idx]
    return True, probs, label, float(probs[top_idx])

# ---- Grad-CAM helpers (jet colormap) and saliency fallback ----
def _list_conv_like_layers(m):
    candidates = []
    for layer in m.layers:
        try:
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                candidates.append(layer.name)
                continue
        except Exception:
            pass
        lname = getattr(layer, 'name', '').lower()
        if 'conv' in lname:
            candidates.append(layer.name)
    return candidates

def find_best_conv_layer_name(m):
    for layer in reversed(m.layers):
        try:
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                return layer.name
        except Exception:
            pass
    conv_like = _list_conv_like_layers(m)
    if conv_like:
        return conv_like[-1]
    layer_names = [getattr(l, 'name', str(type(l))) for l in m.layers]
    raise ValueError(f"GRADCAM: no conv-like layer found. Model layers: {layer_names}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    except Exception as e:
        raise RuntimeError(f"GRADCAM: building grad_model failed: {e}")

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            try:
                if predictions.shape.rank is None:
                    pred_index = 0
                elif predictions.shape.rank == 1:
                    pred_index = 0
                elif predictions.shape.rank == 2 and predictions.shape[1] == 1:
                    pred_index = 0
                else:
                    pred_index = tf.argmax(predictions[0])
            except Exception:
                pred_index = 0
        try:
            class_channel = predictions[:, pred_index]
        except Exception:
            class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        grads = tape.gradient(tf.reduce_sum(class_channel), conv_outputs)
        if grads is None:
            conv_shape = conv_outputs.shape[1:3]
            return np.zeros((int(conv_shape[0]), int(conv_shape[1])))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)
    heatmap /= max_val
    return heatmap.numpy()

def make_saliency_heatmap(img_array, model, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        if pred_index is None:
            try:
                if predictions.shape.rank is None:
                    pred_index = 0
                elif predictions.shape.rank == 1:
                    pred_index = 0
                elif predictions.shape.rank == 2 and predictions.shape[1] == 1:
                    pred_index = 0
                else:
                    pred_index = tf.argmax(predictions[0])
            except Exception:
                pred_index = 0
        try:
            class_channel = predictions[:, pred_index]
        except Exception:
            class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, img_tensor)
    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))
    sal = tf.reduce_mean(tf.abs(grads), axis=-1)[0].numpy()
    sal = sal - sal.min() if sal.max() - sal.min() != 0 else sal
    if sal.max() != 0:
        sal = sal / sal.max()
    return sal

def apply_jet_colormap_and_overlay(original_img_path, heatmap_2d, out_path, alpha=0.45):
    base = Image.open(original_img_path).convert("RGBA")
    base_w, base_h = base.size
    hm_resized = Image.fromarray(np.uint8(255 * heatmap_2d)).resize((base_w, base_h), resample=Image.BILINEAR)
    hm_arr = np.array(hm_resized)
    colormap = cm.get_cmap('jet')
    colored = colormap(hm_arr / 255.0)
    colored_img = np.uint8(colored * 255)
    colored_pil = Image.fromarray(colored_img, mode='RGBA')
    alpha_channel = (hm_arr.astype(np.float32) / 255.0 * 255.0 * alpha).astype(np.uint8)
    colored_np = np.array(colored_pil)
    colored_np[..., 3] = alpha_channel
    colored_pil = Image.fromarray(colored_np, mode='RGBA')
    composite = Image.alpha_composite(base, colored_pil)
    composite.save(out_path)
    return out_path

# --- Bar chart for class probabilities (multi-class) ---
def save_class_bar_chart(probs, class_names, out_path):
    plt.figure(figsize=(6, 3))
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probs, align='center', color=cm.viridis(probs))
    plt.yticks(y_pos, class_names)
    plt.xlim(0, 1)
    plt.xlabel('Probability')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# --- PDF report generation (reportlab) ---
def generate_pdf_report(image_path, heatmap_path, prediction, probability, class_probs=None, class_names=None, patient_name=None, patient_sex=None, out_pdf_path=None):
    """
    Now includes patient_name and patient_sex in the PDF header and the filename.
    """
    if out_pdf_path is None:
        safe_name = ''
        if patient_name:
            safe = sanitize_patient_name(patient_name)
            safe_name = "_" + safe if safe else ''
        out_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"report{safe_name}_{uuid.uuid4().hex}.pdf")

    doc = SimpleDocTemplate(out_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Diabetic Retinopathy Prediction Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 8))

    # patient info
    if patient_name or patient_sex:
        info = "<b>Patient:</b> " + (patient_name or "N/A")
        if patient_sex:
            info += f" &nbsp;&nbsp; <b>Sex:</b> {patient_sex}"
        story.append(Paragraph(info, styles['Normal']))
        story.append(Spacer(1, 6))

    summary_text = f"<b>Prediction:</b> {prediction} &nbsp;&nbsp; <b>Confidence:</b> {round(probability*100,2)}%"
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))

    try:
        story.append(Paragraph("<b>Uploaded Image:</b>", styles['Heading3']))
        story.append(Spacer(1, 6))
        story.append(RLImage(image_path, width=350, height=250))
        story.append(Spacer(1, 12))
    except Exception:
        pass

    if heatmap_path:
        try:
            story.append(Paragraph("<b>Model Attention (Heatmap):</b>", styles['Heading3']))
            story.append(Spacer(1, 6))
            story.append(RLImage(heatmap_path, width=350, height=250))
            story.append(Spacer(1, 12))
        except Exception:
            pass

    if class_probs is not None and class_names is not None:
        try:
            # fixed syntax here (was a mismatched quote in your earlier file)
            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], f"class_chart_{uuid.uuid4().hex}.png")
            save_class_bar_chart(class_probs, class_names, chart_path)
            story.append(Paragraph("<b>Class probabilities:</b>", styles['Heading3']))
            story.append(Spacer(1, 6))
            story.append(RLImage(chart_path, width=400, height=150))
            story.append(Spacer(1, 12))
        except Exception:
            pass

    doc.build(story)
    return out_pdf_path

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    image_url = None
    heatmap_url = None
    report_url = None
    class_probs = None
    class_names = None

    if request.method == 'POST':
        # check for file
        if 'file' not in request.files:
            flash('No file part in request', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        # read patient metadata from form
        patient_name_raw = request.form.get('patient_name', '') or ''
        patient_sex = request.form.get('patient_sex', '') or ''
        patient_name = patient_name_raw.strip()
        safe_patient = sanitize_patient_name(patient_name)  # empty string if none

        if file and allowed_file(file.filename):
            # create filename that includes patient (if present)
            orig_fname = secure_filename(file.filename)
            prefix = (safe_patient + "_") if safe_patient else ""
            unique = f"{uuid.uuid4().hex}_{prefix}{orig_fname}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
            try:
                file.save(save_path)
            except Exception as e:
                flash(f"Could not save file: {e}", 'error')
                return redirect(request.url)

            # Predict
            try:
                arr = preprocess_image(save_path)
                # warm-up model
                try:
                    _ = model.predict(arr, verbose=0)
                except Exception:
                    try:
                        _ = model(arr)
                    except Exception:
                        for layer in model.layers:
                            try:
                                if isinstance(layer, tf.keras.Model):
                                    try:
                                        _ = layer(arr, training=False)
                                    except Exception:
                                        try:
                                            _ = layer.predict(arr)
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                preds = predict_raw_array(arr)
                is_multi, probs, label, top_prob = interpret_predictions(preds)
                prediction = label
                probability = float(top_prob)
                image_url = url_for('static', filename=f'uploads/{unique}')
                if is_multi:
                    class_probs = probs.tolist()
                    if len(probs) == len(DEFAULT_CLASS_NAMES):
                        class_names = DEFAULT_CLASS_NAMES
                    else:
                        class_names = [f"Class {i}" for i in range(len(probs))]
                else:
                    # binary: convert to two-class display
                    class_probs = probs.tolist()
                    class_names = ["No DR", "Diabetic Retinopathy"]

            except Exception as e:
                app.logger.warning(f"PREDICTION: failed: {e}")
                flash(f"Prediction failed: {e}", 'error')
                return redirect(request.url)

            # Build heatmap (Grad-CAM preferred; saliency fallback)
            heatmap_name = None
            heatmap_path = None
            try:
                last_conv = None
                try:
                    last_conv = find_best_conv_layer_name(model)
                except Exception as e:
                    app.logger.warning(f"GRADCAM: {e}")
                    last_conv = None

                heatmap_generated = False
                if last_conv is not None:
                    try:
                        h = make_gradcam_heatmap(arr, model, last_conv, pred_index=None)
                        heatmap_name = f"heatmap_{uuid.uuid4().hex}_{orig_fname}.png"
                        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_name)
                        apply_jet_colormap_and_overlay(save_path, h, heatmap_path, alpha=0.45)
                        heatmap_url = url_for('static', filename=f'uploads/{heatmap_name}')
                        heatmap_generated = True
                    except Exception as e:
                        app.logger.warning(f"GRADCAM: failed to build heatmap: {e}")

                if not heatmap_generated:
                    try:
                        sal = make_saliency_heatmap(arr, model, pred_index=None)
                        heatmap_name = f"heatmap_sal_{uuid.uuid4().hex}_{orig_fname}.png"
                        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_name)
                        apply_jet_colormap_and_overlay(save_path, sal, heatmap_path, alpha=0.5)
                        heatmap_url = url_for('static', filename=f'uploads/{heatmap_name}')
                        heatmap_generated = True
                    except Exception as e:
                        app.logger.warning(f"SALIENCY: failed: {e}")
                        heatmap_url = None

            except Exception as e:
                app.logger.warning(f"HEATMAP: unexpected error: {e}")
                heatmap_url = None

            # Save class bar chart if class_probs available
            chart_url = None
            if class_probs is not None and class_names is not None:
                try:
                    chart_name = f"classchart_{uuid.uuid4().hex}.png"
                    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], chart_name)
                    save_class_bar_chart(np.array(class_probs), class_names, chart_path)
                    chart_url = url_for('static', filename=f'uploads/{chart_name}')
                except Exception as e:
                    app.logger.warning(f"CHART: failed: {e}")
                    chart_url = None

            # Generate PDF report (best-effort) — pass patient metadata
            try:
                pdf_path = generate_pdf_report(save_path,
                                               heatmap_path if heatmap_path else None,
                                               prediction,
                                               probability,
                                               class_probs,
                                               class_names,
                                               patient_name=patient_name if patient_name else None,
                                               patient_sex=patient_sex if patient_sex else None)
                report_url = url_for('static', filename=f'uploads/{os.path.basename(pdf_path)}')
            except Exception as e:
                app.logger.warning(f"PDF: failed: {e}")
                report_url = None

            # Provide results to template
            return render_template('index.html',
                                   prediction=prediction,
                                   probability=probability,
                                   image_url=image_url,
                                   heatmap_url=heatmap_url,
                                   chart_url=chart_url,
                                   report_url=report_url,
                                   class_probs=class_probs,
                                   class_names=class_names)

        else:
            flash('Allowed image types: png, jpg, jpeg', 'error')
            return redirect(request.url)

    return render_template('index.html',
                           prediction=None,
                           probability=None,
                           image_url=None,
                           heatmap_url=None,
                           chart_url=None,
                           report_url=None,
                           class_probs=None,
                           class_names=None)

# Static route helper to download reports if needed (optional)
@app.route('/downloads/<path:filename>')
def downloads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    try:
        layers_info = [(layer.name, getattr(layer, 'output_shape', None)) for layer in model.layers]
        app.logger.info(f"MODEL LAYERS: {layers_info}")
    except Exception:
        app.logger.info("MODEL LAYERS: (could not fetch layer info)")
    app.run(host='0.0.0.0', port=5000, debug=True)
