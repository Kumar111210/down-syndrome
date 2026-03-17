import os
import sqlite3
import signal
from functools import wraps
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import base64

# Load .env from BACKEND directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
import secrets
import unicodedata
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import requests


from fpdf import FPDF
from openai import OpenAI
from xai.gradcam import analyze_heatmap_regions_by_face
import re
from uuid import uuid4
import json
import ast
import time

# ---------------- TIMEOUT HELPER ----------------
class TimeoutException(Exception):
    pass


def timeout(seconds=180, error_message="llm call timed out"):
    """Timeout decorator using SIGALRM (Linux/Render)."""
    def decorator(func):
        # On Windows (no SIGALRM), return the original function to avoid AttributeError
        if not hasattr(signal, "SIGALRM"):
            return func
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutException(error_message)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "model", "down_syndrome_modeltf29.h5")
DB_PATH = os.path.join(BASE_DIR, "users.db")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.6"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

IMG_SIZE = 224
# With include_top=False, MobileNetV2 last conv layer is block_16_project_BN
LAST_CONV_LAYER = "block_16_project_BN"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- JWT ----------------
# Fixed, deterministic fallback so local runs don't rotate secrets and break existing tokens.
# For production, always set JWT_SECRET_KEY/NMREC_JWT_SECRET via environment.
app.config["JWT_SECRET_KEY"] = (
    os.getenv("JWT_SECRET_KEY")
    or os.getenv("NMREC_JWT_SECRET")
    or "dev-local-jwt-secret"
)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(app)

@jwt.unauthorized_loader
def unauthorized_callback(reason):
    # Log for debugging bad/missing tokens
    try:
        print(f"[JWT unauthorized] {reason}")
    except Exception:
        pass
    return jsonify({"jwt_error": reason}), 401

@jwt.invalid_token_loader
def invalid_token_callback(reason):
    try:
        print(f"[JWT invalid] {reason}")
    except Exception:
        pass
    return jsonify({"jwt_error": reason}), 401

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    try:
        print(f"[JWT expired] {jwt_payload}")
    except Exception:
        pass
    return jsonify({"jwt_error": "Token expired"}), 401

# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    # Users table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT,
                profession TEXT
        )
    """)
    # Patient info table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT UNIQUE,
            patient_age INTEGER,
            mother_age INTEGER,
            father_age INTEGER,
            relation TEXT,
            income TEXT,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    # Screenings table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            patient_name TEXT,
            patient_age INTEGER,
            mother_age INTEGER,
            father_age INTEGER,
            parent_relation TEXT,
            living_area TEXT,
            previous_pregnancies INTEGER,
            date TEXT,
            uploaded_image TEXT,
            heatmap_image TEXT,
            probability REAL,
            confidence REAL,
            high_confidence INTEGER,
            facial_symmetry REAL,
            eye_spacing REAL,
            nasal_bridge REAL,
            ear_position REAL,
            explanation TEXT,
            recommended_food TEXT,
            recommended_exercises TEXT,
            risk_level TEXT,
            title TEXT,
            risk_factor_explanation TEXT,
            affected_region_explanation TEXT,
            future_health_explanation TEXT,
            food_explanation TEXT,
            exercise_explanation TEXT,
            video_titles TEXT,
            food_recommendations TEXT,
            exercise_recommendations TEXT,
            risk_factors TEXT,
            risk_factor_sections TEXT,
            risk_factor_highlights TEXT,
            affected_region_details TEXT,
            future_health_issues TEXT,
            explanation_sections TEXT,
            frontend_sections TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def migrate_patient_info_table():
    try:
        conn = get_db()
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(patient_info)").fetchall()]
        if "spouse_relation" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN spouse_relation TEXT")
        if "annual_income" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN annual_income TEXT")
        if "extra_info" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN extra_info TEXT")
        conn.commit()
        conn.close()
    except Exception:
        pass

migrate_patient_info_table()
def migrate_patient_info_extended_table():
    try:
        conn = get_db()
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(patient_info)").fetchall()]
        if "patient_name" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN patient_name TEXT")
        if "mother_health" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN mother_health TEXT")
        if "father_health" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN father_health TEXT")
        if "parent_relation" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN parent_relation TEXT")
        if "living_area" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN living_area TEXT")
        if "previous_pregnancies" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN previous_pregnancies INTEGER")
        if "pregnancy_complications" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN pregnancy_complications TEXT")
        if "family_history" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN family_history TEXT")
        if "family_genetic" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN family_genetic TEXT")
        if "family_genetic_details" not in cols:
            conn.execute("ALTER TABLE patient_info ADD COLUMN family_genetic_details TEXT")
        conn.commit()
        conn.close()
    except Exception:
        pass

migrate_patient_info_extended_table()
def migrate_users_table():
    try:
        conn = get_db()
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "profession" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN profession TEXT")
        conn.commit()
        conn.close()
    except Exception:
        pass

migrate_users_table()
def migrate_screenings_table():
    try:
        conn = get_db()
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(screenings)").fetchall()]
        if "patient_age" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN patient_age INTEGER")
        if "mother_age" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN mother_age INTEGER")
        if "father_age" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN father_age INTEGER")
        if "parent_relation" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN parent_relation TEXT")
        if "living_area" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN living_area TEXT")
        if "previous_pregnancies" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN previous_pregnancies INTEGER")
        if "recommended_food" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN recommended_food TEXT")
        if "recommended_exercises" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN recommended_exercises TEXT")
        if "risk_level" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN risk_level TEXT")
        if "title" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN title TEXT")
        if "risk_factor_explanation" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN risk_factor_explanation TEXT")
        if "affected_region_explanation" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN affected_region_explanation TEXT")
        if "future_health_explanation" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN future_health_explanation TEXT")
        if "food_explanation" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN food_explanation TEXT")
        if "exercise_explanation" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN exercise_explanation TEXT")
        if "video_titles" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN video_titles TEXT")
        if "food_recommendations" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN food_recommendations TEXT")
        if "exercise_recommendations" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN exercise_recommendations TEXT")
        if "risk_factors" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN risk_factors TEXT")
        if "risk_factor_sections" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN risk_factor_sections TEXT")
        if "risk_factor_highlights" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN risk_factor_highlights TEXT")
        if "affected_region_details" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN affected_region_details TEXT")
        if "future_health_issues" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN future_health_issues TEXT")
        if "explanation_sections" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN explanation_sections TEXT")
        if "frontend_sections" not in cols:
            conn.execute("ALTER TABLE screenings ADD COLUMN frontend_sections TEXT")
        conn.commit()
        conn.close()
    except Exception:
        pass
migrate_screenings_table()
# Using llm (local qwen2.5vl:latest) for explanations
# ---------------- GRAD-CAM ----------------
def _get_last_conv_layer(model):
    """Get last conv layer name - fallback if block_16_project_BN not found."""
    for name in ["block_16_project_BN", "block_16_project", "Conv_1"]:
        try:
            model.get_layer(name)
            return name
        except ValueError:
            continue
    # Fallback: find last Conv2D layer
    for layer in reversed(model.layers):
        if "Conv" in layer.name or "conv" in layer.name:
            return layer.name
    raise ValueError("No conv layer found in model")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    img_tensor = tf.convert_to_tensor(img_array)
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        last_conv_layer_name = _get_last_conv_layer(model)
        last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # Align Grad-CAM target with app probability convention:
        # DS probability is modeled as (1 - raw_output) for sigmoid outputs.
        out_shape = getattr(predictions, "shape", None)
        if out_shape is not None and len(out_shape) == 2 and int(out_shape[-1]) == 1:
            use_inverse = str(os.environ.get("GRADCAM_DS_INVERSE_OUTPUT", "1")).strip().lower() in {"1", "true", "yes"}
            loss = (1.0 - predictions[:, 0]) if use_inverse else predictions[:, 0]
        else:
            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[2], img_array.shape[1]))
    return np.uint8(255 * heatmap)

def save_heatmap(original_path, heatmap, output_path, face_rect=None, alpha=0.4):
    img = cv2.imread(original_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    if face_rect is None:
        try:
            face_rect = detect_face_rect(image=img)
        except Exception:
            face_rect = None
    hm = heatmap.copy()
    # Smooth heatmap to avoid blocky/square visual artifacts.
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=2.0, sigmaY=2.0)
    heatmap_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay_full = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    face_soft_mask = None
    face_hard_mask = None
    if face_rect:
        x, y, w, h = face_rect
        # Use a soft elliptical mask centered on face to avoid hard rectangular edges.
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        hard = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cx, cy = int(x + w / 2), int(y + h / 2)
        ax, ay = max(8, int(w * 0.52)), max(8, int(h * 0.66))
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        cv2.ellipse(hard, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        blur_k = max(21, int(max(w, h) * 0.35))
        if blur_k % 2 == 0:
            blur_k += 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        face_soft_mask = mask
        face_hard_mask = hard
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        combined = (img.astype(np.float32) * (1.0 - mask_f) + overlay_full.astype(np.float32) * mask_f).astype(np.uint8)
    else:
        combined = overlay_full

    # Highlight high-activation Grad-CAM zones so affected parts are clearly visible.
    try:
        p = float(np.percentile(hm, 78))
        thr = int(max(70, min(200, p)))
        hot_mask = (hm >= thr).astype(np.uint8) * 255
        if face_hard_mask is not None:
            hot_mask = cv2.bitwise_and(hot_mask, face_hard_mask)
        # If hotspot coverage is too sparse, relax threshold adaptively.
        hot_ratio = float(np.count_nonzero(hot_mask)) / float(hot_mask.size or 1)
        if hot_ratio < 0.003:
            thr2 = max(55, int(thr * 0.85))
            hot_mask = (hm >= thr2).astype(np.uint8) * 255
            if face_hard_mask is not None:
                hot_mask = cv2.bitwise_and(hot_mask, face_hard_mask)
        hot_mask = cv2.morphologyEx(
            hot_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        hot_mask = cv2.morphologyEx(
            hot_mask,
            cv2.MORPH_DILATE,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        hot_mask_bgr = cv2.cvtColor(hot_mask, cv2.COLOR_GRAY2BGR)
        # Bright yellow hotspot tint for affected zones.
        hotspot_color = np.zeros_like(combined)
        hotspot_color[:] = (0, 255, 255)
        hotspot_overlay = cv2.addWeighted(combined, 0.55, hotspot_color, 0.45, 0)
        combined = np.where(hot_mask_bgr > 0, hotspot_overlay, combined)

        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            cv2.drawContours(combined, [cnt], -1, (0, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass

    # Mark affected facial parts using curved outlines (no boxes/parameter labels).
    try:
        analysis = analyze_heatmap_regions(heatmap, face_rect=face_rect)
        affected_regions = analysis.get("affected_regions", []) if isinstance(analysis, dict) else []

        h_img, w_img = combined.shape[:2]
        if face_rect:
            x, y, w, h = face_rect
        else:
            x, y, w, h = 0, 0, w_img, h_img

        cx, cy = int(x + w / 2), int(y + h / 2)
        ax, ay = max(8, int(w * 0.52)), max(8, int(h * 0.66))
        face_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # Vertical bands in face coordinates; contours become curved via ellipse mask.
        bands = {
            "forehead": (0.00, 0.22),
            "eyes": (0.20, 0.42),
            "nose": (0.40, 0.62),
            "mouth": (0.60, 0.82),
            "chin": (0.80, 1.00),
            "general face": (0.00, 1.00),
        }
        for region in affected_regions or []:
            rr = _canonical_region_name(region)
            if rr not in bands:
                continue
            r0, r1 = bands[rr]
            y0 = max(0, min(h_img - 1, int(y + h * r0)))
            y1 = max(0, min(h_img - 1, int(y + h * r1)))
            if y1 <= y0:
                continue
            band_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            band_mask[y0:y1, :] = 255
            curved_region = cv2.bitwise_and(face_mask, band_mask)
            contours, _ = cv2.findContours(curved_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 120:
                    continue
                cv2.drawContours(combined, [cnt], -1, (0, 255, 255), 2, cv2.LINE_AA)
    except Exception:
        pass
    cv2.imwrite(output_path, combined)

def analyze_heatmap_regions(heatmap, face_rect=None):
    if face_rect:
        x, y, w, h = face_rect
        roi = heatmap[y:y+h, x:x+w]
    else:
        roi = heatmap
    h, w = roi.shape
    forehead = float(np.mean(roi[0:int(h*0.2), :]))/255.0
    eyes = float(np.mean(roi[int(h*0.2):int(h*0.4), :]))/255.0
    nose = float(np.mean(roi[int(h*0.4):int(h*0.6), :]))/255.0
    mouth = float(np.mean(roi[int(h*0.6):int(h*0.8), :]))/255.0
    chin = float(np.mean(roi[int(h*0.8):, :]))/255.0
    region_means = {
        "forehead": forehead,
        "eyes": eyes,
        "nose": nose,
        "mouth": mouth,
        "chin": chin
    }
    avg_mean = float(np.mean(list(region_means.values())))
    dynamic_threshold = max(0.12, min(0.45, avg_mean + 0.02))
    top_region = max(region_means.items(), key=lambda kv: kv[1])[0]
    affected = [r for r, v in region_means.items() if v >= dynamic_threshold]
    if region_means[top_region] > 0.12 and top_region not in affected:
        affected.append(top_region)
    if not affected:
        affected = ["general face"]
    return {
        "affected_regions": affected,
        "region_scores": {k: round(v, 3) for k, v in region_means.items()}
    }

def _canonical_region_name(value):
    s = str(value or "").strip().lower()
    if not s:
        return ""
    if "forehead" in s:
        return "forehead"
    if "eye" in s:
        return "eyes"
    if "nose" in s or "nasal" in s:
        return "nose"
    if "mouth" in s or "oral" in s or "lip" in s:
        return "mouth"
    if "chin" in s or "jaw" in s:
        return "chin"
    if "ear" in s:
        return "ears"
    if "face" in s:
        return "general face"
    return s


def _normalize_affected_regions(regions):
    out = []
    for r in (regions or []):
        rr = _canonical_region_name(r)
        if rr and rr not in out:
            out.append(rr)
    return out or ["general face"]


def _rank_regions_by_score(regions, region_scores):
    region_scores = region_scores if isinstance(region_scores, dict) else {}
    uniq = _normalize_affected_regions(regions)
    return sorted(
        uniq,
        key=lambda r: float(region_scores.get(_canonical_region_name(r), -1.0)),
        reverse=True,
    )


def _merge_affected_regions(face_regions, score_regions, region_scores):
    """Merge detector and score-based regions, prioritizing stronger score regions."""
    fr = _normalize_affected_regions(face_regions)
    sr = _normalize_affected_regions(score_regions)
    if fr == ["general face"]:
        return _rank_regions_by_score(sr, region_scores)
    merged = []
    for r in fr + sr:
        rr = _canonical_region_name(r)
        if rr and rr not in merged:
            merged.append(rr)
    return _rank_regions_by_score(merged, region_scores)


def _region_recommendations(affected, risk_level):
    if risk_level == "low":
        return "Normal balanced diet", "Regular physical activity"
    if any(x in affected for x in ["eyes", "forehead"]):
        food = "Omega-3 rich foods, leafy greens, fruits"
        exercises = "Eye focus drills, gentle neck stretches, breathing"
    elif "nose" in affected:
        food = "Balanced protein intake, hydration, vitamin C"
        exercises = "Facial yoga for nasal area, diaphragmatic breathing"
    elif any(x in affected for x in ["mouth", "chin"]):
        food = "Soft-texture balanced diet if needed, multivitamins"
        exercises = "Orofacial myofunctional routines, light jaw stretches"
    else:
        food = "Balanced diet"
        exercises = "Light exercises"
    if risk_level == "high":
        exercises = exercises + ", supervised physiotherapy"
    return food, exercises

# ---------------- FACIAL FEATURES ----------------
def analyze_facial_features(heatmap, prob, face_rect=None):
    if face_rect:
        x, y, w, h = face_rect
        roi = heatmap[y:y+h, x:x+w]
    else:
        roi = heatmap
    h, w = roi.shape
    forehead = float(np.mean(roi[0:int(h*0.2), :]))/255.0
    eyes = float(np.mean(roi[int(h*0.2):int(h*0.4), :]))/255.0
    nose = float(np.mean(roi[int(h*0.4):int(h*0.6), :]))/255.0
    mouth = float(np.mean(roi[int(h*0.6):int(h*0.8), :]))/255.0
    chin = float(np.mean(roi[int(h*0.8):, :]))/255.0
    left_mean = float(np.mean(roi[:, :int(w*0.5)]))/255.0
    right_mean = float(np.mean(roi[:, int(w*0.5):]))/255.0
    sym_raw = 1.0 - abs(left_mean - right_mean)
    facial_symmetry = float(np.clip(sym_raw, 0.3, 1.0))
    eye_spacing = float(np.clip(0.3 + 0.5 * eyes, 0.3, 0.8))
    nasal_bridge = float(np.clip(0.4 + 0.5 * nose, 0.4, 0.9))
    ear_position = float(np.clip(0.4 + 0.3 * (forehead - chin + 0.5), 0.4, 0.9))
    features = {
        "facial_symmetry": round(facial_symmetry, 2),
        "eye_spacing": round(eye_spacing, 2),
        "nasal_bridge": round(nasal_bridge, 2),
        "ear_position": round(ear_position, 2),
    }
    theory = {
        "facial_symmetry": "Symmetry of the left and right sides of the face.",
        "eye_spacing": "Distance between eyes relative to face width.",
        "nasal_bridge": "Proportion and height of the nasal bridge.",
        "ear_position": "Vertical alignment of ears relative to eyes."
    }
    return features, theory

def _compute_region_score(heatmap, face_rect=None):
    if face_rect:
        x, y, w, h = face_rect
        roi = heatmap[y:y+h, x:x+w]
    else:
        roi = heatmap
    h, w = roi.shape
    forehead = float(np.mean(roi[0:int(h*0.2), :]))/255.0
    eyes = float(np.mean(roi[int(h*0.2):int(h*0.4), :]))/255.0
    nose = float(np.mean(roi[int(h*0.4):int(h*0.6), :]))/255.0
    mouth = float(np.mean(roi[int(h*0.6):int(h*0.8), :]))/255.0
    chin = float(np.mean(roi[int(h*0.8):, :]))/255.0
    return max(forehead, eyes, nose, mouth, chin)

def calibrate_probability(prob, heatmap, face_rect=None):
    """Safer calibration: use region score to scale probability without collapsing low values."""
    region_score = _compute_region_score(heatmap, face_rect=face_rect)
    # Safer scaling (does not destroy low probabilities)
    adjusted = prob * (0.6 + 0.4 * region_score)
    return float(np.clip(adjusted, 0.0, 1.0))


def generate_local_explanation(prob_adj, status, affected_regions, region_scores, features, food_rec, exercise_rec, patient_context, confidence):
    """Generate a warm, concise explanation using available numeric and contextual data.
    This is used when a local LLM (llm) is unavailable.
    """
    try:
        pct = int(round(prob_adj * 100))
    except Exception:
        pct = int(prob_adj * 100) if prob_adj is not None else 0

    main_regions = ', '.join(affected_regions) if affected_regions else 'face'
    region_lines = []
    region_note = {
        "forehead": "Forehead activation may reflect proportional facial pattern emphasis.",
        "eyes": "Eye-region activation can relate to spacing and peri-orbital facial cues.",
        "nose": "Nasal-bridge activation suggests the model focused on mid-face structure.",
        "mouth": "Mouth-region activation can reflect lip and oral-facial contour cues.",
        "chin": "Chin activation may reflect lower-face shape and jawline proportion.",
        "ears": "Ear-region activation may reflect ear position and alignment patterns.",
        "general face": "General face activation indicates broad facial pattern attention.",
    }
    for r in (affected_regions or ["general face"]):
        k = str(r).strip().lower()
        k = "eyes" if k == "eye" else ("ears" if k == "ear" else k)
        if k in region_note:
            score_txt = ""
            try:
                if isinstance(region_scores, dict) and k in region_scores:
                    score_txt = f" (score {round(float(region_scores.get(k)), 3)})"
            except Exception:
                score_txt = ""
            region_lines.append(f"{k.capitalize()}: {region_note[k]}{score_txt}")
    regions_block = " ".join(region_lines[:5])
    # Sentence 1: summary
    s1 = f"Our automated screening estimates a {status.lower()} likelihood ({pct}%) of features associated with Down syndrome traits."
    # Sentence 2: what the model looked at
    s2 = f"The model focused most on the {main_regions}, which can sometimes be associated with characteristic facial patterns."
    s2b = f"Region-wise brief summary: {regions_block}" if regions_block else ""
    # Sentence 3: feature risk and health challenges to monitor
    risk_health_note = {
        "high": "With this higher screening signal, monitor common associated challenges such as delayed speech, low muscle tone, feeding difficulty, and slower developmental milestones, and seek early specialist support.",
        "moderate": "With this moderate screening signal, monitor possible developmental concerns such as speech delay, muscle tone differences, feeding issues, or slower milestone progress, and plan pediatric follow-up.",
        "low": "This is a lower screening signal, but continue routine monitoring for speech, feeding, growth, and developmental milestones during regular pediatric visits.",
    }
    s3 = risk_health_note.get(str(status or "").strip().lower(), risk_health_note["low"])
    # Sentence 4: reassurance + next steps
    s4 = "This screening is not a diagnosis. We recommend seeing a pediatrician or genetic counselor for a formal evaluation and, if appropriate, diagnostic testing."
    # Sentence 4: practical suggestions
    s5 = f"Supportive steps you can consider now: {food_rec}. Gentle activities: {exercise_rec}." if (food_rec or exercise_rec) else "Consider maintaining a balanced diet and regular gentle activity appropriate for your child's age."
    # Sentence 5: optional detail about confidence
    conf_note = "" if confidence is None else (" The result is moderately confident." if confidence < 0.8 else " The result has higher confidence.")

    parts = [s1.strip(), s2.strip()]
    if s2b:
        parts.append(s2b.strip())
    parts.extend([s3.strip(), s4.strip(), s5.strip()])
    explanation = " ".join(parts) + conf_note
    # Keep it under ~200 words
    return explanation.strip()


def _extract_json_payload(text):
    """Best-effort parse for model output that may wrap JSON in markdown fences."""
    if not text:
        return None
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", candidate)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        pass
    # Some small VLMs return Python-like dict/list strings instead of strict JSON.
    try:
        py_obj = ast.literal_eval(candidate)
    except Exception:
        return None

    if isinstance(py_obj, dict):
        return py_obj

    if isinstance(py_obj, list) and py_obj:
        first = py_obj[0]
        if isinstance(first, dict):
            # Handle patterns like [{'keys': [...], 'vals': [...]}]
            keys = first.get("keys")
            if isinstance(keys, list):
                return {
                    "title": "**AI Screening Summary**",
                    "explanation": "The AI model analyzed the provided Grad-CAM image and patient details. Please review this result with a medical professional.",
                    "affected_regions": [str(k) for k in keys[:5]],
                    "food_recommendations": [],
                    "exercise_recommendations": [],
                    "video_titles": [],
                }
            return first
    return None


def _coerce_text_payload(text, fallback_regions=None):
    """Convert free-form LLM text into the expected payload schema."""
    if not text:
        return None

    lines = [ln.strip(" -*\t") for ln in text.splitlines() if ln.strip()]
    lower = text.lower()

    def _is_junk_line(s):
        sl = s.lower().strip()
        if not sl:
            return True
        if sl in ["[", "]", "{", "}", ","]:
            return True
        if any(tok in sl for tok in ['"title":', '"explanation":', '"food_recommendations":', '"exercise_recommendations":', '"video_titles":']):
            return True
        alpha = sum(ch.isalpha() for ch in sl)
        return alpha < 3

    # Title: first markdown heading or first meaningful line.
    title = ""
    for ln in lines:
        if _is_junk_line(ln):
            continue
        if ln.startswith("**") and ln.endswith("**") and len(ln.strip("* ").strip()) >= 3:
            title = ln
            break
    if not title and lines:
        first_good = next((ln for ln in lines if not _is_junk_line(ln)), "")
        raw = first_good or "AI Screening Summary"
        title = raw if raw.startswith("**") else f"**{raw[:80]}**"

    region_vocab = ["forehead", "eyes", "eye", "nose", "mouth", "chin", "ear", "ears", "general face"]
    found_regions = []
    for r in region_vocab:
        if re.search(rf"\b{re.escape(r)}\b", lower):
            canon = "eyes" if r == "eye" else ("ears" if r == "ear" else r)
            if canon not in found_regions:
                found_regions.append(canon)
    if not found_regions:
        found_regions = list(fallback_regions or ["general face"])

    # Very lightweight extraction for food/exercise/video-like lines
    food_recs = []
    ex_recs = []
    videos = []
    for ln in lines:
        lnl = ln.lower()
        if any(k in lnl for k in ["food", "diet", "nutrition", "vitamin", "protein", "fruit", "vegetable"]):
            food_recs.append(ln)
        if any(k in lnl for k in ["exercise", "stretch", "activity", "walk", "physio", "yoga"]):
            ex_recs.append(ln)
        if any(k in lnl for k in ["video", "youtube", "routine"]):
            videos.append(ln)

    if not food_recs:
        food_recs = []
    if not ex_recs:
        ex_recs = []
    if not videos:
        videos = []

    explanation = text.strip()
    # If output contains serialized JSON-like markers, try to recover explanation
    # instead of replacing with generic fallback text.
    if any(tok in explanation.lower() for tok in ['"title":', '"food_recommendations":', '"exercise_recommendations":', '"video_titles":']):
        extracted = _extract_json_payload(explanation)
        if isinstance(extracted, dict):
            extracted_text = str(
                extracted.get("explanation")
                or extracted.get("summary")
                or extracted.get("description")
                or ""
            ).strip()
            if extracted_text:
                explanation = extracted_text
    explanation = _briefen_explanation(explanation, max_words=600)

    return {
        "title": title or "**AI Screening Summary**",
        "explanation": explanation,
        "affected_regions": found_regions[:5],
        "food_recommendations": food_recs[:4],
        "exercise_recommendations": ex_recs[:4],
        "video_titles": videos[:3],
    }


def _is_weak_llm_payload(payload):
    """Detect generic/visual-only responses that ignore clinical context."""
    if not payload:
        return True
    explanation = str(payload.get("explanation") or "").lower()
    regions = payload.get("affected_regions") or []
    food = payload.get("food_recommendations") or []
    exercises = payload.get("exercise_recommendations") or []
    videos = payload.get("video_titles") or []

    weak_visual_tokens = [
        "blue and white stripes",
        "striped mask",
        "heatmap overlay",
        "looks like",
        "appears to show",
    ]
    has_visual_only_hint = any(tok in explanation for tok in weak_visual_tokens)
    lacks_substance = (len(explanation) < 70) or (len(regions) == 0) or (len(food) == 0) or (len(exercises) == 0)
    no_video_ideas = len(videos) == 0
    too_sparse = (len(regions) == 0 and len(food) == 0 and len(exercises) == 0)
    return has_visual_only_hint or too_sparse or (lacks_substance and no_video_ideas)

def _has_structured_sections(payload):
    if not isinstance(payload, dict):
        return False
    region_details = payload.get("affected_region_details") or []
    future_issues = payload.get("future_health_issues") or []
    rf = payload.get("risk_factor_sections") or {}
    grad = rf.get("gradcam") if isinstance(rf, dict) else []
    face = rf.get("facial_features") if isinstance(rf, dict) else []
    has_rf = bool(grad) or bool(face) or bool(payload.get("risk_factors"))
    return bool(region_details) and bool(future_issues) and has_rf


def _limit_text(value, max_chars=900):
    s = str(value or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0].strip()


def _to_clean_list(value, max_items=5):
    items = []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.split(r"[\n,;]", value)
    else:
        raw_items = []

    for it in raw_items:
        s = str(it).strip(" -*\t")
        if not s:
            continue
        # Drop obvious malformed templating/serialized fragments.
        if any(tok in s for tok in [".join(", "{", "}", "['", "']", '"title":', "'title':", '"explanation":', "'explanation':", '"food_recommendations":', "'food_recommendations':", '"exercise_recommendations":', "'exercise_recommendations':", '"video_titles":', "'video_titles':"]):
            continue
        if len(s) < 2:
            continue
        items.append(s)

    dedup = []
    seen = set()
    for it in items:
        k = it.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(it)
        if len(dedup) >= max_items:
            break
    return dedup


def _sanitize_title(title_text, status):
    s = str(title_text or "").strip()
    if not s:
        return f"**{status} Risk Screening Summary**"
    low = s.lower()
    if "ai screening summary" in low or "model-screening" in low:
        return f"**{status} Risk Screening Summary**"
    junk_tokens = [
        "forehead", "eye", "eyes", "nose", "mouth", "chin", "ear", "ears",
        "food recommendations", "exercise recommendations", ",",
    ]
    token_hits = sum(1 for t in junk_tokens if t in low)
    if token_hits >= 4 or len(s) < 6:
        return f"**{status} Risk Screening Summary**"
    if not s.startswith("**"):
        s = f"**{s}**"
    return s


def _explanation_conflicts_status(explanation_text, status):
    txt = str(explanation_text or "").lower()
    # Serialized/object-like payload leaked into text.
    if any(tok in txt for tok in ["{'title':", '"title":', "'affected_regions':", '"affected_regions":', "['", "],"]):
        return True
    if "low probability" in txt and status.lower() in ["high", "moderate"]:
        return True
    if "high probability" in txt and status.lower() == "low":
        return True
    if "low risk" in txt and status.lower() == "high":
        return True
    if "high risk" in txt and status.lower() == "low":
        return True
    return False


def _payload_uses_context(payload, status, prob_adj, affected_regions):
    """Check if LLM output reflects this specific screening context."""
    if not isinstance(payload, dict):
        return False
    explanation = str(payload.get("explanation") or "").lower()
    title = str(payload.get("title") or "").lower()
    text = f"{title} {explanation}"
    pct = int(round(float(prob_adj) * 100)) if prob_adj is not None else None

    mentions_status = status.lower() in text
    mentions_pct = (pct is not None) and (f"{pct}%" in text or f"{pct} %" in text)

    expected_regions = []
    for r in (affected_regions or []):
        rr = _canonical_region_name(r)
        if rr:
            expected_regions.append(rr)
    mentions_region = any(r in text for r in expected_regions)

    # Structured anchors from payload fields (not just prose text).
    payload_regions = []
    for r in (payload.get("affected_regions") or []):
        rr = _canonical_region_name(r)
        if rr:
            payload_regions.append(rr)
    region_overlap = bool(set(expected_regions) & set(payload_regions)) if expected_regions else bool(payload_regions)

    has_food = bool(payload.get("food_recommendations"))
    has_exercises = bool(payload.get("exercise_recommendations"))
    has_video_titles = bool(payload.get("video_titles"))

    # Grounded if text has at least one core anchor and payload has clinical structure.
    text_anchors = sum([bool(mentions_status), bool(mentions_pct), bool(mentions_region)])
    structured_grounding = region_overlap and has_food and has_exercises
    return (text_anchors >= 1 and structured_grounding) or (text_anchors >= 2) or (structured_grounding and has_video_titles)


def _briefen_explanation(text, max_words=600):
    s = str(text or "").strip()
    if not s:
        return s
    if max_words is None:
        return s
    words = s.split()
    if len(words) <= max_words:
        return s
    clipped = " ".join(words[:max_words]).rstrip(" ,;:")
    if not clipped.endswith((".", "!", "?")):
        clipped += "."
    return clipped


def _word_count(text):
    return len(str(text or "").split())


def _build_full_explanation(
    base_explanation,
    status,
    prob_adj,
    affected_regions,
    region_scores,
    features,
    patient_context,
    food_recommendations=None,
    exercise_recommendations=None,
    video_titles=None,
):
    """Build a concise, case-grounded explanation without generic boilerplate."""
    pct = int(round(float(prob_adj) * 100)) if prob_adj is not None else 0
    intro = (
        f"This screening indicates a {status.lower()} likelihood ({pct}%) based on the uploaded image and Grad-CAM pattern analysis."
    )

    region_note = {
        "forehead": "Forehead emphasis suggests upper-face proportional cues were relevant.",
        "eyes": "Eye-region emphasis suggests peri-orbital spacing cues were relevant.",
        "nose": "Nasal emphasis suggests mid-face and bridge-related cues were relevant.",
        "mouth": "Mouth emphasis suggests lower mid-face contour cues were relevant.",
        "chin": "Chin emphasis suggests jawline/lower-face proportion cues were relevant.",
        "ears": "Ear-region emphasis suggests lateral alignment cues were relevant.",
        "general face": "General-face emphasis indicates broader facial pattern attention.",
    }
    region_lines = []
    for r in (affected_regions or ["general face"]):
        rr = _canonical_region_name(r)
        if rr in region_note:
            score = ""
            try:
                if isinstance(region_scores, dict) and rr in region_scores:
                    score = f" (score {round(float(region_scores.get(rr)), 3)})"
            except Exception:
                score = ""
            region_lines.append(f"{rr.capitalize()}: {region_note[rr]}{score}")
    # Cover all detected regions so the explanation does not omit facial areas.
    region_block = " ".join(region_lines[:6])

    status_l = str(status or "").strip().lower()
    risk_interpretation = {
        "high": "Feature risk interpretation: the combined facial feature pattern suggests a stronger screening-level signal.",
        "moderate": "Feature risk interpretation: the facial feature pattern suggests a moderate screening-level signal.",
        "low": "Feature risk interpretation: the facial feature pattern suggests a lower screening-level signal.",
    }
    health_challenges = {
        "high": "Health challenges to monitor: possible speech delay, low muscle tone, feeding difficulty, and slower milestone progress.",
        "moderate": "Health challenges to monitor: possible mild-to-moderate speech, feeding, tone, or developmental milestone delays.",
        "low": "Health challenges to monitor: continue routine monitoring of speech, feeding, growth, and developmental milestones.",
    }
    risk_block = risk_interpretation.get(status_l, risk_interpretation["low"])
    health_block = health_challenges.get(status_l, health_challenges["low"])
    food_line = ""
    exercise_line = ""
    if food_recommendations:
        if isinstance(food_recommendations, list):
            food_line = ", ".join([str(x).strip() for x in food_recommendations if str(x).strip()][:2])
        else:
            food_line = str(food_recommendations).strip()
    if exercise_recommendations:
        if isinstance(exercise_recommendations, list):
            exercise_line = ", ".join([str(x).strip() for x in exercise_recommendations if str(x).strip()][:2])
        else:
            exercise_line = str(exercise_recommendations).strip()
    overcome_block = ""
    if food_line or exercise_line:
        overcome_block = (
            f"How to overcome/support: Food - {food_line or 'Maintain balanced nutrition'}. "
            f"Exercise - {exercise_line or 'Daily supervised gentle activity'}."
        )
    action_block = (
        "This is a screening output and not a final diagnosis. Please consult a pediatrician or genetic specialist for clinical confirmation."
    )

    base = str(base_explanation or "").strip()
    # Remove repetitive generic model-preface lines from LLM output.
    generic_patterns = [
        r"(?i)\bthe ai model reviewed[^.]*\.",
        r"(?i)\bplease consult a healthcare professional[^.]*\.",
        r"(?i)\bthis screening result indicates[^.]*\.",
    ]
    for pat in generic_patterns:
        base = re.sub(pat, "", base).strip()
    base = re.sub(r"\s+", " ", base).strip(" ,;")

    combined = " ".join(
        p
        for p in [
            base,
            intro,
            region_block,
            risk_block,
            health_block,
            overcome_block,
            action_block,
        ]
        if p
    ).strip()
    return _briefen_explanation(combined, max_words=180)


def _ensure_region_coverage(explanation_text, affected_regions, region_scores=None):
    """Ensure explanation explicitly covers each detected affected region."""
    text = str(explanation_text or "").strip()
    regions = []
    for r in (affected_regions or []):
        rr = _canonical_region_name(r)
        if rr and rr not in regions:
            regions.append(rr)
    if not regions:
        regions = ["general face"]

    notes = {
        "forehead": "Forehead attention suggests upper-face proportion cues were significant.",
        "eyes": "Eye-region attention suggests peri-orbital spacing and alignment cues were significant.",
        "nose": "Nasal attention suggests bridge and mid-face structural cues were significant.",
        "mouth": "Mouth-region attention suggests lower mid-face contour cues were significant.",
        "chin": "Chin attention suggests lower-face and jawline proportion cues were significant.",
        "ears": "Ear-region attention suggests lateral facial alignment cues were significant.",
        "general face": "General-face attention indicates broad facial pattern cues were significant.",
    }

    missing_lines = []
    low_text = text.lower()
    for rr in regions:
        if rr in low_text:
            continue
        score_txt = ""
        try:
            if isinstance(region_scores, dict) and rr in region_scores:
                score_txt = f" (score {round(float(region_scores.get(rr)), 3)})"
        except Exception:
            score_txt = ""
        missing_lines.append(f"{rr.capitalize()}: {notes.get(rr, notes['general face'])}{score_txt}")

    if not missing_lines:
        return text
    if text and not text.endswith((".", "!", "?")):
        text = text + "."
    return (text + " " + " ".join(missing_lines)).strip()


def _risk_band(v):
    try:
        x = float(v)
    except Exception:
        return "unknown"
    if x >= 0.60:
        return "high"
    if x >= 0.35:
        return "moderate"
    return "low"


def _build_gradcam_risk_factors(region_scores):
    """Create structured Grad-CAM risk factors from region activation scores."""
    if not isinstance(region_scores, dict) or not region_scores:
        return []
    cleaned = []
    for k, v in region_scores.items():
        rr = str(k).strip().lower()
        rr = "eyes" if rr == "eye" else ("ears" if rr == "ear" else rr)
        try:
            vv = float(v)
        except Exception:
            continue
        if vv < 0:
            continue
        cleaned.append((rr, vv))
    if not cleaned:
        return []
    total = sum(v for _, v in cleaned) or 1.0
    cleaned.sort(key=lambda x: x[1], reverse=True)
    factors = []
    for rr, vv in cleaned:
        factors.append(
            {
                "feature": rr,
                "activation_score": round(vv, 3),
                "contribution_pct": round((vv / total) * 100.0, 1),
                "severity": _risk_band(vv),
            }
        )
    return factors


def _build_region_risk_matrix(region_scores, affected_regions):
    """Return full per-region matrix including affected flag and severity."""
    base_regions = ["forehead", "eyes", "nose", "mouth", "chin"]
    scores = region_scores if isinstance(region_scores, dict) else {}
    affected = set(_normalize_affected_regions(affected_regions))
    out = []
    for r in base_regions:
        val = float(scores.get(r, 0.0))
        out.append(
            {
                "region": r,
                "activation_score": round(val, 3),
                "severity": _risk_band(val),
                "affected": r in affected,
            }
        )
    return sorted(out, key=lambda x: x["activation_score"], reverse=True)


def _ensure_feature_risk_coverage(explanation_text, affected_regions, region_scores=None, features=None):
    """Ensure explanation includes affected-feature intensity and risk-factor summary."""
    text = str(explanation_text or "").strip()
    lower = text.lower()

    region_order = ["forehead", "eyes", "nose", "mouth", "chin", "ears", "general face"]
    regions = []
    for r in (affected_regions or []):
        rr = _canonical_region_name(r)
        if rr and rr not in regions:
            regions.append(rr)
    if not regions:
        regions = ["general face"]

    if "risk-factor detail" not in lower:
        region_parts = []
        for rr in region_order:
            if rr not in regions:
                continue
            score = None
            try:
                if isinstance(region_scores, dict) and rr in region_scores:
                    score = float(region_scores.get(rr))
            except Exception:
                score = None
            if score is None:
                region_parts.append(f"{rr} (intensity not available)")
            else:
                region_parts.append(f"{rr} {round(score, 3)} ({_risk_band(score)})")
        if region_parts:
            text += (" " if text else "") + "Risk-factor detail (affected-region intensity 0 to 1): " + "; ".join(region_parts) + "."

    if "facial feature factors" not in lower:
        feat = features if isinstance(features, dict) else {}
        f1 = feat.get("facial_symmetry")
        f2 = feat.get("eye_spacing")
        f3 = feat.get("nasal_bridge")
        f4 = feat.get("ear_position")
        if any(v is not None for v in [f1, f2, f3, f4]):
            def _fmt(name, value):
                if value is None:
                    return f"{name}: N/A"
                return f"{name}: {round(float(value), 2)} ({_risk_band(value)})"
            text += (
                " " if text else ""
            ) + "Facial feature factors: " + ", ".join(
                [
                    _fmt("symmetry", f1),
                    _fmt("eye spacing", f2),
                    _fmt("nasal bridge", f3),
                    _fmt("ear position", f4),
                ]
            ) + "."

    return text.strip()


def _build_affected_region_details(affected_regions, region_scores=None):
    """Structured per-region explanation for frontend display."""
    details_map = {
        "forehead": "Upper-face proportion patterns were highlighted in this region.",
        "eyes": "Peri-orbital spacing/alignment patterns were highlighted in this region.",
        "nose": "Mid-face and nasal-bridge structure patterns were highlighted in this region.",
        "mouth": "Lower mid-face/oral contour patterns were highlighted in this region.",
        "chin": "Lower-face and jawline proportion patterns were highlighted in this region.",
        "ears": "Lateral facial alignment near ear level was highlighted in this region.",
        "general face": "Broad facial pattern attention was observed across the face.",
    }
    out = []
    seen = set()
    for r in (affected_regions or ["general face"]):
        rr = _canonical_region_name(r)
        if not rr or rr in seen:
            continue
        seen.add(rr)
        score = None
        try:
            if isinstance(region_scores, dict) and rr in region_scores:
                score = float(region_scores.get(rr))
        except Exception:
            score = None
        out.append(
            {
                "region": rr,
                "activation_score": (round(score, 3) if score is not None else None),
                "affected_pct": (round(float(score) * 100.0, 1) if score is not None else None),
                "severity": (_risk_band(score) if score is not None else "unknown"),
                "explanation": details_map.get(rr, details_map["general face"]),
            }
        )
    return out or [
        {
            "region": "general face",
            "activation_score": None,
            "affected_pct": None,
            "severity": "unknown",
            "explanation": details_map["general face"],
        }
    ]


def _is_truthy_health_flag(value):
    s = str(value or "").strip().lower()
    if not s:
        return False
    if s in {"no", "none", "nil", "na", "n/a", "false", "0"}:
        return False
    return True


def _build_future_health_issues(status, affected_regions, features=None, prob_adj=None, patient_info=None):
    """Return monitorable health issues using prediction + patient background."""
    level = str(status or "").strip().lower()
    try:
        p = float(prob_adj) if prob_adj is not None else None
    except Exception:
        p = None
    if p is not None:
        if p >= 0.8:
            level = "high"
        elif p >= 0.45 and level == "low":
            level = "moderate"

    pinfo = patient_info if isinstance(patient_info, dict) else {}
    patient_age = _to_int_or_none(pinfo.get("patient_age"))
    mother_age = _to_int_or_none(pinfo.get("mother_age"))
    father_age = _to_int_or_none(pinfo.get("father_age"))
    prev_preg = _to_int_or_none(pinfo.get("previous_pregnancies"))
    preg_comp = str(pinfo.get("pregnancy_complications") or "").strip()
    fam_hist = str(pinfo.get("family_history") or "").strip()
    fam_gen = str(pinfo.get("family_genetic") or "").strip()
    mother_health = str(pinfo.get("mother_health") or "").strip()
    father_health = str(pinfo.get("father_health") or "").strip()

    base = []
    if level == "high":
        base = [
            ("Speech and language delay", "Consider early speech and language support.", "high"),
            ("Low muscle tone", "Track posture, motor milestones, and physiotherapy needs.", "high"),
            ("Feeding or swallowing difficulty", "Monitor feeding tolerance and growth trend.", "high"),
            ("Slower developmental milestones", "Use periodic developmental screening follow-up.", "high"),
        ]
    elif level == "moderate":
        base = [
            ("Mild speech delay risk", "Track communication milestones and early interventions.", "moderate"),
            ("Motor tone and coordination concern", "Observe gross/fine motor progression.", "moderate"),
            ("Feeding pattern concern", "Track appetite, chewing, and weight trend.", "moderate"),
            ("Learning/development pace variation", "Plan regular pediatric developmental review.", "moderate"),
        ]
    else:
        base = [
            ("Routine developmental monitoring", "Continue milestone checks at routine visits.", "low"),
            ("Speech and feeding observation", "Watch for any persistent delay signs over time.", "low"),
        ]

    region_set = set(_normalize_affected_regions(affected_regions))
    if "mouth" in region_set:
        base.append(("Oral-motor articulation concern", "Watch speech clarity and oral-motor control.", level or "low"))
    if "nose" in region_set:
        base.append(("Breathing/sleep quality observation", "Track persistent nasal obstruction or sleep issues.", level or "low"))
    if "eyes" in region_set:
        base.append(("Vision alignment observation", "Schedule eye check if alignment concerns appear.", level or "low"))

    feat = features if isinstance(features, dict) else {}
    try:
        symmetry = float(feat.get("facial_symmetry")) if feat.get("facial_symmetry") is not None else None
    except Exception:
        symmetry = None
    if symmetry is not None and symmetry < 0.55:
        base.append(("Asymmetry follow-up", "Consider specialist review if asymmetry persists clinically.", "moderate"))

    # Patient-context modifiers.
    if patient_age is not None and patient_age <= 2:
        base.append(("Early developmental surveillance", "Use frequent milestone checks during infancy/toddler period.", level or "low"))
    ctx_sev = "high" if level == "high" else "moderate"
    if mother_age is not None and mother_age >= 35:
        base.append(("Maternal age risk context", "Ensure continued pediatric/genetic follow-up due to advanced maternal age context.", ctx_sev))
    if father_age is not None and father_age >= 40:
        base.append(("Paternal age risk context", "Track neurodevelopment and growth with periodic review.", ctx_sev))
    if prev_preg is not None and prev_preg >= 3:
        base.append(("Higher obstetric-history context", "Discuss cumulative pregnancy history during follow-up visits.", "moderate"))
    if _is_truthy_health_flag(preg_comp):
        base.append(("Pregnancy complication follow-up", f"Monitor development closely given reported pregnancy complications ({preg_comp}).", "moderate"))
    if _is_truthy_health_flag(fam_hist):
        base.append(("Family history follow-up", f"Review relevant family history in pediatric/genetic consultation ({fam_hist}).", "moderate"))
    if _is_truthy_health_flag(fam_gen):
        base.append(("Genetic history context", f"Prioritize genetic counseling due to reported family genetic history ({fam_gen}).", "high" if level == "high" else "moderate"))
    if _is_truthy_health_flag(mother_health):
        base.append(("Maternal health context", f"Consider maternal health factors in follow-up planning ({mother_health}).", "moderate"))
    if _is_truthy_health_flag(father_health):
        base.append(("Paternal health context", f"Consider paternal health factors in follow-up planning ({father_health}).", "moderate"))

    out = []
    seen = set()
    for issue, monitor, sev in base:
        key = issue.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"issue": issue, "what_to_monitor": monitor, "risk_level": sev})
    return out[:8]


def _build_risk_factor_sections(gradcam_risk_factors, features):
    feat = features if isinstance(features, dict) else {}
    facial = []
    for key, label in [
        ("facial_symmetry", "facial_symmetry"),
        ("eye_spacing", "eye_spacing"),
        ("nasal_bridge", "nasal_bridge"),
        ("ear_position", "ear_position"),
    ]:
        val = feat.get(key)
        if val is None:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        facial.append(
            {
                "feature": label,
                "value": round(fv, 2),
                "severity": _risk_band(fv),
            }
        )
    return {
        "gradcam": list(gradcam_risk_factors or []),
        "facial_features": facial,
    }

def _build_risk_factor_highlights(risk_factor_sections, top_n=4, affected_region_details=None):
    """Create frontend-ready highlight cards from structured risk factors."""
    sev_rank = {"high": 3, "moderate": 2, "low": 1, "unknown": 0}
    sev_color = {
        "high": "#E74C3C",
        "moderate": "#F1C40F",
        "low": "#2ECC71",
        "unknown": "#95A5A6",
    }
    out = []
    rfs = risk_factor_sections if isinstance(risk_factor_sections, dict) else {}
    region_map = {}
    for r in (affected_region_details or []):
        if not isinstance(r, dict):
            continue
        key = _canonical_region_name(r.get("region"))
        if not key:
            continue
        region_map[key] = r

    for g in (rfs.get("gradcam") or []):
        if not isinstance(g, dict):
            continue
        feature = str(g.get("feature") or "").strip().lower()
        if not feature:
            continue
        severity = str(g.get("severity") or "unknown").strip().lower()
        if severity not in sev_rank:
            severity = "unknown"
        rd = region_map.get(feature) or {}
        apct = rd.get("affected_pct")
        ascore = rd.get("activation_score")
        how_much = ""
        if apct is not None:
            how_much = f" Affected about {apct}%."
        elif ascore is not None:
            try:
                how_much = f" Affected intensity {round(float(ascore) * 100, 1)}%."
            except Exception:
                how_much = ""
        out.append(
            {
                "type": "gradcam",
                "feature": feature,
                "severity": severity,
                "color": sev_color.get(severity, sev_color["unknown"]),
                "activation_score": g.get("activation_score"),
                "contribution_pct": g.get("contribution_pct"),
                "affected_pct": apct,
                "value": None,
                "highlight_text": f"{feature.capitalize()} region shows {severity} activation.{how_much}",
                "_rank": sev_rank.get(severity, 0),
                "_metric": float(g.get("activation_score") or 0.0),
            }
        )

    for f in (rfs.get("facial_features") or []):
        if not isinstance(f, dict):
            continue
        feature = str(f.get("feature") or "").strip().lower()
        if not feature:
            continue
        severity = str(f.get("severity") or "unknown").strip().lower()
        if severity not in sev_rank:
            severity = "unknown"
        val = f.get("value")
        try:
            metric = float(val if val is not None else 0.0)
        except Exception:
            metric = 0.0
        out.append(
            {
                "type": "facial_feature",
                "feature": feature,
                "severity": severity,
                "color": sev_color.get(severity, sev_color["unknown"]),
                "activation_score": None,
                "contribution_pct": None,
                "value": val,
                "highlight_text": f"{feature.replace('_', ' ').capitalize()} is in {severity} risk band.",
                "_rank": sev_rank.get(severity, 0),
                "_metric": metric,
            }
        )

    out.sort(key=lambda x: (x.get("_rank", 0), x.get("_metric", 0.0)), reverse=True)
    trimmed = []
    for item in out[: max(1, int(top_n))]:
        item.pop("_rank", None)
        item.pop("_metric", None)
        trimmed.append(item)
    return trimmed

def _build_frontend_sections(
    explanation_text,
    risk_factor_sections,
    risk_factor_highlights,
    affected_region_details,
    future_health_issues,
    food_recommendations,
    exercise_recommendations,
    video_titles,
):
    """Build strict 4-box payload for frontend rendering."""
    rf = risk_factor_sections if isinstance(risk_factor_sections, dict) else {}
    grad = rf.get("gradcam") or []
    face = rf.get("facial_features") or []
    risk_scores = []
    for g in grad[:6]:
        if not isinstance(g, dict):
            continue
        risk_scores.append(
            {
                "feature": g.get("feature"),
                "score": g.get("activation_score"),
                "contribution_pct": g.get("contribution_pct"),
                "severity": g.get("severity"),
            }
        )
    for f in face[:4]:
        if not isinstance(f, dict):
            continue
        risk_scores.append(
            {
                "feature": f.get("feature"),
                "score": f.get("value"),
                "contribution_pct": None,
                "severity": f.get("severity"),
            }
        )

    return {
        "overall_explanation": {
            "title": "Overall Explanation",
            "text": str(explanation_text or "").strip(),
            "risk_scores": risk_scores,
        },
        "risk_factor_highlights": {
            "title": "Risk Factor Highlights",
            "items": list(risk_factor_highlights or []),
        },
        "affected_region_details": {
            "title": "Affected Region Details",
            "items": list(affected_region_details or []),
            "future_health_issues": list(future_health_issues or []),
        },
        "recommendations": {
            "title": "Food and Exercise Recommendations",
            "food_recommendations": _to_clean_list(food_recommendations, max_items=8),
            "exercise_recommendations": _to_clean_list(exercise_recommendations, max_items=8),
            "video_titles": _to_clean_list(video_titles, max_items=6),
        },
    }


def _compose_detailed_explanation(
    base_explanation,
    status,
    prob_adj,
    affected_region_details,
    future_health_issues,
    risk_factor_sections,
    food_rec,
    exercise_rec,
):
    pct = int(round(float(prob_adj) * 100)) if prob_adj is not None else 0
    lines = []
    base = str(base_explanation or "").strip()
    if base:
        lines.append(base)
    lines.append(f"Risk overview: {status} likelihood ({pct}%) based on Grad-CAM facial pattern analysis.")

    region_lines = []
    for item in (affected_region_details or [])[:6]:
        rg = str(item.get("region") or "general face")
        sev = str(item.get("severity") or "unknown")
        sc = item.get("activation_score")
        sc_txt = f"{sc}" if sc is not None else "N/A"
        ex = str(item.get("explanation") or "").strip()
        region_lines.append(f"- {rg}: score {sc_txt}, severity {sev}. {ex}")
    if region_lines:
        lines.append("Affected facial regions:")
        lines.extend(region_lines)

    rf = risk_factor_sections if isinstance(risk_factor_sections, dict) else {}
    grad = rf.get("gradcam") or []
    if grad:
        parts = []
        for g in grad[:4]:
            parts.append(
                f"{g.get('feature')}: {g.get('activation_score')} ({g.get('severity')}, {g.get('contribution_pct')}%)"
            )
        if parts:
            lines.append("Risk factors (Grad-CAM): " + "; ".join(parts) + ".")

    ff = rf.get("facial_features") or []
    if ff:
        fparts = [f"{f.get('feature')}={f.get('value')} ({f.get('severity')})" for f in ff[:4]]
        lines.append("Risk factors (facial features): " + "; ".join(fparts) + ".")

    issue_lines = []
    for i in (future_health_issues or [])[:6]:
        issue = str(i.get("issue") or "").strip()
        monitor = str(i.get("what_to_monitor") or "").strip()
        lev = str(i.get("risk_level") or "unknown")
        if issue:
            issue_lines.append(f"- {issue} [{lev}]: {monitor}")
    if issue_lines:
        lines.append("Future health issues to monitor:")
        lines.extend(issue_lines)

    if food_rec or exercise_rec:
        lines.append(f"Support plan: Food - {food_rec or 'balanced nutrition'}. Exercise - {exercise_rec or 'daily supervised activity'}.")

    lines.append("This is a screening result, not a final diagnosis. Confirm clinically with a pediatric specialist.")
    return _briefen_explanation(" ".join(lines), max_words=420)

def _build_frontend_explanation_sections(
    status,
    prob_adj,
    risk_factor_sections,
    affected_region_details,
    future_health_issues,
    food_recommendations,
    exercise_recommendations,
):
    """Build section-wise explanation text fields for frontend binding."""
    pct = int(round(float(prob_adj) * 100)) if prob_adj is not None else 0
    overview = f"Screening risk level is {status} with estimated probability {pct}%."

    rf = risk_factor_sections if isinstance(risk_factor_sections, dict) else {}
    grad = rf.get("gradcam") or []
    face = rf.get("facial_features") or []
    risk_parts = []
    for g in grad[:4]:
        if not isinstance(g, dict):
            continue
        risk_parts.append(
            f"{g.get('feature')}: score {g.get('activation_score') if g.get('activation_score') is not None else 'N/A'}, severity {g.get('severity') or 'unknown'}, contribution {g.get('contribution_pct') if g.get('contribution_pct') is not None else 'N/A'}%"
        )
    for f in face[:4]:
        if not isinstance(f, dict):
            continue
        risk_parts.append(
            f"{f.get('feature')}: value {f.get('value') if f.get('value') is not None else 'N/A'}, severity {f.get('severity') or 'unknown'}"
        )
    risk_text = "; ".join([str(x).strip() for x in risk_parts if str(x).strip()]) or "Risk-factor detail unavailable."

    region_parts = []
    for r in (affected_region_details or [])[:8]:
        if not isinstance(r, dict):
            continue
        region_parts.append(
            f"{r.get('region')}: score {r.get('activation_score') if r.get('activation_score') is not None else 'N/A'}, severity {r.get('severity') or 'unknown'}. {str(r.get('explanation') or '').strip()}"
        )
    region_text = " ".join([str(x).strip() for x in region_parts if str(x).strip()]) or "Affected-region detail unavailable."

    future_parts = []
    for i in (future_health_issues or [])[:8]:
        if isinstance(i, dict):
            future_parts.append(
                f"{i.get('issue') or 'Health issue'} [{i.get('risk_level') or 'unknown'}]{(': ' + str(i.get('what_to_monitor')).strip()) if str(i.get('what_to_monitor') or '').strip() else ''}"
            )
        elif isinstance(i, str) and i.strip():
            future_parts.append(i.strip())
    future_text = " ".join([str(x).strip() for x in future_parts if str(x).strip()]) or "Future-health detail unavailable."

    food_items = _to_clean_list(food_recommendations, max_items=6)
    exercise_items = _to_clean_list(exercise_recommendations, max_items=6)
    food_text = "; ".join(food_items) if food_items else "Food recommendation unavailable."
    exercise_text = "; ".join(exercise_items) if exercise_items else "Exercise recommendation unavailable."

    return {
        "overall_summary": overview,
        "risk_factor_explanation": _briefen_explanation(risk_text, max_words=160),
        "affected_region_explanation": _briefen_explanation(region_text, max_words=180),
        "future_health_explanation": _briefen_explanation(future_text, max_words=180),
        "food_explanation": _briefen_explanation(food_text, max_words=120),
        "exercise_explanation": _briefen_explanation(exercise_text, max_words=120),
    }


def _clean_domain_items(items, kind):
    deny = {"forehead", "eye", "eyes", "nose", "mouth", "chin", "ear", "ears", "general face"}
    cleaned = []
    for it in items or []:
        s = str(it).strip()
        if not s:
            continue
        if s.lower() in deny:
            continue
        if "recommendation" in s.lower() and len(s.split()) <= 3:
            continue
        cleaned.append(s)
    if cleaned:
        return cleaned
    return []


def _clean_video_titles(items):
    raw = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not raw:
        return []

    deny_tokens = {
        "meal", "snack", "diet", "food", "recipe", "nutrition", "cooking", "kitchen"
    }
    allow_tokens = {
        "exercise", "workout", "activity", "mobility", "stretch", "physio",
        "therapy", "movement", "strength", "balance", "walk", "cardio"
    }

    cleaned = []
    for title in raw:
        low = title.lower()
        if any(tok in low for tok in deny_tokens):
            continue
        if any(tok in low for tok in allow_tokens):
            cleaned.append(title)

    if cleaned:
        return cleaned[:3]
    return []


def _normalize_llm_payload(payload, fallback_regions=None):
    if not isinstance(payload, dict):
        return None

    def _pick(keys):
        for k in keys:
            v = payload.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    title = _pick(["title", "heading"]).strip()
    explanation = _pick(["explanation", "summary", "description"]).strip()
    original_explanation = explanation

    # Reject serialized-dict garbage in title/explanation.
    bad_markers = ["{'title':", "'title':", "'affected_regions':", '".join(', "'.join("]
    if any(m in title.lower() for m in bad_markers) or any(tok in title.lower() for tok in ['"title":', "{", "}", "[", "]"]):
        title = ""
    if any(m in explanation.lower() for m in bad_markers) or any(tok in explanation.lower() for tok in ['"title":', "'title':", '"food_recommendations":', "'food_recommendations':", '"exercise_recommendations":', "'exercise_recommendations':", '"video_titles":', "'video_titles':"]):
        # Try to recover useful prose from serialized/templated content instead of
        # discarding the explanation entirely.
        extracted = _extract_json_payload(explanation)
        if isinstance(extracted, dict):
            explanation = str(
                extracted.get("explanation")
                or extracted.get("summary")
                or extracted.get("description")
                or ""
            ).strip()
        if not explanation:
            # Recover prose by filtering structured/key-value lines instead of
            # deleting all bracketed blocks (which can wipe valid text).
            raw_exp = str(payload.get("explanation") or "")
            recovered_lines = []
            for ln in raw_exp.splitlines():
                s = ln.strip()
                if not s:
                    continue
                sl = s.lower()
                if sl in {"{", "}", "[", "]", ",", "```", "```json"}:
                    continue
                if any(tok in sl for tok in [
                    '"title":', "'title':",
                    '"affected_regions":', "'affected_regions':",
                    '"food_recommendations":', "'food_recommendations':",
                    '"exercise_recommendations":', "'exercise_recommendations':",
                    '"video_titles":', "'video_titles':",
                ]):
                    continue
                recovered_lines.append(s)

            recovered = " ".join(recovered_lines).strip()
            if not recovered:
                # Last-pass cleanup for single-line mixed payloads.
                recovered = re.sub(
                    r'(?i)\b(title|affected_regions|food_recommendations|exercise_recommendations|video_titles|summary|description)\b\s*:\s*',
                    " ",
                    raw_exp,
                )
                recovered = recovered.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
                recovered = recovered.replace('"', " ").replace("'", " ")
                recovered = re.sub(r"\s+", " ", recovered).strip(" ,;")
            explanation = recovered
    if explanation:
        brace_count = explanation.count("{") + explanation.count("}")
        if brace_count >= 4:
            # Keep llm-produced text by cleaning JSON-ish wrappers instead of dropping it.
            cleaned = re.sub(r"(?i)\b(title|affected_regions|food_recommendations|exercise_recommendations|video_titles|summary|description|risk_factor_sections|future_health_issues|affected_region_details)\b\s*:\s*", " ", explanation)
            cleaned = cleaned.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
            cleaned = cleaned.replace('"', " ").replace("'", " ").replace("`", " ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
            explanation = cleaned if len(cleaned.split()) >= 6 else ""

    if not explanation and original_explanation:
        cleaned = original_explanation.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
        cleaned = cleaned.replace('"', " ").replace("'", " ").replace("`", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        if len(cleaned.split()) >= 6:
            explanation = cleaned

    region_raw = payload.get("affected_regions")
    if region_raw is None:
        region_raw = payload.get("affected areas") or payload.get("affectedAreas")
    regions = _to_clean_list(region_raw, max_items=6)
    allowed_regions = {"forehead", "eyes", "nose", "mouth", "chin", "ears", "general face"}
    canonical = []
    for r in regions:
        rr = _canonical_region_name(r)
        if rr in allowed_regions and rr not in canonical:
            canonical.append(rr)
    if not canonical:
        canonical = list(fallback_regions or ["general face"])

    food = _to_clean_list(
        payload.get("food_recommendations")
        or payload.get("food recommendations")
        or payload.get("recommended_food")
        or payload.get("diet_recommendations"),
        max_items=4,
    )
    exercises = _to_clean_list(
        payload.get("exercise_recommendations")
        or payload.get("exercise recommendations")
        or payload.get("recommended_exercises"),
        max_items=4,
    )
    videos = _to_clean_list(
        payload.get("video_titles")
        or payload.get("videos")
        or payload.get("exercise_video_titles"),
        max_items=3,
    )

    # Optional structured sections produced by llm.
    region_details_raw = (
        payload.get("affected_region_details")
        or payload.get("affected_regions_detail")
        or payload.get("affected_region_info")
        or []
    )
    affected_region_details = []
    if isinstance(region_details_raw, list):
        for it in region_details_raw[:8]:
            if not isinstance(it, dict):
                continue
            rr = _canonical_region_name(it.get("region"))
            if not rr:
                continue
            sc = (
                it.get("activation_score")
                if it.get("activation_score") is not None
                else it.get("score")
            )
            try:
                sc_val = None if sc is None else round(float(sc), 3)
            except Exception:
                sc_val = None
            pct = (
                it.get("affected_pct")
                if it.get("affected_pct") is not None
                else it.get("affected_percentage")
            )
            try:
                pct_val = None if pct is None else round(float(pct), 1)
            except Exception:
                pct_val = None
            sev = str(it.get("severity") or "").strip().lower()
            if sev not in {"low", "moderate", "high", "unknown"}:
                sev = "unknown"
            ex = str(it.get("explanation") or "").strip()
            affected_region_details.append(
                {
                    "region": rr,
                    "activation_score": sc_val,
                    "affected_pct": pct_val,
                    "severity": sev,
                    "explanation": ex,
                }
            )

    future_issues_raw = payload.get("future_health_issues") or payload.get("health_issues") or []
    future_health_issues = []
    if isinstance(future_issues_raw, list):
        for it in future_issues_raw[:10]:
            if isinstance(it, dict):
                issue = str(it.get("issue") or "").strip()
                monitor = str(it.get("what_to_monitor") or it.get("monitor") or "").strip()
                lev = str(it.get("risk_level") or "").strip().lower()
                if lev not in {"low", "moderate", "high"}:
                    lev = "unknown"
                if issue:
                    future_health_issues.append(
                        {"issue": issue, "what_to_monitor": monitor, "risk_level": lev}
                    )
            elif isinstance(it, str) and it.strip():
                future_health_issues.append(
                    {"issue": it.strip(), "what_to_monitor": "", "risk_level": "unknown"}
                )

    rf_raw = payload.get("risk_factor_sections") or payload.get("risk_factors_section") or {}
    risk_factor_sections = {"gradcam": [], "facial_features": []}
    if isinstance(rf_raw, dict):
        grad_raw = rf_raw.get("gradcam") or []
        if isinstance(grad_raw, list):
            for g in grad_raw[:8]:
                if not isinstance(g, dict):
                    continue
                feat = str(g.get("feature") or "").strip().lower()
                if not feat:
                    continue
                try:
                    score = None if g.get("activation_score") is None else round(float(g.get("activation_score")), 3)
                except Exception:
                    score = None
                try:
                    contrib = None if g.get("contribution_pct") is None else round(float(g.get("contribution_pct")), 1)
                except Exception:
                    contrib = None
                sev = str(g.get("severity") or "").strip().lower() or "unknown"
                risk_factor_sections["gradcam"].append(
                    {
                        "feature": feat,
                        "activation_score": score,
                        "contribution_pct": contrib,
                        "severity": sev,
                    }
                )
        face_raw = rf_raw.get("facial_features") or []
        if isinstance(face_raw, list):
            for f in face_raw[:8]:
                if not isinstance(f, dict):
                    continue
                feat = str(f.get("feature") or "").strip().lower()
                if not feat:
                    continue
                try:
                    value = None if f.get("value") is None else round(float(f.get("value")), 2)
                except Exception:
                    value = None
                sev = str(f.get("severity") or "").strip().lower() or "unknown"
                risk_factor_sections["facial_features"].append(
                    {"feature": feat, "value": value, "severity": sev}
                )
    # Accept direct "risk_factors" list as a frontend-friendly alias for Grad-CAM factors.
    if not risk_factor_sections["gradcam"]:
        rf_list = payload.get("risk_factors") or []
        if isinstance(rf_list, list):
            for g in rf_list[:8]:
                if not isinstance(g, dict):
                    continue
                feat = str(g.get("feature") or g.get("region") or "").strip().lower()
                if not feat:
                    continue
                try:
                    score = None if g.get("activation_score") is None else round(float(g.get("activation_score")), 3)
                except Exception:
                    score = None
                try:
                    contrib = None if g.get("contribution_pct") is None else round(float(g.get("contribution_pct")), 1)
                except Exception:
                    contrib = None
                sev = str(g.get("severity") or "").strip().lower() or "unknown"
                risk_factor_sections["gradcam"].append(
                    {
                        "feature": feat,
                        "activation_score": score,
                        "contribution_pct": contrib,
                        "severity": sev,
                    }
                )

    if not title:
        title = "**AI Screening Summary**"
    elif not title.startswith("**"):
        title = f"**{title}**"

    if not explanation:
        return None
    if len(explanation) > 1200:
        explanation = explanation[:1200].rstrip() + "..."

    food = _clean_domain_items(food, "food")
    exercises = _clean_domain_items(exercises, "exercise")
    videos = _clean_video_titles(videos)

    return {
        "title": title,
        "explanation": explanation,
        "affected_regions": canonical[:5],
        "food_recommendations": food,
        "exercise_recommendations": exercises,
        "video_titles": videos,
        "affected_region_details": affected_region_details,
        "future_health_issues": future_health_issues,
        "risk_factor_sections": risk_factor_sections,
        "risk_factors": list(risk_factor_sections.get("gradcam") or []),
    }


def _salvage_llm_text_payload(raw_text, status, affected_regions, food_rec, exercise_rec):
    """Last-resort schema recovery using raw llm text (no local narrative fallback)."""
    txt = str(raw_text or "").strip()
    if not txt:
        return None
    payload = {
        "title": f"**{status} Risk Screening Summary**",
        "explanation": txt,
        "affected_regions": affected_regions or ["general face"],
        "food_recommendations": (
            food_rec.split(",") if isinstance(food_rec, str) and food_rec.strip() else [str(food_rec or "").strip()]
        ),
        "exercise_recommendations": (
            exercise_rec.split(",") if isinstance(exercise_rec, str) and exercise_rec.strip() else [str(exercise_rec or "").strip()]
        ),
        "video_titles": ["Gentle Child Mobility Routine", "Beginner Facial Muscle Exercise Session"],
    }
    normalized = _normalize_llm_payload(payload, fallback_regions=affected_regions)
    if normalized:
        return normalized

    # If strict normalization still fails, keep the raw text as usable explanation
    # so we do not unnecessarily drop to deterministic fallback.
    coerced = _coerce_text_payload(txt, fallback_regions=affected_regions)
    if coerced:
        normalized_coerced = _normalize_llm_payload(coerced, fallback_regions=affected_regions)
        if normalized_coerced:
            return normalized_coerced
        return coerced

    return {
        "title": f"**{status} Risk Screening Summary**",
        "explanation": _briefen_explanation(txt, max_words=260),
        "affected_regions": _normalize_affected_regions(affected_regions),
        "food_recommendations": _clean_domain_items(_to_clean_list(food_rec, max_items=4), "food"),
        "exercise_recommendations": _clean_domain_items(_to_clean_list(exercise_rec, max_items=4), "exercise"),
        "video_titles": ["Gentle Child Mobility Routine", "Beginner Facial Muscle Exercise Session"],
    }

def _ensure_payload_fields(
    payload,
    status,
    prob_adj,
    affected_regions,
    region_scores,
    features,
    gradcam_risk_factors,
    patient_info,
    default_food,
    default_exercise,
):
    """Force payload fields to stay aligned with this prediction result."""
    p = dict(payload or {})
    pred_regions = _normalize_affected_regions(affected_regions)
    p["affected_regions"] = pred_regions

    food = _clean_domain_items(
        _to_clean_list(p.get("food_recommendations") or default_food, max_items=6),
        "food",
    )
    ex = _clean_domain_items(
        _to_clean_list(p.get("exercise_recommendations") or default_exercise, max_items=6),
        "exercise",
    )
    # Keep deterministic prediction-based defaults present in final output.
    for item in _clean_domain_items(_to_clean_list(default_food, max_items=4), "food"):
        if item not in food:
            food.append(item)
    for item in _clean_domain_items(_to_clean_list(default_exercise, max_items=4), "exercise"):
        if item not in ex:
            ex.append(item)

    p["food_recommendations"] = food
    p["exercise_recommendations"] = ex
    if not p.get("video_titles"):
        p["video_titles"] = ["Gentle Child Mobility Routine", "Beginner Facial Muscle Exercise Session"]

    # Always derive risk factors from current prediction.
    p["risk_factor_sections"] = _build_risk_factor_sections(gradcam_risk_factors, features)
    p["risk_factors"] = list((p.get("risk_factor_sections") or {}).get("gradcam") or [])

    # Build region details from current prediction; keep LLM prose where available.
    llm_region_details = p.get("affected_region_details") if isinstance(p.get("affected_region_details"), list) else []
    llm_region_note = {}
    for it in llm_region_details:
        if not isinstance(it, dict):
            continue
        rr = _canonical_region_name(it.get("region"))
        ex_text = str(it.get("explanation") or "").strip()
        if rr and ex_text:
            llm_region_note[rr] = ex_text
    pred_region_details = _build_affected_region_details(pred_regions, region_scores=region_scores)
    for it in pred_region_details:
        rr = _canonical_region_name(it.get("region"))
        if rr in llm_region_note:
            it["explanation"] = llm_region_note[rr]
    p["affected_region_details"] = pred_region_details

    # Always derive future issues from current prediction + patient info.
    p["future_health_issues"] = _build_future_health_issues(
        status=status,
        affected_regions=pred_regions,
        features=features,
        prob_adj=prob_adj,
        patient_info=patient_info,
    )

    exp = str(p.get("explanation") or "").strip()
    if exp:
        p["explanation"] = _briefen_explanation(exp, max_words=700)
    return p

def _resolve_retry_model(preferred_model):
    """Force retries to stay on VL model; avoid tiny text-only model fallback."""
    env_retry = str(os.environ.get("llm_RETRY_MODEL", "") or "").strip()
    blocked = {"qwen2.5:0.5b"}
    if env_retry and env_retry not in blocked:
        return env_retry
    pref = str(preferred_model or "").strip()
    return pref or "qwen2.5vl:3b"

def detect_face_rect(img_path=None, image=None, target_size=None):
    if image is not None:
        img = image.copy()
    else:
        img = cv2.imread(img_path) if img_path else None
    if img is None:
        return None
    if target_size:
        img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        # Fallback to central face prior so highlighting remains face-focused
        # when Haar detection fails on varied facial morphology.
        h, w = gray.shape[:2]
        bw = int(w * 0.62)
        bh = int(h * 0.76)
        x = max(0, int((w - bw) / 2))
        y = max(0, int((h - bh) / 2) - int(h * 0.04))
        return (x, y, bw, bh)
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    return (x, y, w, h)



def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@timeout(seconds=int(os.environ.get("llm_TIMEOUT_SECONDS", "180")))
def call_openai(prompt: str, image_path: str, model: str = None, text_only: bool = False) -> str:
    """Send the screening prompt to OpenAI (vision if an image is present)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    if openai_client is None:
        raise RuntimeError("OpenAI client not initialized")

    chosen_model = model or OPENAI_MODEL
    content = [{"type": "text", "text": prompt}]
    if (not text_only) and image_path and os.path.exists(image_path):
        try:
            b64 = _encode_image_b64(image_path)
            content.append({"type": "input_image", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        except Exception as e:
            print(f"OpenAI: failed to read image for vision prompt: {e}")

    resp = openai_client.chat.completions.create(
        model=chosen_model,
        messages=[{"role": "user", "content": content}],
        temperature=OPENAI_TEMPERATURE,
        max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "900")),
    )
    return (resp.choices[0].message.content or "").strip()


# Unified LLM entry point (OpenAI)
def call_llm(prompt: str, image_path: str, model: str = None, text_only: bool = False) -> str:
    return call_openai(prompt=prompt, image_path=image_path, model=model, text_only=text_only)
print("🔄 Loading model...")
# ---------------- LOAD MODEL ----------------
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights(MODEL_PATH)
print("✅ Model loaded")
LAST_CONV_LAYER = _get_last_conv_layer(model)

# ---------------- AUTH ROUTES ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    profession = (data.get("profession") or "").strip()

    # Basic validation
    if not name or not email or not password or not profession:
        return jsonify({"error": "Missing name, email, password or profession"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    # simple email regex
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"error": "Invalid email address"}), 400

    hashed = generate_password_hash(password)
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (name, email, password, profession) VALUES (?, ?, ?, ?)",
            (name, email, hashed, profession)
        )
        conn.commit()
        conn.close()
        # Do NOT auto-issue a token on registration. Require explicit login.
        return jsonify({"success": True, "message": "User registered. Please log in."}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "User exists"}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    profession = (data.get("profession") or "").strip()

    if not email or not password or not profession:
        return jsonify({"error": "Missing email, password or profession"}), 400

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()
    if user and check_password_hash(user["password"], password):
        saved_profession = (user["profession"] or "").strip().lower()
        if saved_profession != profession.strip().lower():
            return jsonify({"error": "Profession does not match this email"}), 401
        token = create_access_token(identity=email)
        return jsonify({
            "token": token,
            "user": {"name": user["name"], "email": user["email"], "profession": user["profession"]}
        })
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/translate_tts", methods=["POST"])
def translate_tts():
    def _contains_script(text_value, lang_code):
        if not text_value:
            return False
        if lang_code == "hi":
            # Devanagari block
            return re.search(r"[\u0900-\u097F]", text_value) is not None
        if lang_code == "te":
            # Telugu block
            return re.search(r"[\u0C00-\u0C7F]", text_value) is not None
        return True

    def _translate_chunk(chunk_text, lang_code):
        target_name = "Hindi" if lang_code == "hi" else "Telugu"
        script_name = "Devanagari" if lang_code == "hi" else "Telugu script"
        base_prompt = f"""Translate the following English text into natural {target_name}.
Requirements:
- Output MUST be in {script_name} only.
- Do not keep English words unless they are medical terms.
- Keep meaning exact and simple for caregivers.
- Return only translated text.

Text:
{chunk_text}
"""
        translated_local = ""
        for _ in range(2):
            translated_local = (call_llm(
                prompt=base_prompt,
                image_path="",
                model=os.environ.get("llm_TRANSLATE_MODEL", "qwen2.5:0.5b"),
                text_only=True,
            ) or "").strip().strip('"').strip("'")
            if translated_local and _contains_script(translated_local, lang_code):
                break
        return translated_local

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    target_lang = (data.get("target_lang") or "en").strip().lower()

    if not text:
        return jsonify({"error": "Missing text"}), 400
    if target_lang not in ["en", "hi", "te"]:
        return jsonify({"error": "Unsupported target language"}), 400
    if target_lang == "en":
        return jsonify({"translated_text": text, "target_lang": "en"})

    try:
        # Chunk long page text so the model translates reliably.
        normalized = re.sub(r"\s+", " ", text).strip()
        chunks = [normalized[i:i + 700] for i in range(0, len(normalized), 700)]
        translated_parts = []
        for chunk in chunks:
            out = _translate_chunk(chunk, target_lang)
            if not out:
                continue
            translated_parts.append(out)

        translated = " ".join(translated_parts).strip()
        # Be permissive for reliability: if model returns usable translated text
        # (including romanized output), still speak it instead of failing.
        if not translated:
            return jsonify({"error": "Translation failed"}), 502
        return jsonify({"translated_text": translated, "target_lang": target_lang})
    except Exception as e:
        print(f"Translate TTS error: {e}")
        return jsonify({"error": "Translation failed"}), 502

@app.route("/assistant_chat", methods=["POST"])
def assistant_chat():
    def _plain_chat_text(value):
        text_value = str(value or "").strip()
        if not text_value:
            return ""
        # If model responded with JSON, extract common text keys.
        if text_value.startswith("{") and text_value.endswith("}"):
            parsed = None
            try:
                parsed = json.loads(text_value)
            except Exception:
                # Some runtimes stringify Python dicts (single quotes); recover safely.
                try:
                    parsed = ast.literal_eval(text_value)
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                original = str(value or "").strip()
                for k in ["message", "reply", "text", "response", "content", "explanation"]:
                    v = parsed.get(k)
                    if isinstance(v, str) and v.strip():
                        text_value = v.strip()
                        break
                # Some local servers return envelopes like:
                # {"Id":"...","type":"text/plain","body":"..."}
                if text_value == original and "body" in parsed:
                    body_v = parsed.get("body")
                    if isinstance(body_v, str):
                        text_value = body_v.strip()
                    elif isinstance(body_v, dict):
                        for k in ["text", "content", "message"]:
                            vv = body_v.get(k)
                            if isinstance(vv, str) and vv.strip():
                                text_value = vv.strip()
                                break
                # If this is an envelope and body is empty, force fallback path.
                if ("Id" in parsed and "type" in parsed and "body" in parsed) and not str(parsed.get("body") or "").strip():
                    text_value = ""
        # Strip extra wrapping quotes/backticks if present.
        text_value = text_value.strip().strip("`").strip()
        if (text_value.startswith('"') and text_value.endswith('"')) or (
            text_value.startswith("'") and text_value.endswith("'")
        ):
            text_value = text_value[1:-1].strip()
        return text_value

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    language = (data.get("language") or "en").strip().lower()
    history = data.get("history") or []

    if not message:
        return jsonify({"error": "Message is required"}), 400
    if language not in ["en", "hi", "te"]:
        language = "en"

    # Keep prompt bounded for reliability.
    if len(message) > 2000:
        message = message[:2000]

    lang_name = {"en": "English", "hi": "Hindi", "te": "Telugu"}[language]
    history_text = ""
    if isinstance(history, list):
        pairs = history[-8:]
        lines = []
        for item in pairs:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role in ["user", "assistant"] and content:
                lines.append(f"{role.capitalize()}: {content}")
        history_text = "\n".join(lines)

    prompt = f"""You are NMREC Assistant inside the NMREC app.
Identity and tone:
- Speak as an in-app guide for caregivers/parents.
- NEVER say lines like: "I am an AI", "I don't have emotions", "I don't have experiences".
- Do not discuss your limitations unless asked directly.
- Use simple, compassionate wording.

Scope:
- Explain screening steps and how to use this app.
- Help interpret app sections (patient info, upload, results, report download).
- Do not provide final medical diagnosis.
- For urgent/serious concerns, advise consulting a healthcare professional.

Output rules:
- Reply strictly in {lang_name}.
- Keep reply concise (4-8 sentences).
- Be directly useful; avoid generic filler.

Conversation so far:
{history_text if history_text else "N/A"}

User message:
{message}
"""
    try:
        reply = call_llm(
            prompt=prompt,
            image_path="",
            model=os.environ.get("llm_CHAT_MODEL", "qwen2.5:0.5b"),
            text_only=True,
        )
        reply = _plain_chat_text(reply)
        if not reply:
            if language == "hi":
                reply = (
                    "मैं आपकी स्क्रीनिंग प्रक्रिया में मदद कर सकता हूँ। पहले Patient Information भरें, "
                    "फिर साफ frontal face image अपलोड करें, उसके बाद परिणाम और रिपोर्ट देखें। "
                    "यदि परिणाम High या Moderate risk दिखाए, तो डॉक्टर से परामर्श करें।"
                )
            elif language == "te":
                reply = (
                    "స్క్రీనింగ్ ప్రక్రియలో నేను మీకు సహాయం చేస్తాను. ముందుగా Patient Information పూరించండి, "
                    "తర్వాత స్పష్టమైన frontal face image అప్‌లోడ్ చేయండి, ఆపై ఫలితాలు మరియు రిపోర్ట్ చూడండి. "
                    "ఫలితం High లేదా Moderate risk అయితే వైద్యుడిని సంప్రదించండి."
                )
            else:
                reply = (
                    "I can guide you through screening. First fill Patient Information, then upload a clear frontal face image, "
                    "then review results and download the report. If the result shows High or Moderate risk, please consult a doctor."
                )

        low = reply.lower()
        generic_markers = [
            "i'm an ai",
            "i am an ai",
            "i don't have personal experiences",
            "i do not have personal experiences",
            "i don't have emotions",
            "i do not have emotions",
        ]
        if any(m in low for m in generic_markers):
            if language == "hi":
                reply = (
                    "मैं आपकी स्क्रीनिंग प्रक्रिया में मदद कर सकता हूँ। पहले Patient Information भरें, "
                    "फिर साफ़ frontal face image अपलोड करें, उसके बाद परिणाम और रिपोर्ट देखें। "
                    "यदि परिणाम High या Moderate risk दिखाए, तो डॉक्टर से परामर्श करें।"
                )
            elif language == "te":
                reply = (
                    "స్క్రీనింగ్ ప్రక్రియలో నేను మీకు సహాయం చేస్తాను. ముందుగా Patient Information పూరించండి, "
                    "తర్వాత స్పష్టమైన frontal face image అప్‌లోడ్ చేయండి, ఆపై ఫలితాలు మరియు రిపోర్ట్ చూడండి. "
                    "ఫలితం High లేదా Moderate risk అయితే వైద్యుడిని సంప్రదించండి."
                )
            else:
                reply = (
                    "I can guide you through screening. First fill Patient Information, then upload a clear frontal face image, "
                    "then review results and download the report. If the result shows High or Moderate risk, please consult a doctor."
                )
        return jsonify({"reply": _plain_chat_text(reply)})
    except Exception as e:
        print(f"Assistant chat error: {e}")
        return jsonify({"error": "Assistant unavailable"}), 502


@app.route("/me", methods=["GET"])
@jwt_required()
def me():
    user_email = get_jwt_identity()
    conn = get_db()
    user = conn.execute("SELECT name, email, profession FROM users WHERE email = ?", (user_email,)).fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"name": user["name"], "email": user["email"], "profession": user["profession"]})

# ---------------- PATIENT INFO ----------------
def _to_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return None

@app.route("/patient_info", methods=["GET"])
@jwt_required()
def get_patient_info():
    user_email = get_jwt_identity()
    if not user_email:
        return jsonify({"jwt_error": "Missing or invalid token"}), 401
    conn = get_db()
    row = conn.execute(
        """SELECT patient_name, patient_age, mother_age, father_age,
                  mother_health, father_health, parent_relation, living_area,
                  previous_pregnancies, pregnancy_complications, family_history,
                  family_genetic, family_genetic_details,
                  relation, income, spouse_relation, annual_income, extra_info, notes
           FROM patient_info WHERE user_email = ?""",
        (user_email,)
    ).fetchone()
    conn.close()
    if not row:
        return jsonify(None)
    return jsonify(dict(row))

@app.route("/patient_info", methods=["POST"])
@jwt_required()
def save_patient_info():
    user_email = get_jwt_identity()
    if not user_email:
        return jsonify({"jwt_error": "Missing or invalid token"}), 401
    data = request.get_json(silent=True) or {}
    patient_name = (data.get("patient_name") or "").strip()
    patient_age = _to_int_or_none(data.get("patient_age"))
    mother_age = _to_int_or_none(data.get("mother_age"))
    father_age = _to_int_or_none(data.get("father_age"))
    parent_relation = (data.get("parent_relation") or data.get("relation") or "").strip()
    living_area = (data.get("living_area") or data.get("income") or "").strip()
    previous_pregnancies = _to_int_or_none(data.get("previous_pregnancies"))

    missing_fields = []
    if not patient_name:
        missing_fields.append("patient_name")
    if patient_age is None:
        missing_fields.append("patient_age")
    if mother_age is None:
        missing_fields.append("mother_age")
    if father_age is None:
        missing_fields.append("father_age")
    if not parent_relation:
        missing_fields.append("parent_relation")
    if not living_area:
        missing_fields.append("living_area")
    if previous_pregnancies is None:
        missing_fields.append("previous_pregnancies")

    if missing_fields:
        return jsonify({
            "error": "Please fill all required patient fields",
            "missing_fields": missing_fields
        }), 400

    mother_health = (data.get("mother_health") or "").strip()
    father_health = (data.get("father_health") or "").strip()
    pregnancy_complications = (data.get("pregnancy_complications") or "").strip()
    family_history = (data.get("family_history") or "").strip()
    family_genetic = (data.get("family_genetic") or "").strip()
    family_genetic_details = (data.get("family_genetic_details") or "").strip()
    notes = (data.get("notes") or "").strip()
    legacy_extra = (data.get("extra_info") or "").strip()
    extra_info = legacy_extra or " | ".join(
        [x for x in [mother_health, father_health, pregnancy_complications, family_history] if x]
    )

    conn = get_db()
    conn.execute(
        """INSERT INTO patient_info (
               user_email, patient_name, patient_age, mother_age, father_age,
               mother_health, father_health, parent_relation, living_area, previous_pregnancies,
               pregnancy_complications, family_history, family_genetic, family_genetic_details,
               relation, income, spouse_relation, annual_income, extra_info, notes, created_at, updated_at
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
           ON CONFLICT(user_email) DO UPDATE SET
               patient_name = excluded.patient_name,
               patient_age = excluded.patient_age,
               mother_age = excluded.mother_age,
               father_age = excluded.father_age,
               mother_health = excluded.mother_health,
               father_health = excluded.father_health,
               parent_relation = excluded.parent_relation,
               living_area = excluded.living_area,
               previous_pregnancies = excluded.previous_pregnancies,
               pregnancy_complications = excluded.pregnancy_complications,
               family_history = excluded.family_history,
               family_genetic = excluded.family_genetic,
               family_genetic_details = excluded.family_genetic_details,
               relation = excluded.relation,
               income = excluded.income,
               spouse_relation = excluded.spouse_relation,
               annual_income = excluded.annual_income,
               extra_info = excluded.extra_info,
               notes = excluded.notes,
               updated_at = datetime('now')""",
        (
            user_email,
            patient_name,
            patient_age,
            mother_age,
            father_age,
            mother_health or None,
            father_health or None,
            parent_relation,
            living_area,
            previous_pregnancies,
            pregnancy_complications or None,
            family_history or None,
            family_genetic or None,
            family_genetic_details or None,
            parent_relation,
            living_area,
            family_genetic or None,
            str(previous_pregnancies) if previous_pregnancies is not None else None,
            extra_info or None,
            notes or None,
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/patient_info", methods=["PUT"])
@jwt_required()
def update_patient_info():
    user_email = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    patient_name = (data.get("patient_name") or "").strip()
    patient_age = _to_int_or_none(data.get("patient_age"))
    mother_age = _to_int_or_none(data.get("mother_age"))
    father_age = _to_int_or_none(data.get("father_age"))
    parent_relation = (data.get("parent_relation") or data.get("relation") or "").strip()
    living_area = (data.get("living_area") or data.get("income") or "").strip()
    previous_pregnancies = _to_int_or_none(data.get("previous_pregnancies"))

    missing_fields = []
    if not patient_name:
        missing_fields.append("patient_name")
    if patient_age is None:
        missing_fields.append("patient_age")
    if mother_age is None:
        missing_fields.append("mother_age")
    if father_age is None:
        missing_fields.append("father_age")
    if not parent_relation:
        missing_fields.append("parent_relation")
    if not living_area:
        missing_fields.append("living_area")
    if previous_pregnancies is None:
        missing_fields.append("previous_pregnancies")
    if missing_fields:
        return jsonify({
            "error": "Please fill all required patient fields",
            "missing_fields": missing_fields
        }), 400

    mother_health = (data.get("mother_health") or "").strip()
    father_health = (data.get("father_health") or "").strip()
    pregnancy_complications = (data.get("pregnancy_complications") or "").strip()
    family_history = (data.get("family_history") or "").strip()
    family_genetic = (data.get("family_genetic") or "").strip()
    family_genetic_details = (data.get("family_genetic_details") or "").strip()
    notes = (data.get("notes") or "").strip()
    legacy_extra = (data.get("extra_info") or "").strip()
    extra_info = legacy_extra or " | ".join(
        [x for x in [mother_health, father_health, pregnancy_complications, family_history] if x]
    )

    conn = get_db()
    conn.execute(
        """UPDATE patient_info SET
           patient_name = ?, patient_age = ?, mother_age = ?, father_age = ?,
           mother_health = ?, father_health = ?, parent_relation = ?, living_area = ?, previous_pregnancies = ?,
           pregnancy_complications = ?, family_history = ?, family_genetic = ?, family_genetic_details = ?,
           relation = ?, income = ?, spouse_relation = ?, annual_income = ?, extra_info = ?, notes = ?,
           updated_at = datetime('now')
           WHERE user_email = ?""",
        (
            patient_name,
            patient_age,
            mother_age,
            father_age,
            mother_health or None,
            father_health or None,
            parent_relation,
            living_area,
            previous_pregnancies,
            pregnancy_complications or None,
            family_history or None,
            family_genetic or None,
            family_genetic_details or None,
            parent_relation,
            living_area,
            family_genetic or None,
            str(previous_pregnancies) if previous_pregnancies is not None else None,
            extra_info or None,
            notes or None,
            user_email,
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
@jwt_required()
def predict_route():
    user_email = get_jwt_identity()
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    original_filename = secure_filename(file.filename or "")
    if not original_filename:
        return jsonify({"error": "Invalid file name"}), 400
    name_root, ext = os.path.splitext(original_filename)
    filename = f"{name_root}_{uuid4().hex}{ext.lower()}"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    try:
        img = image.load_img(upload_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        raw_output = float(model.predict(img_array, verbose=0)[0][0])
        prob = float(np.clip(1.0 - raw_output, 0, 1))

        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        heatmap_name = f"heatmap_{filename}"
        heatmap_path = os.path.join(OUTPUT_FOLDER, heatmap_name)

        face_rect = detect_face_rect(img_path=upload_path, target_size=(heatmap.shape[1], heatmap.shape[0]))
        save_heatmap(upload_path, heatmap, heatmap_path, face_rect=face_rect)

        face_detected_regions = analyze_heatmap_regions_by_face(upload_path, heatmap, threshold=100)
        region_analysis = analyze_heatmap_regions(heatmap, face_rect=face_rect)
        score_regions = region_analysis.get("affected_regions", []) if isinstance(region_analysis, dict) else []
        region_scores = region_analysis.get("region_scores", {}) if isinstance(region_analysis, dict) else {}
        affected_regions = _merge_affected_regions(face_detected_regions, score_regions, region_scores)
        prob_adj = calibrate_probability(prob, heatmap, face_rect=face_rect)

        confidence = float(np.clip(max(prob_adj, 1.0 - prob_adj), 0.0, 1.0))
        features, theory = analyze_facial_features(heatmap, prob, face_rect=face_rect)
        gradcam_risk_factors = _build_gradcam_risk_factors(region_scores)
        risk_matrix = _build_region_risk_matrix(region_scores, affected_regions)

        status = "High" if prob_adj > 0.7 else ("Moderate" if prob_adj > 0.4 else "Low")
        risk_level = "high" if prob_adj >= 0.7 else ("moderate" if prob_adj >= 0.4 else "low")
        food_rec, exercise_rec = _region_recommendations(affected_regions, risk_level)

        explanation_text = ""
        parsed_explanation = None
        patient_context = ""
        llm_title = ""
        llm_video_titles = []
        llm_affected_regions = list(affected_regions or ["general face"])
        llm_food_list = _to_clean_list(food_rec, max_items=4)
        llm_exercise_list = _to_clean_list(exercise_rec, max_items=4)
        raw_llm_output = ""
        llm_model_used = ""
        llm_output_source = "local_fallback"

        print(f"-> Attempting LLM explanation for user {user_email[:8]}...")

        llm_available = openai_client is not None
        if not llm_available:
            print("-> OPENAI_API_KEY not configured; will use local fallback explanation")
        pinfo = None
        try:
            conn = get_db()
            pinfo = conn.execute(
                """
                SELECT patient_name, patient_age, mother_age, father_age, mother_health, father_health,
                       parent_relation, living_area, previous_pregnancies, pregnancy_complications,
                       family_history, family_genetic, family_genetic_details,
                       relation, income, spouse_relation, annual_income, extra_info, notes
                FROM patient_info
                WHERE user_email = ?
                """,
                (user_email,),
            ).fetchone()
            conn.close()

            if pinfo:
                patient_context = f"""
Patient Background Information:
- Patient Name: {pinfo['patient_name']}
- Child Age: {pinfo['patient_age']}
- Mother Age: {pinfo['mother_age']}
- Father Age: {pinfo['father_age']}
- Parent Relation: {pinfo['parent_relation'] or pinfo['relation']}
- Living Area: {pinfo['living_area'] or pinfo['income']}
- Previous Pregnancies: {pinfo['previous_pregnancies']}
- Mother Health: {pinfo['mother_health']}
- Father Health: {pinfo['father_health']}
- Pregnancy Complications: {pinfo['pregnancy_complications']}
- Family History: {pinfo['family_history']}
- Known Family Genetic Disorder: {pinfo['family_genetic']}
- Family Genetic Details: {pinfo['family_genetic_details']}
- Additional Info: {pinfo['extra_info']}
- Notes: {pinfo['notes']}
"""
                patient_context = _limit_text(
                    patient_context,
                    max_chars=int(os.environ.get("llm_PATIENT_CONTEXT_MAX_CHARS", "900")),
                )

            explain_image = heatmap_path if os.path.exists(heatmap_path) else upload_path
            resized_path = None
            try:
                img_cv = cv2.imread(explain_image)
                if img_cv is not None:
                    h, w = img_cv.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > 640:
                        scale = 640.0 / float(max_dim)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        small = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        small = img_cv
                    resized_name = f"resized_{os.path.basename(explain_image)}"
                    resized_path = os.path.join(UPLOAD_FOLDER, resized_name)
                    cv2.imwrite(resized_path, small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    print(f"-> Using resized heatmap for llm: {resized_path}")
                else:
                    resized_path = explain_image

                json_prompt = f"""You are a compassionate pediatric screening assistant. Use the ATTACHED IMAGE (a grad-CAM heatmap) AND the patient background information below.

Instructions:
- Provide a short title (one-line) that summarizes the result and HIGHLIGHT it by surrounding with double asterisks, e.g. **Title Here**.
- List the most affected facial regions (choose from: forehead, eyes, nose, mouth, chin, ears, general face) in an array.
- Provide a clear explanation (4-6 short sentences) in plain, reassuring language.
- In the explanation, explicitly state which facial area is affected and how much it is affected (use region score/intensity wording).
- In the explanation, include a brief line for each affected region explaining what that region focus means.
- Explicitly include the current risk level and probability percentage in the explanation.
- Explicitly include a short "feature risk" interpretation based on facial symmetry, eye spacing, nasal bridge, and ear position.
- Explicitly include likely "health challenges to monitor" based on risk level (for example: speech delay, low muscle tone, feeding difficulty, slower milestones).
- In the explanation, include clear mini-sections in plain text: "Risk factors", "Affected region details", and "Future health issues to monitor".
- Provide practical food recommendations as an array of short strings.
- Provide exercise suggestions as an array of short strings and also suggest 2-3 YouTube-style video titles (strings).
- Include one short sentence on "how to overcome/support" using the recommended food and exercises.
- Write in full, clear sentences (no shorthand, no abbreviations, no clipped phrases).
- Build `affected_region_details` with objects: `region`, `activation_score` (0 to 1), `affected_pct` (0 to 100), `severity`, `explanation`.
- Build `future_health_issues` with objects: `issue`, `what_to_monitor`, `risk_level`.
- Build `risk_factor_sections` with:
  - `gradcam`: objects `{{feature, activation_score, contribution_pct, severity}}`
  - `facial_features`: objects `{{feature, value, severity}}`
- Also provide `risk_factors` as the same list as `risk_factor_sections.gradcam`.

Output requirements:
- ONLY return valid JSON (no extra commentary) with keys: `title`, `explanation`, `affected_regions`, `food_recommendations`, `exercise_recommendations`, `video_titles`, `affected_region_details`, `future_health_issues`, `risk_factor_sections`, `risk_factors`.
- Keep text short; avoid technical jargon.
- Keep explanation around 110-180 words.
- Do not skip any detected affected region in the explanation.
- Tie the explanation to the current case context, not a generic template.
- Ensure the explanation uses patient background context when provided.
- Ensure explanation content is grounded in both Grad-CAM region scores and patient background context.

Model Screening Context:
- Probability of Down syndrome traits: {round(prob_adj,3)}
- Risk level: {status}
- Detected affected regions (from analyzer): {', '.join(affected_regions) if affected_regions else 'none'}
- Region activation scores: {region_scores if region_scores else {}}

Patient Background:\n{patient_context}

Model, produce the JSON now."""

                preferred_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
                retry_model = _resolve_retry_model(preferred_model)
                llm_model_used = preferred_model
                if llm_available:
                    try:
                        explanation_text = call_llm(
                            prompt=json_prompt,
                            image_path=(resized_path or explain_image),
                            model=preferred_model,
                        )
                        raw_llm_output = str(explanation_text or "")
                    except Exception as llm_err:
                        print(f"LLM call failed: {llm_err}")
                        explanation_text = ""
                        raw_llm_output = str(llm_err)
                        llm_output_source = "local_fallback_no_llm"
                else:
                    explanation_text = ""
                    raw_llm_output = ""
                    llm_output_source = "local_fallback_no_llm"
            finally:
                try:
                    if resized_path and resized_path != explain_image and os.path.exists(resized_path):
                        os.remove(resized_path)
                except Exception:
                    pass

            if explanation_text:
                parsed_explanation = _extract_json_payload(explanation_text)
                parsed_explanation = _normalize_llm_payload(
                    parsed_explanation, fallback_regions=affected_regions
                )
                if parsed_explanation:
                    print("-> llm SUCCESS (parsed JSON)")
                    llm_output_source = "llm_primary_json"
                else:
                    parsed_explanation = _coerce_text_payload(
                        explanation_text, fallback_regions=affected_regions
                    )
                    parsed_explanation = _normalize_llm_payload(
                        parsed_explanation, fallback_regions=affected_regions
                    )
                    if parsed_explanation:
                        print("-> llm SUCCESS (coerced non-JSON output)")
                        llm_output_source = "llm_primary_coerced"
                    else:
                        print(f"-> llm returned non-JSON output (length: {len(explanation_text)}). Falling back to text explanation")
                        parsed_explanation = _salvage_llm_text_payload(
                            explanation_text,
                            status=status,
                            affected_regions=affected_regions,
                            food_rec=food_rec,
                            exercise_rec=exercise_rec,
                        )
                        if parsed_explanation:
                            print("-> llm SUCCESS (salvaged raw non-JSON output)")
                            llm_output_source = "llm_primary_salvaged"
            else:
                raise ValueError("Empty response from llm")

            # If the vision output is generic or not grounded in this case context,
            # run a second text-only pass using structured model context.
            enable_second_pass = str(os.environ.get("llm_ENABLE_SECOND_PASS", "0")).strip().lower() in {"1", "true", "yes"}
            has_structured_now = bool(parsed_explanation) and _has_structured_sections(parsed_explanation)
            needs_context_retry = bool(parsed_explanation) and (
                (not has_structured_now)
                and (
                    _is_weak_llm_payload(parsed_explanation)
                    or not _payload_uses_context(
                        parsed_explanation, status=status, prob_adj=prob_adj, affected_regions=affected_regions
                    )
                )
            )
            if enable_second_pass and needs_context_retry and llm_available:
                print("-> LLM output looked generic/context-light; retrying with context-only prompt")
                text_prompt = f"""You are a pediatric screening assistant.

Use ONLY the provided context and produce valid JSON with keys:
title, explanation, affected_regions, food_recommendations, exercise_recommendations, video_titles, affected_region_details, future_health_issues, risk_factor_sections, risk_factors

Rules:
- Do not describe image colors or stripes.
- Explain likely affected facial areas (eyes, nose, mouth, chin, forehead, ears) using region scores.
- Mention this is screening, not diagnosis.
- Keep explanation practical, compassionate, and slightly detailed.
- Clearly state which area is affected and how much (intensity/score) in explanation.
- Include the exact risk level text from context and include the probability percentage from context.
- Include one short "feature risk" interpretation sentence using facial symmetry, eye spacing, nasal bridge, and ear position.
- Include one short "health challenges to monitor" sentence based on risk level (speech, feeding, muscle tone, milestones).
- In explanation, explicitly include short sections named: "Risk factors", "Affected region details", and "Future health issues to monitor".
- Include one short "how to overcome/support" sentence using food and exercise recommendations.
- Suggest 2-3 exercise video title ideas.
- Write in complete sentences with normal wording (no shorthand/abbreviations).
- Include a short brief note for each affected region.
- `affected_region_details`: objects with `region`, `activation_score`, `affected_pct`, `severity`, `explanation`.
- `future_health_issues`: objects with `issue`, `what_to_monitor`, `risk_level`.
- `risk_factor_sections`: include `gradcam` and `facial_features`.
- `risk_factors` must repeat the `gradcam` list for frontend compatibility.

Context:
- Probability: {round(prob_adj,3)}
- Risk level: {status}
- Detected regions: {', '.join(affected_regions) if affected_regions else 'none'}
- Region scores: {region_scores if region_scores else {}}
- Patient info: {patient_context if patient_context else 'N/A'}
"""
                try:
                    retry_text = call_llm(
                        prompt=text_prompt,
                        image_path=(heatmap_path if os.path.exists(heatmap_path) else upload_path),
                        model=retry_model,
                        text_only=False,
                    )
                    if retry_text and str(retry_text).strip():
                        raw_llm_output = str(retry_text)
                        llm_model_used = retry_model
                except Exception as llm_err:
                    print(f"LLM retry failed: {llm_err}")
                    retry_text = ""
                    raw_llm_output = str(llm_err)
                retry_payload = _extract_json_payload(retry_text) or _coerce_text_payload(
                    retry_text, fallback_regions=affected_regions
                )
                retry_payload = _normalize_llm_payload(
                    retry_payload, fallback_regions=affected_regions
                )
                if not retry_payload and retry_text and str(retry_text).strip():
                    retry_payload = _salvage_llm_text_payload(
                        retry_text,
                        status=status,
                        affected_regions=affected_regions,
                        food_rec=food_rec,
                        exercise_rec=exercise_rec,
                    )
                if retry_payload:
                    # Accept parseable retry output; weak parts are enriched later with
                    # risk-grounded local context instead of dropping the whole result.
                    parsed_explanation = retry_payload
                    if _is_weak_llm_payload(retry_payload):
                        print("-> Context-only retry parsed but weak; will enrich with local context")
                        llm_output_source = "llm_retry_weak"
                    else:
                        print("-> llm SUCCESS (context-only retry)")
                        llm_output_source = "llm_retry"
                else:
                    print("-> Context-only retry unparseable and empty; unable to recover")

            # Final llm-only salvage: if parsing failed but first-pass text exists, keep that text.
            if not parsed_explanation and explanation_text and str(explanation_text).strip():
                parsed_explanation = _salvage_llm_text_payload(
                    explanation_text,
                    status=status,
                    affected_regions=affected_regions,
                    food_rec=food_rec,
                    exercise_rec=exercise_rec,
                )
                if parsed_explanation:
                    print("-> Recovered usable payload from first-pass llm text")
                    llm_output_source = "llm_salvaged"

            if parsed_explanation and not _payload_uses_context(
                parsed_explanation, status=status, prob_adj=prob_adj, affected_regions=affected_regions
            ):
                # Preserve llm structure but enrich explanation with case-grounded context.
                print("-> llm output not fully grounded; enriching explanation with case context")
                try:
                    enriched_text = _build_full_explanation(
                        base_explanation=parsed_explanation.get("explanation"),
                        status=status,
                        prob_adj=prob_adj,
                        affected_regions=parsed_explanation.get("affected_regions") or affected_regions,
                        region_scores=region_scores,
                        features=features,
                        patient_context=patient_context,
                        food_recommendations=parsed_explanation.get("food_recommendations") or food_rec,
                        exercise_recommendations=parsed_explanation.get("exercise_recommendations") or exercise_rec,
                        video_titles=parsed_explanation.get("video_titles"),
                    )
                    if enriched_text and str(enriched_text).strip():
                        parsed_explanation["explanation"] = str(enriched_text).strip()
                        llm_output_source = "llm_enriched_context"
                except Exception as enrich_err:
                    print(f"-> Context enrichment skipped due to error: {enrich_err}")

        except Exception as e:
            print(f"llm call failed: {e}")
            explanation_text = ""

        if not parsed_explanation:
            # Graceful local fallback to avoid hard failure when LLM is unavailable.
            print("-> LLM output unavailable; generating local fallback explanation")
            llm_output_source = "local_fallback_no_llm"
            fallback_payload = {
                "title": "Screening Summary",
                "explanation": _compose_detailed_explanation(
                    base_explanation="Screening completed. See detected risk factors and recommendations below.",
                    status=status,
                    prob_adj=prob_adj,
                    affected_region_details=[],
                    future_health_issues=[],
                    risk_factor_sections={},
                    food_rec=food_rec,
                    exercise_rec=exercise_rec,
                ),
                "affected_regions": affected_regions or ["face"],
                "food_recommendations": list(_to_clean_list(food_rec)),
                "exercise_recommendations": list(_to_clean_list(exercise_rec)),
                "video_titles": [],
                "affected_region_details": [],
                "future_health_issues": [],
                "risk_factor_sections": {},
                "risk_factors": [],
            }
            parsed_explanation = _ensure_payload_fields(
                payload=fallback_payload,
                status=status,
                prob_adj=prob_adj,
                affected_regions=affected_regions,
                region_scores=region_scores,
                features=features,
                gradcam_risk_factors=gradcam_risk_factors,
                default_food=food_rec,
                default_exercise=exercise_rec,
                patient_info=dict(pinfo) if pinfo else {},
            )

        parsed_explanation = _ensure_payload_fields(
            payload=parsed_explanation,
            status=status,
            prob_adj=prob_adj,
            affected_regions=affected_regions,
            region_scores=region_scores,
            features=features,
            gradcam_risk_factors=gradcam_risk_factors,
            patient_info=(dict(pinfo) if pinfo else {}),
            default_food=food_rec,
            default_exercise=exercise_rec,
        )

        llm_title = str(parsed_explanation.get("title") or "").strip() or "**llm Screening Summary**"
        raw_explanation = str(parsed_explanation.get("explanation") or "").strip()
        food_list = _clean_domain_items(parsed_explanation.get("food_recommendations") or [], "food")
        ex_list = _clean_domain_items(parsed_explanation.get("exercise_recommendations") or [], "exercise")
        llm_affected_regions = list(parsed_explanation.get("affected_regions") or affected_regions or ["general face"])
        llm_food_list = food_list
        llm_exercise_list = ex_list
        food_rec = ", ".join(food_list)
        exercise_rec = ", ".join(ex_list)
        llm_video_titles = _clean_video_titles(parsed_explanation.get("video_titles") or [])
        # Keep grounded explanation as primary output for frontend visibility.
        explanation_text = raw_explanation
        if not llm_output_source.startswith("llm") and llm_output_source != "gradcam_context_fallback":
            llm_output_source = "llm_primary"

        explanation_text = _briefen_explanation(explanation_text, max_words=700)
        affected_region_details = list(parsed_explanation.get("affected_region_details") or [])
        future_health_issues = list(parsed_explanation.get("future_health_issues") or [])
        risk_factor_sections = parsed_explanation.get("risk_factor_sections") or {"gradcam": [], "facial_features": []}
        risk_factor_highlights = _build_risk_factor_highlights(
            risk_factor_sections, top_n=4, affected_region_details=affected_region_details
        )

        # Enforce llm-generated structured sections (no handwritten/template fallback sections).
        has_any_rf_section = isinstance(risk_factor_sections, dict) and (
            (risk_factor_sections.get("gradcam") or []) or (risk_factor_sections.get("facial_features") or [])
        )
        missing_all_structured_sections = (not affected_region_details) and (not future_health_issues) and (not has_any_rf_section)
        if missing_all_structured_sections:
            print("-> Missing structured sections from llm; requesting strict structured rewrite")
            structured_prompt = f"""Rewrite the screening output into strict JSON only.

Required keys:
- title (string)
- explanation (string, detailed)
- affected_regions (array of strings)
- food_recommendations (array of strings)
- exercise_recommendations (array of strings)
- video_titles (array of strings)
- affected_region_details (array of objects: region, activation_score, affected_pct, severity, explanation)
- future_health_issues (array of objects: issue, what_to_monitor, risk_level)
- risk_factor_sections (object with keys: gradcam, facial_features)
  - gradcam: array of objects {{feature, activation_score, contribution_pct, severity}}
  - facial_features: array of objects {{feature, value, severity}}
- risk_factors (array; same content as risk_factor_sections.gradcam)

Use this context:
- risk_level: {risk_level}
- status: {status}
- probability: {round(prob_adj,3)}
- affected_regions: {', '.join(llm_affected_regions or affected_regions)}
- region_scores: {region_scores if region_scores else {}}
- facial_features: {features}
- patient_background: {patient_context if patient_context else 'N/A'}
- existing_explanation: {explanation_text}

Output must be JSON only."""
            structured_text = call_llm(
                prompt=structured_prompt,
                image_path=(heatmap_path if os.path.exists(heatmap_path) else upload_path),
                model=_resolve_retry_model(llm_model_used or os.environ.get("llm_MODEL", "qwen2.5vl:3b")),
                text_only=False,
            )
            if structured_text and str(structured_text).strip():
                raw_llm_output = str(structured_text)
                llm_model_used = _resolve_retry_model(llm_model_used or os.environ.get("llm_MODEL", "qwen2.5vl:3b"))
            structured_payload = _extract_json_payload(structured_text) or _coerce_text_payload(
                structured_text, fallback_regions=affected_regions
            )
            structured_payload = _normalize_llm_payload(
                structured_payload, fallback_regions=affected_regions
            )
            if not structured_payload:
                return jsonify(
                    {
                        "success": False,
                        "error": "llm failed to generate structured sections. Please retry.",
                        "llm_output_source": "llm_structured_missing",
                        "raw_llm_output": raw_llm_output,
                    }
                ), 502

            parsed_explanation = structured_payload
            llm_output_source = "llm_structured_retry"
            explanation_text = _briefen_explanation(str(parsed_explanation.get("explanation") or "").strip(), max_words=700)
            llm_title = str(parsed_explanation.get("title") or llm_title).strip() or llm_title
            llm_affected_regions = list(parsed_explanation.get("affected_regions") or llm_affected_regions or affected_regions)
            food_list = _clean_domain_items(parsed_explanation.get("food_recommendations") or [], "food")
            ex_list = _clean_domain_items(parsed_explanation.get("exercise_recommendations") or [], "exercise")
            llm_food_list = food_list
            llm_exercise_list = ex_list
            food_rec = ", ".join(food_list)
            exercise_rec = ", ".join(ex_list)
            llm_video_titles = _clean_video_titles(parsed_explanation.get("video_titles") or [])
            affected_region_details = list(parsed_explanation.get("affected_region_details") or [])
            future_health_issues = list(parsed_explanation.get("future_health_issues") or [])
            risk_factor_sections = parsed_explanation.get("risk_factor_sections") or {"gradcam": [], "facial_features": []}
            risk_factor_highlights = _build_risk_factor_highlights(
                risk_factor_sections, top_n=4, affected_region_details=affected_region_details
            )

        # Force final explanation to explicitly include risk factors, affected-region details,
        # and future-health issues for frontend display.
        explanation_text = _compose_detailed_explanation(
            base_explanation=explanation_text,
            status=status,
            prob_adj=prob_adj,
            affected_region_details=affected_region_details,
            future_health_issues=future_health_issues,
            risk_factor_sections=risk_factor_sections,
            food_rec=food_rec,
            exercise_rec=exercise_rec,
        )
        explanation_sections = _build_frontend_explanation_sections(
            status=status,
            prob_adj=prob_adj,
            risk_factor_sections=risk_factor_sections,
            affected_region_details=affected_region_details,
            future_health_issues=future_health_issues,
            food_recommendations=llm_food_list,
            exercise_recommendations=llm_exercise_list,
        )
        frontend_sections = _build_frontend_sections(
            explanation_text=explanation_text,
            risk_factor_sections=risk_factor_sections,
            risk_factor_highlights=risk_factor_highlights,
            affected_region_details=affected_region_details,
            future_health_issues=future_health_issues,
            food_recommendations=llm_food_list,
            exercise_recommendations=llm_exercise_list,
            video_titles=llm_video_titles,
        )

        screening_patient_name = filename
        screening_patient_age = None
        screening_mother_age = None
        screening_father_age = None
        screening_parent_relation = None
        screening_living_area = None
        screening_previous_pregnancies = None
        if pinfo and pinfo["patient_name"]:
            screening_patient_name = pinfo["patient_name"]
            screening_patient_age = _to_int_or_none(pinfo["patient_age"])
            screening_mother_age = _to_int_or_none(pinfo["mother_age"])
            screening_father_age = _to_int_or_none(pinfo["father_age"])
            screening_parent_relation = (pinfo["parent_relation"] or pinfo["relation"] or "").strip() or None
            screening_living_area = (pinfo["living_area"] or pinfo["income"] or "").strip() or None
            screening_previous_pregnancies = _to_int_or_none(pinfo["previous_pregnancies"])

        risk_factor_explanation_txt = str(explanation_sections.get("risk_factor_explanation", "") or "")
        affected_region_explanation_txt = str(explanation_sections.get("affected_region_explanation", "") or "")
        future_health_explanation_txt = str(explanation_sections.get("future_health_explanation", "") or "")
        food_explanation_txt = str(explanation_sections.get("food_explanation", "") or "")
        exercise_explanation_txt = str(explanation_sections.get("exercise_explanation", "") or "")
        video_titles_json = json.dumps(list(llm_video_titles or []), ensure_ascii=False)
        food_recommendations_json = json.dumps(list(llm_food_list or []), ensure_ascii=False)
        exercise_recommendations_json = json.dumps(list(llm_exercise_list or []), ensure_ascii=False)
        risk_factors_json = json.dumps(list((risk_factor_sections or {}).get("gradcam") or []), ensure_ascii=False)
        risk_factor_sections_json = json.dumps(risk_factor_sections or {}, ensure_ascii=False)
        risk_factor_highlights_json = json.dumps(risk_factor_highlights or [], ensure_ascii=False)
        affected_region_details_json = json.dumps(affected_region_details or [], ensure_ascii=False)
        future_health_issues_json = json.dumps(future_health_issues or [], ensure_ascii=False)
        explanation_sections_json = json.dumps(explanation_sections or {}, ensure_ascii=False)
        frontend_sections_json = json.dumps(frontend_sections or {}, ensure_ascii=False)

        conn = get_db()
        conn.execute(
            """
            INSERT INTO screenings (
                user_email, patient_name, patient_age, mother_age, father_age,
                parent_relation, living_area, previous_pregnancies,
                date, uploaded_image, heatmap_image,
                probability, confidence, high_confidence,
                facial_symmetry, eye_spacing, nasal_bridge, ear_position,
                explanation, recommended_food, recommended_exercises, risk_level, title,
                risk_factor_explanation, affected_region_explanation, future_health_explanation,
                food_explanation, exercise_explanation, video_titles, food_recommendations,
                exercise_recommendations, risk_factors, risk_factor_sections, risk_factor_highlights,
                affected_region_details, future_health_issues, explanation_sections, frontend_sections
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_email,
                screening_patient_name,
                screening_patient_age,
                screening_mother_age,
                screening_father_age,
                screening_parent_relation,
                screening_living_area,
                screening_previous_pregnancies,
                f"/uploads/{filename}",
                f"/outputs/{heatmap_name}",
                prob_adj,
                confidence,
                int(confidence >= 0.95),
                features["facial_symmetry"],
                features["eye_spacing"],
                features["nasal_bridge"],
                features["ear_position"],
                explanation_text,
                food_rec,
                exercise_rec,
                risk_level,
                llm_title,
                risk_factor_explanation_txt,
                affected_region_explanation_txt,
                future_health_explanation_txt,
                food_explanation_txt,
                exercise_explanation_txt,
                video_titles_json,
                food_recommendations_json,
                exercise_recommendations_json,
                risk_factors_json,
                risk_factor_sections_json,
                risk_factor_highlights_json,
                affected_region_details_json,
                future_health_issues_json,
                explanation_sections_json,
                frontend_sections_json,
            ),
        )
        conn.commit()
        conn.close()

        print("----- DEBUG START -----")
        print("Raw output:", raw_output)
        print("Down prob (1-raw):", prob)
        print("Adjusted prob:", prob_adj)
        print("Confidence:", confidence)
        print("Risk level:", risk_level)
        print("----- DEBUG END -----")

        prob_adj = float(prob_adj if prob_adj is not None else 0.0)
        confidence = float(confidence if confidence is not None else 0.0)
        risk_color = "#E74C3C" if risk_level == "high" else ("#F1C40F" if risk_level == "moderate" else "#2ECC71")

        return jsonify(
            {
                "success": True,
                "probability": round(prob_adj, 3),
                "confidence": round(confidence, 3),
                "high_confidence": bool(int(round(confidence * 100)) >= 95),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "patient_snapshot": {
                    "patient_name": screening_patient_name,
                    "patient_age": screening_patient_age,
                    "mother_age": screening_mother_age,
                    "father_age": screening_father_age,
                    "parent_relation": screening_parent_relation,
                    "living_area": screening_living_area,
                    "previous_pregnancies": screening_previous_pregnancies,
                },
                "uploaded_image": f"/uploads/{filename}",
                "heatmap": f"/outputs/{heatmap_name}",
                "title": llm_title,
                "explanation": explanation_text,
                "risk_factor_explanation": explanation_sections.get("risk_factor_explanation", ""),
                "affected_region_explanation": explanation_sections.get("affected_region_explanation", ""),
                "future_health_explanation": explanation_sections.get("future_health_explanation", ""),
                "food_explanation": explanation_sections.get("food_explanation", ""),
                "exercise_explanation": explanation_sections.get("exercise_explanation", ""),
                "explanation_sections": explanation_sections,
                "frontend_sections": frontend_sections,
                "video_titles": llm_video_titles,
                "recommended_food": food_rec,
                "recommended_exercises": exercise_rec,
                "food_recommendations": llm_food_list,
                "exercise_recommendations": llm_exercise_list,
                "llm_model_used": llm_model_used,
                "llm_output_source": llm_output_source,
                "raw_llm_output": raw_llm_output,
                "llm_sections": {
                    "affected_regions": llm_affected_regions,
                    "food_recommendations": llm_food_list,
                    "exercise_recommendations": llm_exercise_list,
                    "video_titles": llm_video_titles,
                    "risk_factors": list((risk_factor_sections or {}).get("gradcam") or []),
                    "risk_factor_sections": risk_factor_sections,
                    "risk_factor_highlights": risk_factor_highlights,
                    "affected_region_details": affected_region_details,
                    "future_health_issues": future_health_issues,
                    "explanation_sections": explanation_sections,
                    "frontend_sections": frontend_sections,
                },
                "analysis_details": {
                    "activation_level": status,
                    "affected_regions": affected_regions,
                },
                "facial_features": features,
                "facial_theory": theory,
                "gradcam_risk_factors": gradcam_risk_factors,
                "risk_factors": list((risk_factor_sections or {}).get("gradcam") or []),
                "risk_matrix": risk_matrix,
                "risk_factor_sections": risk_factor_sections,
                "risk_factor_highlights": risk_factor_highlights,
                "affected_region_details": affected_region_details,
                "future_health_issues": future_health_issues,
                "xai_details": {
                    "affected_regions": affected_regions,
                    "region_scores": region_scores,
                },
            }
        )
    except TimeoutException:
        print("Prediction error: llm timed out")
        return jsonify({"success": False, "error": "llm request timed out. Please retry."}), 504
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"success": False, "error": "Prediction failed"}), 500
# ---------------- DOWNLOAD REPORT ----------------
@app.route("/download_report/<int:screening_id>", methods=["GET"])
@jwt_required()
def download_report(screening_id):
    user_email = get_jwt_identity()
    conn = get_db()
    record = conn.execute("SELECT * FROM screenings WHERE id = ?", (screening_id,)).fetchone()
    conn.close()
    if not record:
        return jsonify({"success": False, "error": "Record not found"}), 404
    if record["user_email"] != user_email:
        return jsonify({"success": False, "error": "Forbidden"}), 403

    # Fetch patient_info for this user (used in the PDF)
    conn = get_db()
    pinfo = conn.execute(
        """SELECT patient_age, mother_age, father_age, parent_relation, living_area,
                  previous_pregnancies, relation, income, spouse_relation, annual_income,
                  extra_info, notes
           FROM patient_info
           WHERE user_email = ?""",
        (record['user_email'],)
    ).fetchone()
    conn.close()

    def _sanitize_for_pdf(text):
        """Normalize common unicode punctuation and ensure latin-1 safe string.
        Replaces em-dash, smart quotes, ellipsis, non-breaking spaces, then
        falls back to latin-1 replacement for any remaining unsupported chars.
        """
        if text is None:
            return ""
        try:
            s = str(text)
        except Exception:
            s = ""
        s = unicodedata.normalize("NFKD", s)
        reps = {
            "\u2014": " - ",
            "\u2013": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": '...',
            "\u00a0": ' ',
        }
        for k, v in reps.items():
            s = s.replace(k, v)
        try:
            s.encode("latin-1")
            return s
        except UnicodeEncodeError:
            return s.encode("latin-1", "replace").decode("latin-1")

    def _json_load_maybe(value, default):
        raw = str(value or "").strip()
        if not raw:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default

    def _to_text_lines(value, limit=12):
        out = []
        if isinstance(value, list):
            for item in value[:limit]:
                if isinstance(item, dict):
                    issue = str(item.get("issue") or item.get("feature") or item.get("region") or "").strip()
                    sev = str(item.get("risk_level") or item.get("severity") or "").strip()
                    mon = str(item.get("what_to_monitor") or item.get("explanation") or "").strip()
                    line = issue
                    if sev:
                        line += f" [{sev}]"
                    if mon:
                        line += f": {mon}"
                    if line.strip():
                        out.append(line.strip())
                else:
                    s = str(item).strip()
                    if s:
                        out.append(s)
        elif isinstance(value, dict):
            for k, v in list(value.items())[:limit]:
                out.append(f"{k}: {v}")
        else:
            s = str(value or "").strip()
            if s:
                out.append(s)
        return out[:limit]

    def _extract_explanation_text(raw_value):
        raw_text = str(raw_value or "").strip()
        if not raw_text:
            return "N/A"
        parsed = _json_load_maybe(raw_text, None)
        if isinstance(parsed, dict):
            expl = str(parsed.get("explanation") or "").strip()
            if expl:
                return expl
        return raw_text

    def _normalize_recommendation_lines(lines, limit=12):
        out = []
        for line in lines:
            text = str(line or "").strip()
            if not text:
                continue
            text = re.sub(r"(?i)^(food|foods|exercise|exercises)\s*:\s*", "", text).strip()
            text = re.sub(r"^\s*-\s*", "", text).strip()
            parts = [p.strip() for p in re.split(r"[;\n]+|,\s*", text) if p.strip()]
            for p in parts:
                if p and p not in out:
                    out.append(p)
                if len(out) >= limit:
                    return out
        return out[:limit]

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    epw = pdf.w - pdf.l_margin - pdf.r_margin

    def _ensure_space(min_height=12):
        if pdf.get_y() + float(min_height) > pdf.page_break_trigger:
            pdf.add_page()

    def _display_value(value, fallback="N/A"):
        if value is None:
            return fallback
        s = str(value).strip()
        return s if s else fallback

    def _friendly_value(value, mapping=None):
        raw = _display_value(value)
        if raw == "N/A":
            return raw
        if mapping:
            return mapping.get(raw, raw.replace("_", " ").title())
        return raw.replace("_", " ").title()

    def _draw_section_title(title):
        _ensure_space(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(245, 247, 250)
        pdf.cell(0, 7, _sanitize_for_pdf(title), border=1, ln=True, fill=True)

    def _draw_key_value_rows(rows):
        _ensure_space(max(10, len(rows) * 6 + 2))
        key_w = epw * 0.34
        val_w = epw - key_w
        pdf.set_font('Arial', '', 10)
        for key, val in rows:
            _ensure_space(7)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(key_w, 6, _sanitize_for_pdf(str(key)), border=1)
            pdf.set_font('Arial', '', 10)
            pdf.cell(val_w, 6, _sanitize_for_pdf(_display_value(val)), border=1, ln=True)

    def _draw_bullet_box(title, items, limit=8):
        lines = [str(x).strip() for x in (items or []) if str(x).strip()]
        lines = lines[:limit]
        if not lines:
            return
        _ensure_space(14)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, _sanitize_for_pdf(title), ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(epw, 6, _sanitize_for_pdf("- " + "\n- ".join(lines)), border=1)

    # Header
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, _sanitize_for_pdf('SMART FACIAL SCREENING FOR DS DETECTION'), ln=True, align='C')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, _sanitize_for_pdf('Automated AI-assisted screening - not a diagnosis'), ln=True, align='C')
    pdf.ln(4)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    relation_map = {
        "non_relative": "Not Related",
        "first_cousin": "First Cousin",
        "second_cousin": "Second Cousin",
        "distant_relative": "Distant Relative",
        "same_family": "Same Extended Family",
    }
    living_area_map = {
        "urban_area": "Urban Area",
        "semi_urban_area": "Semi-Urban Area",
        "rural_area": "Rural Area",
        "tribal_area": "Tribal Area",
        "metro_city": "Metro City",
        "city": "City",
        "town": "Town",
        "village": "Village",
        "remote_village": "Remote Village",
    }

    # Patient info details
    patient_age_value = record["patient_age"] if "patient_age" in record.keys() and record["patient_age"] is not None else (pinfo["patient_age"] if pinfo and pinfo["patient_age"] is not None else "N/A")
    mother_age_value = record["mother_age"] if "mother_age" in record.keys() and record["mother_age"] is not None else (pinfo["mother_age"] if pinfo and pinfo["mother_age"] is not None else "N/A")
    father_age_value = record["father_age"] if "father_age" in record.keys() and record["father_age"] is not None else (pinfo["father_age"] if pinfo and pinfo["father_age"] is not None else "N/A")
    parent_relation_value = record["parent_relation"] if "parent_relation" in record.keys() and record["parent_relation"] else ((pinfo["parent_relation"] if pinfo and pinfo["parent_relation"] else (pinfo["relation"] if pinfo and pinfo["relation"] else "N/A")))
    living_area_value = record["living_area"] if "living_area" in record.keys() and record["living_area"] else ((pinfo["living_area"] if pinfo and pinfo["living_area"] else (pinfo["income"] if pinfo and pinfo["income"] else "N/A")))
    previous_pregnancies_value = record["previous_pregnancies"] if "previous_pregnancies" in record.keys() and record["previous_pregnancies"] is not None else (pinfo["previous_pregnancies"] if pinfo and pinfo["previous_pregnancies"] is not None else "N/A")
    _draw_section_title("Patient Information")
    _draw_key_value_rows([
        ("Name", record["patient_name"]),
        ("Date", record["date"]),
        ("Child Age", patient_age_value),
        ("Mother Age", mother_age_value),
        ("Father Age", father_age_value),
        ("Parents Relation", _friendly_value(parent_relation_value, relation_map)),
        ("Living Area", _friendly_value(living_area_value, living_area_map)),
        ("Previous Pregnancies", previous_pregnancies_value),
    ])

    _draw_section_title("Screening Summary")
    prob_pct = int(round(record['probability'] * 100))
    conf_pct = int(round(record['confidence'] * 100))
    rl = record['risk_level'] if record['risk_level'] else ("High" if record['probability'] >= 0.7 else ("Moderate" if record['probability'] >= 0.4 else "Low"))
    _draw_key_value_rows([
        ("Probability", f"{prob_pct}%"),
        ("Confidence", f"{conf_pct}%"),
        ("Risk Level", rl.capitalize()),
        ("High Confidence", "Yes" if record['high_confidence'] else "No"),
    ])
    pdf.ln(3)

    _draw_section_title("Facial Features (AI Analysis)")
    facial_symmetry = float(record["facial_symmetry"] or 0.0)
    eye_spacing = float(record["eye_spacing"] or 0.0)
    nasal_bridge = float(record["nasal_bridge"] or 0.0)
    ear_position = float(record["ear_position"] or 0.0)
    _draw_key_value_rows([
        ("Facial Symmetry", f"{facial_symmetry:.2f}"),
        ("Eye Spacing", f"{eye_spacing:.2f}"),
        ("Nasal Bridge", f"{nasal_bridge:.2f}"),
        ("Ear Position", f"{ear_position:.2f}"),
    ])
    pdf.ln(4)

    # Images side-by-side
    upload_path = os.path.join(BASE_DIR, record['uploaded_image'][1:])
    heatmap_path = os.path.join(BASE_DIR, record['heatmap_image'][1:])
    img_h = 60
    img_w = (epw - 4) / 2
    x0 = pdf.l_margin
    _ensure_space(img_h + 8)
    y_img_base = pdf.get_y()
    if os.path.exists(upload_path):
        pdf.image(upload_path, x=x0, y=y_img_base, w=img_w, h=img_h)
    else:
        pdf.rect(x0, y_img_base, img_w, img_h)
        pdf.set_xy(x0, y_img_base + (img_h / 2) - 3)
        pdf.cell(img_w, 6, _sanitize_for_pdf("Uploaded image not available"), align='C')
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=x0 + img_w + 4, y=y_img_base, w=img_w, h=img_h)
    else:
        pdf.rect(x0 + img_w + 4, y_img_base, img_w, img_h)
        pdf.set_xy(x0 + img_w + 4, y_img_base + (img_h / 2) - 3)
        pdf.cell(img_w, 6, _sanitize_for_pdf("Heatmap image not available"), align='C')
    pdf.set_xy(pdf.l_margin, y_img_base)
    pdf.ln(img_h + 6)

    # AI Explanation boxed
    _draw_section_title("AI Explanation")
    pdf.set_font('Arial', '', 10)
    pdf.set_x(pdf.l_margin)
    exp_raw = _extract_explanation_text(record['explanation'])
    exp_raw = re.sub(r"\s+", " ", str(exp_raw)).strip()
    exp_words = exp_raw.split()
    if len(exp_words) > 220:
        exp_raw = " ".join(exp_words[:220]) + " ..."
    exp_text = _sanitize_for_pdf(exp_raw)
    pdf.multi_cell(epw, 6, exp_text, border=1)
    pdf.ln(4)

    # Recommendations
    _draw_section_title("Recommendations")
    food_lines = []
    exercise_lines = []
    if record['recommended_food']:
        food_lines.append(str(record['recommended_food']))
    if record['recommended_exercises']:
        exercise_lines.append(str(record['recommended_exercises']))
    stored_food_items = _to_text_lines(_json_load_maybe(record["food_recommendations"] if "food_recommendations" in record.keys() else "", []), limit=12)
    stored_exercise_items = _to_text_lines(_json_load_maybe(record["exercise_recommendations"] if "exercise_recommendations" in record.keys() else "", []), limit=12)
    if stored_food_items:
        food_lines.extend(stored_food_items)
    if stored_exercise_items:
        exercise_lines.extend(stored_exercise_items)
    food_lines = _normalize_recommendation_lines(food_lines, limit=12)
    exercise_lines = _normalize_recommendation_lines(exercise_lines, limit=12)

    _draw_bullet_box("Food", food_lines, limit=10)
    _draw_bullet_box("Exercises", exercise_lines, limit=10)
    if (not food_lines) and (not exercise_lines):
        pdf.multi_cell(epw, 6, _sanitize_for_pdf("Recommendations are not available for this record."), border=1)

    # Stored detailed result fields (if available)
    risk_factor_expl = _sanitize_for_pdf(record["risk_factor_explanation"] or "")
    affected_region_expl = _sanitize_for_pdf(record["affected_region_explanation"] or "")
    future_health_expl = _sanitize_for_pdf(record["future_health_explanation"] or "")
    food_expl = _sanitize_for_pdf(record["food_explanation"] or "")
    exercise_expl = _sanitize_for_pdf(record["exercise_explanation"] or "")

    stored_videos = _json_load_maybe(record["video_titles"], [])
    stored_risk_factors = _json_load_maybe(record["risk_factors"] if "risk_factors" in record.keys() else "", [])
    stored_risk_sections = _json_load_maybe(record["risk_factor_sections"] if "risk_factor_sections" in record.keys() else "", {})
    stored_risk_highlights = _json_load_maybe(record["risk_factor_highlights"], [])
    stored_region_details = _json_load_maybe(record["affected_region_details"], [])
    stored_future_issues = _json_load_maybe(record["future_health_issues"], [])
    video_lines = _to_text_lines(stored_videos, limit=6)

    if video_lines:
        _ensure_space(18)
        _draw_section_title("Exercise Video Titles")
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(epw, 6, _sanitize_for_pdf("- " + "\n- ".join(video_lines)), border=1, align='L')
        pdf.ln(2)

    # Build robust fallbacks so these sections print whenever backend has partial data.
    risk_lines = _to_text_lines(stored_risk_highlights, limit=8)
    if not risk_lines:
        risk_lines = _to_text_lines(stored_risk_factors, limit=8)
    if (not risk_lines) and isinstance(stored_risk_sections, dict):
        risk_lines = _to_text_lines(stored_risk_sections.get("gradcam") or [], limit=8)
        if not risk_lines:
            risk_lines = _to_text_lines(stored_risk_sections.get("facial_features") or [], limit=8)
    if (not risk_lines) and risk_factor_expl:
        expl_parts = [p.strip(" -") for p in re.split(r"[;\n]+", str(risk_factor_expl)) if p.strip()]
        risk_lines = expl_parts[:8]

    issue_lines = _to_text_lines(stored_future_issues, limit=8)
    if (not issue_lines) and future_health_expl:
        issue_parts = [p.strip(" -") for p in re.split(r"[;\n]+", str(future_health_expl)) if p.strip()]
        issue_lines = issue_parts[:8]

    region_lines = _to_text_lines(stored_region_details, limit=6)

    if any([
        risk_factor_expl, affected_region_expl, future_health_expl, food_expl, exercise_expl,
        stored_risk_highlights, stored_risk_factors, stored_risk_sections,
        stored_region_details, stored_future_issues, risk_lines, issue_lines
    ]):
        _ensure_space(18)
        pdf.ln(4)
        _draw_section_title("Detailed Result Fields")
        pdf.set_font('Arial', '', 10)
        detail_parts = []
        if risk_factor_expl:
            detail_parts.append(f"Risk Factor Explanation: {risk_factor_expl}")
        if affected_region_expl:
            detail_parts.append(f"Affected Region Explanation: {affected_region_expl}")
        if future_health_expl:
            detail_parts.append(f"Future Health Explanation: {future_health_expl}")
        if food_expl:
            detail_parts.append(f"Food Explanation: {food_expl}")
        if exercise_expl:
            detail_parts.append(f"Exercise Explanation: {exercise_expl}")
        if risk_lines:
            detail_parts.append("Risk Factors:\n- " + "\n- ".join(risk_lines))
        if region_lines:
            detail_parts.append("Affected Region Details:\n- " + "\n- ".join(region_lines))
        if issue_lines:
            detail_parts.append("Future Health Issues:\n- " + "\n- ".join(issue_lines))

        if detail_parts:
            detail_text = "\n\n".join(detail_parts)
            pdf.multi_cell(epw, 6, _sanitize_for_pdf(detail_text), border=1)

    # Footer / metadata
    pdf.ln(6)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 5, _sanitize_for_pdf('This report is for screening purposes only and is not a diagnosis.'), ln=True, align='C')

    report_filename = f"screening_report_{record['id']}.pdf"
    report_path = os.path.join(OUTPUT_FOLDER, report_filename)
    pdf.output(report_path)
    return send_from_directory(OUTPUT_FOLDER, report_filename, as_attachment=True)

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET"])
@jwt_required()
def dashboard():
    user_email = get_jwt_identity()
    conn = get_db()
    screenings = conn.execute("""
        SELECT id, patient_name, date, probability, confidence, high_confidence,
               uploaded_image, heatmap_image, recommended_exercises, risk_level
        FROM screenings
        WHERE user_email = ?
        ORDER BY date DESC
    """, (user_email,)).fetchall()
    conn.close()

    data = [dict(screening) for screening in screenings]

    total_screenings = len(data)
    high_risk_count = sum(1 for s in data if s["probability"] >= 0.7)
    medium_risk_count = sum(1 for s in data if 0.4 <= s["probability"] < 0.7)
    low_risk_count = sum(1 for s in data if s["probability"] < 0.4)

    return jsonify({
        "screenings": data,
        "stats": {
            "total": total_screenings,
            "high_risk": high_risk_count,
            "medium_risk": medium_risk_count,
            "low_risk": low_risk_count
        }
    })

# ---------------- STATIC FILES ----------------
@app.route("/uploads/<path:filename>")
@jwt_required()
def serve_uploads(filename):
    user_email = get_jwt_identity()
    rel_path = f"/uploads/{filename}"
    conn = get_db()
    allowed = conn.execute(
        "SELECT 1 FROM screenings WHERE user_email = ? AND uploaded_image = ? LIMIT 1",
        (user_email, rel_path),
    ).fetchone()
    conn.close()
    if not allowed:
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/outputs/<path:filename>")
@jwt_required()
def serve_outputs(filename):
    user_email = get_jwt_identity()
    rel_path = f"/outputs/{filename}"
    conn = get_db()
    allowed = conn.execute(
        "SELECT 1 FROM screenings WHERE user_email = ? AND heatmap_image = ? LIMIT 1",
        (user_email, rel_path),
    ).fetchone()
    conn.close()
    if not allowed:
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(OUTPUT_FOLDER, filename)

# ---------------- RUN ----------------
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))  # Use the port Render gives
    app.run(host="0.0.0.0", port=PORT)



