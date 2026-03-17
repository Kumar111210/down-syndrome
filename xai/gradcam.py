import numpy as np
import tensorflow as tf
import cv2
import os
try:
    import mediapipe as mp
    _MP_HAS_SOLUTIONS = hasattr(mp, "solutions")
except Exception:
    mp = None
    _MP_HAS_SOLUTIONS = False

# ----------------- Grad-CAM -----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image array and model.
    """
    # Ensure model input is a tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Check if last conv layer exists
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in the model.")

    # Create a model that maps input to last conv layer output & predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    # Compute gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    # Resize to original image size
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)

    return heatmap

# ----------------- Save Heatmap -----------------
def save_heatmap(img_path, heatmap, output_path, alpha=0.4):
    """
    Save the Grad-CAM heatmap overlay on the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(output_path, superimposed_img)

# ----------------- Analyze Heatmap Facial Regions -----------------
def analyze_heatmap_regions_by_face(img_path, heatmap, threshold=100):
    """
    Analyze the heatmap using MediaPipe face landmarks to return affected facial regions:
    eyes, nose, mouth. This detects real landmark locations and computes mean
    heatmap activation inside small boxes around those landmarks.
    """
    img = cv2.imread(img_path)
    if img is None:
        return ["Image not found"]

    h_img, w_img = img.shape[:2]

    regions = []

    # Try to use MediaPipe FaceMesh if available; otherwise fall back to geometric slicing
    if mp is not None and _MP_HAS_SOLUTIONS:
        mp_face_mesh = mp.solutions.face_mesh
        try:
            with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_img)

                if not results or not getattr(results, 'multi_face_landmarks', None):
                    return ["Face not detected"]

                landmarks = results.multi_face_landmarks[0]

                # Landmark indices (MediaPipe face mesh)
                LEFT_EYE = [33, 133]
                RIGHT_EYE = [362, 263]
                NOSE = [1]
                MOUTH = [13, 14]

                def get_region_mean(indices):
                    coords = []
                    for idx in indices:
                        lm = landmarks.landmark[idx]
                        x, y = int(lm.x * w_img), int(lm.y * h_img)
                        coords.append((x, y))

                    xs = [c[0] for c in coords]
                    ys = [c[1] for c in coords]

                    x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, w_img)
                    y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, h_img)

                    # Ensure indices within heatmap bounds
                    hm_h, hm_w = heatmap.shape[:2]
                    x1_h = int(x1 * (hm_w / w_img))
                    x2_h = int(x2 * (hm_w / w_img))
                    y1_h = int(y1 * (hm_h / h_img))
                    y2_h = int(y2 * (hm_h / h_img))

                    if x1_h >= x2_h or y1_h >= y2_h:
                        return 0.0

                    region = heatmap[y1_h:y2_h, x1_h:x2_h]
                    if region.size == 0:
                        return 0.0
                    return float(np.mean(region))

                eyes_mean = max(get_region_mean(LEFT_EYE), get_region_mean(RIGHT_EYE))
                nose_mean = get_region_mean(NOSE)
                mouth_mean = get_region_mean(MOUTH)

                if eyes_mean > threshold:
                    regions.append("Eyes region highly activated")
                if nose_mean > threshold:
                    regions.append("Nose region highly activated")
                if mouth_mean > threshold:
                    regions.append("Mouth region highly activated")

                if not regions:
                    regions.append("No major facial region activation detected")

                return regions
        except Exception:
            # If MediaPipe fails at runtime, fall through to geometric fallback
            pass

    # ----------------- Geometric fallback (no MediaPipe) -----------------
    # Compute simple horizontal slices (eyes/nose/mouth) based on heatmap proportions
    hm_h, hm_w = heatmap.shape[:2]
    forehead = float(np.mean(heatmap[0:int(hm_h*0.2), :]))
    eyes = float(np.mean(heatmap[int(hm_h*0.2):int(hm_h*0.4), :]))
    nose = float(np.mean(heatmap[int(hm_h*0.4):int(hm_h*0.6), :]))
    mouth = float(np.mean(heatmap[int(hm_h*0.6):int(hm_h*0.8), :]))
    chin = float(np.mean(heatmap[int(hm_h*0.8):, :]))

    region_vals = {
        "forehead": forehead,
        "eyes": eyes,
        "nose": nose,
        "mouth": mouth,
        "chin": chin,
    }

    dynamic_threshold = max(30, min(120, np.mean(list(region_vals.values())) + 10))
    if eyes > dynamic_threshold:
        regions.append("Eyes region highly activated")
    if nose > dynamic_threshold:
        regions.append("Nose region highly activated")
    if mouth > dynamic_threshold:
        regions.append("Mouth region highly activated")
    if not regions:
        regions.append("No major facial region activation detected")

    return regions

# ----------------- AI Explanation -----------------
def generate_ai_explanation(probability, affected_regions):
    """
    Generate explanation and recommendations based on prediction & facial regions.
    """
    explanation = ""
    food_recommendation = ""
    exercise_recommendation = ""

    # Explanation based on probability
    if probability > 0.7:
        explanation = "High likelihood detected."
    elif probability > 0.4:
        explanation = "Moderate likelihood detected."
    else:
        explanation = "Low likelihood detected."

    # Add region influence
    if affected_regions:
        explanation += f" Activated regions: {', '.join(affected_regions)}."

    # Recommendations based on regions
    if affected_regions:
        region = affected_regions[0]
        if "highly" in region.lower():
            food_recommendation = "Protein-rich diet with vitamins and minerals"
            exercise_recommendation = "Cardio, Strength training, and Stretching"
        else:
            food_recommendation = "Maintain a healthy diet"
            exercise_recommendation = "Light physical activity, daily walks"

    return {
        "explanation": explanation,
        "recommended_food": food_recommendation,
        "recommended_exercises": exercise_recommendation
    }
