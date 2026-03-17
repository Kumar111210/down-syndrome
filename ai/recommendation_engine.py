import numpy as np
import tensorflow as tf
import cv2

def _get_last_conv_layer(model):
    for name in ["block_16_project_BN", "block_16_project", "Conv_1"]:
        try:
            model.get_layer(name)
            return name
        except ValueError:
            continue
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

def analyze_heatmap_regions(heatmap):
    h, w = heatmap.shape
    regions = {}
    regions["forehead"] = float(np.mean(heatmap[0:int(h*0.2), :]))/255.0
    regions["eyes"] = float(np.mean(heatmap[int(h*0.2):int(h*0.4), :]))/255.0
    regions["nose"] = float(np.mean(heatmap[int(h*0.4):int(h*0.6), :]))/255.0
    regions["mouth"] = float(np.mean(heatmap[int(h*0.6):int(h*0.8), :]))/255.0
    regions["chin"] = float(np.mean(heatmap[int(h*0.8):, :]))/255.0
    threshold = 0.35
    affected = [r for r,v in regions.items() if v > threshold]
    if not affected:
        affected = ["general face"]
    mean_val = float(np.mean(heatmap))
    if mean_val > 180:
        activation = "Strong facial feature activation"
    elif mean_val > 120:
        activation = "Moderate facial feature activation"
    else:
        activation = "Low facial feature activation"
    return {"affected_regions": affected, "mean_heatmap_value": round(mean_val, 2), "activation_level": activation}

def _recommendations_from_regions(affected):
    if any(x in affected for x in ["eyes", "forehead"]):
        food = "Omega-3 rich foods, leafy greens, fruits"
        exercises = "Eye focus drills, gentle neck stretches, breathing"
    elif any(x in affected for x in ["nose"]):
        food = "Balanced protein intake, hydration, vitamin C"
        exercises = "Facial yoga for nasal area, diaphragmatic breathing"
    elif any(x in affected for x in ["mouth", "chin"]):
        food = "Soft-texture balanced diet if needed, multivitamins"
        exercises = "Orofacial myofunctional routines, light jaw stretches"
    else:
        food = "Balanced diet"
        exercises = "Light exercises"
    return food, exercises

def explain_with_gradcam(img_array, model, prob, last_conv_layer_name="block_16_project_BN"):
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    analysis = analyze_heatmap_regions(heatmap)
    affected = analysis["affected_regions"]
    if prob > 0.7:
        base = "High likelihood detected."
    elif prob > 0.4:
        base = "Moderate likelihood detected."
    else:
        base = "Low likelihood detected."
    if affected:
        base += " Activated regions: " + ", ".join(affected) + "."
    food, exercises = _recommendations_from_regions(affected)
    return {
        "explanation": base,
        "affected_regions": affected,
        "recommended_food": food,
        "recommended_exercises": exercises,
        "mean_heatmap_value": analysis["mean_heatmap_value"],
        "activation_level": analysis["activation_level"],
        "heatmap": heatmap
    }

def generate_ai_explanation(prob, affected_regions):
    if prob > 0.7:
        level = "High likelihood detected."
    elif prob > 0.4:
        level = "Moderate likelihood detected."
    else:
        level = "Low likelihood detected."
    if affected_regions:
        level += " Activated regions: " + ", ".join(affected_regions) + "."
    return level
