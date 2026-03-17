import cv2
import numpy as np

def check_image_quality(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
    brightness = np.mean(img)
    is_valid = blur_score > 80 and 50 < brightness < 220
    message = "Good quality" if is_valid else "Poor quality"
    return {
        "blur_score": round(blur_score,2),
        "brightness": round(brightness,2),
        "is_valid": is_valid,
        "message": message
    }
