from PIL import Image
import numpy as np

def preprocess_image(img):
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)
