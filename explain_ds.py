from PIL import Image

# Change this line ↓↓↓
path = r"C:\Users\kumar\OneDrive\Pictures\Desktop\some_real_photo.jpg"

try:
    img = Image.open(path)
    print("Image loaded successfully! Size:", img.size)
except Exception as e:
    print("Error:", e)