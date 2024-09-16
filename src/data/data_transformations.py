from PIL import Image
import numpy as np
import pandas as pd

def img_to_tabular(img):
    # Check if the image is already a PIL image
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    
    # Convert to RGB mode if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to numpy array
    img_np = np.array(img)
    
    # Flatten the image and convert to a tabular format
    img_tab = pd.DataFrame(img_np.reshape(-1, 3), columns=['r', 'g', 'b'])
    img_flat = img_tab.values
    
    return img, img_np, img_flat, img_tab
    
def downsample_image(img, target_pixels = 10000):
    original_pixels = img.width * img.height
    
    # Check if downsampling is needed
    if original_pixels <= target_pixels:
        return img
    
    # Calculate downsample factor based on target size
    downsample_factor = int(np.sqrt(original_pixels / target_pixels))
    
    downsampled_img_pil = img.resize(
        (img.width // downsample_factor, img.height // downsample_factor)
    )
    
    return downsampled_img_pil
    
