
import tensorflow as tf
import os
from PIL import Image
import numpy as np

dir_path = 'data/original/training/0'
img_list = os.listdir(dir_path)

save_dir_path = "data/stain_augmentation/training/0"
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

def random_color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=0.35)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.5)
    return image

for img_path in img_list:
    full_img_path = os.path.join(dir_path, img_path)
    image = Image.open(full_img_path)
    image_np = np.array(image)
    
    # Apply random color jitter using TensorFlow
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32) / 255.0
    image_tf = random_color_jitter(image_tf)
    
    # Convert the image back to the PIL format
    image_np = tf.image.convert_image_dtype(image_tf, dtype=tf.uint8)
    image_np = image_np.numpy()
    image_augmented = Image.fromarray(image_np)
    
    save_img_path = os.path.join(save_dir_path, img_path)
    image_augmented.save(save_img_path)
