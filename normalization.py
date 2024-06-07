import numpy as np
import cv2
import os

# Input and output directories
input_dir = "data/original/validation/1/"
output_dir = "data/stain_normalization/validation/1/"

# Get list of images in the input directory
input_image_list = os.listdir(input_dir)

def get_mean_and_std(image):
    """
    Compute the mean and standard deviation of each channel in the given image.
    
    Args:
        image (numpy.ndarray): The input image.
    
    Returns:
        tuple: Means and standard deviations of the image channels.
    """
    mean, std = cv2.meanStdDev(image)
    mean = np.hstack(np.around(mean, 2))
    std = np.hstack(np.around(std, 2))
    return mean, std

# Load the template image and convert it to LAB color space
template_img = cv2.imread('data/utils/stitched_image.png')
template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
template_mean, template_std = get_mean_and_std(template_img)

# Process each image in the input directory
for img_name in input_image_list:
    print(f"Processing image: {img_name}")
    
    # Read the input image and convert it to LAB color space
    input_img = cv2.imread(os.path.join(input_dir, img_name))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    
    # Compute the mean and standard deviation of the input image
    img_mean, img_std = get_mean_and_std(input_img)
    
    height, width, channels = input_img.shape
    
    # Normalize each pixel in the image
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                pixel_value = input_img[i, j, k]
                # Normalize the pixel value using the template image statistics
                normalized_value = ((pixel_value - img_mean[k]) * (template_std[k] / img_std[k])) + template_mean[k]
                normalized_value = round(normalized_value)
                # Boundary check to keep the pixel value within valid range [0, 255]
                normalized_value = max(0, min(255, normalized_value))
                input_img[i, j, k] = normalized_value
    
    # Convert the normalized image back to BGR color space
    input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
    
    # Save the processed image to the output directory
    output_path = os.path.join(output_dir, f"modified_{img_name}")
    cv2.imwrite(output_path, input_img)
