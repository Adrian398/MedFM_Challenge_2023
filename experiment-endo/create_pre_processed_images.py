import cv2
import numpy as np
import os
from tqdm import tqdm


def sh_detection(img_path, Ts=0.35, Tv=0.85):
    # Read the image
    img = cv2.imread(img_path)
    
    # Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, w, _ = img.shape
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    # Create Ts and Tv matrices of the same size as the input image
    Ts_mat = np.full((h, w), Ts)
    Tv_mat = np.full((h, w), Tv)
    
    # Compute highlight masks
    highlight_mask1 = (1 - np.exp((-2 * np.maximum(Ts_mat - saturation, 0)**2) / 0.01))
    highlight_mask2 = (1 - np.exp((-2 * np.maximum(value - Tv_mat, 0)**2) / 0.01))
    
    highlight_mask = highlight_mask1 * highlight_mask2
    
    return highlight_mask



def draw_black_on_mask(img, mask):
    binary_mask = (mask > 0.5).astype(np.uint8)
    output_img = img.copy()
    
    # Set pixels to black where the mask is set
    average_pixel = img.mean(axis=0).mean(axis=0)
    #output_img[binary_mask == 1] = [0, 0, 0]
    output_img[binary_mask == 1] = average_pixel
    
    print(average_pixel)
    return output_img


def process_image(input_directory, output_directory, filename):
    img_path = os.path.join(input_directory, filename)
    
    img = cv2.imread(img_path)
    img = img.astype(np.float32) / 255.0
    
    highlight_mask = sh_detection(img_path)
    output_img = draw_black_on_mask(img, highlight_mask)

    # Save the original and masked images to the output directory
    cv2.imwrite(os.path.join(output_directory, f'original_{filename}'), img * 255.0)
    cv2.imwrite(os.path.join(output_directory, f'masked_{filename}'), output_img * 255.0)



input_directory = '/scratch/medfm/medfm-challenge/data/MedFMC/endo/images'
output_directory = '/scratch/medfm/medfm-challenge/data/MedFMC/endo/pre_processed_images'

image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png'))]

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' removed successfully!")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied: Cannot remove '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def rename_file(original_path, new_path):
    try:
        os.rename(original_path, new_path)
        print(f"File '{original_path}' renamed to '{new_path}' successfully!")
    except FileNotFoundError:
        print(f"File '{original_path}' not found.")
    except PermissionError:
        print(f"Permission denied: Cannot rename '{original_path}'.")
    except FileExistsError:
        print(f"A file with the name '{new_path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")



for filename in tqdm(image_files, desc="Processing images"):
    #process_image(input_directory, output_directory, filename)
    img_path = os.path.join(output_directory, f'masked_{filename}')
    img_path_new = os.path.join(output_directory, filename)
    rename_file(img_path, img_path_new)
