import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_disparity(left_img, right_img, num_disparities=64, block_size=15):
    """
    Computes a disparity map using OpenCV's StereoBM algorithm.

    Parameters:
        left_img (np.ndarray): Grayscale left image.
        right_img (np.ndarray): Grayscale right image.
        num_disparities (int): Maximum disparity minus minimum disparity. Must be divisible by 16.
        block_size (int): Matching block size. Must be odd and within 5..255.

    Returns:
        disparity_map (np.ndarray): Computed disparity map.
    """
    # Create StereoBM object. Note: numDisparities must be divisible by 16.
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    
    # Compute the raw disparity (in fixed-point format)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Replace negative disparities (if any) with 0 for visualization
    disparity[disparity < 0] = 0

    # Apply a median filter to reduce noise
    disparity_filtered = cv2.medianBlur(disparity, 5)

    return disparity_filtered

def create_heatmap(disparity_map, colormap=cv2.COLORMAP_JET):
    """
    Normalize the disparity map and apply a colormap for visualization.

    Parameters:
        disparity_map (np.ndarray): The raw disparity map.
        colormap (int): OpenCV colormap to use (default is cv2.COLORMAP_JET).

    Returns:
        colored_map (np.ndarray): Color-mapped image.
    """
    # Normalize disparity map to range 0-255 for proper visualization
    disp_norm = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    # Apply the chosen colormap
    colored_map = cv2.applyColorMap(disp_norm, colormap)
    return colored_map

if __name__ == "__main__":
    # Load the stereo images in grayscale
    left_img_path = 'left.png'
    right_img_path = 'right.png'
    
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    
    if left_img is None or right_img is None:
        print("Error: One or both images not found. Please ensure 'left.png' and 'right.png' exist.")
        exit(1)
    
    print("Computing disparity map using StereoBM...")
    disparity_map = compute_disparity(left_img, right_img, num_disparities=64, block_size=15)
    
    # Create a heatmap from the refined disparity map
    heatmap = create_heatmap(disparity_map, colormap=cv2.COLORMAP_JET)
    
    # Save the final depth (heatmap) image
    output_path = 'depth.png'
    cv2.imwrite(output_path, heatmap)
    print(f"Depth map saved as {output_path}")
    
    # Display the images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(left_img, cmap='gray')
    plt.title("Left Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(right_img, cmap='gray')
    plt.title("Right Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # Convert BGR to RGB for correct color display in matplotlib
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Depth Map Heatmap")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
#  import cv2
#  import os
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def check_and_load_image(filename):
# #     abs_path = os.path.abspath(filename)
# #     print(f"Checking: {abs_path}")
    
# #     if not os.path.exists(abs_path):
# #         print(f"Error: {filename} not found!")
# #         return None
    
# #     img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
# #     if img is None:
# #         print(f"Error: {filename} could not be loaded. Check file integrity.")
# #     else:
# #         print(f"Successfully loaded {filename}")
    
# #     return img

# # def generate_depth_map(left_img, right_img):
# #     stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
# #     disparity = stereo.compute(left_img, right_img)
    
# #     plt.imshow(disparity, cmap='plasma')  # Plasma gives better visualization
# #     plt.colorbar(label='Depth')
# #     plt.title('Depth Map')
# #     plt.savefig('depth.png')  # Save the depth map
# #     plt.show()

# # def main():
# #     print("Current Working Directory:", os.getcwd())
    
# #     left_img = check_and_load_image("left.png")
# #     right_img = check_and_load_image("right.png")
    
# #     if left_img is not None and right_img is not None:
# #         print("Both images loaded successfully. Generating depth map...")
# #         generate_depth_map(left_img, right_img)
# #     else:
# #         print("One or both images failed to load. Cannot generate depth map.")

# if __name__ == "__main__":
# #     main()
