# import cv2
import numpy as np
import math
import cv2
def extract_filter(pi_image_path):
    # Load pi image in grayscale
    pi_image = cv2.imread(pi_image_path, cv2.IMREAD_GRAYSCALE)
    pixels = pi_image[:2, :2].flatten()
    print(f"Extracted pixels: {pixels}")

    # Sort pixels in descending order
    pixels_sorted = np.sort(pixels)[::-1]
    print(f"Sorted pixels: {pixels_sorted}")

    # Apply transformation: floor( (1 / 10π) * pixel )
    transformed_pixels = np.floor((1 / (10 * math.pi)) * pixels_sorted).astype(np.uint8)
    print(f"Transformed pixels: {transformed_pixels}")

    # Reshape into a 2×2 matrix
    filter_2x2 = transformed_pixels.reshape(2, 2)
    print(f"Final 2×2 Filter:\n{filter_2x2}")
    return filter_2x2

def recover_portrait(distorted_image_path, filter_2x2):
    

    distorted_img = cv2.imread(distorted_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = distorted_img.shape
    print(f"Distorted image loaded with shape: {height}x{width}")
    recovered_img = np.zeros((height, width), dtype=np.uint8)
    step_size = 2

    for y in range(0, height - 1, step_size):
        for x in range(0, width - 1, step_size):
            block = distorted_img[y:y+2, x:x+2]

            if block.shape == (2, 2):
                restored_block = block ^ filter_2x2
                recovered_img[y:y+2, x:x+2] = restored_block

    output_path = "recovered_portrait.png"
    cv2.imwrite(output_path, recovered_img)
    print(f"Recovered portrait saved as '{output_path}'")
    return output_path

def find_portrait_in_collage(recovered_portrait_path, collage_path):
    portrait = cv2.imread(recovered_portrait_path, cv2.IMREAD_GRAYSCALE)
    portrait_resized = cv2.resize(portrait, (100, 100))
    print(f"Resized portrait to 100x100")

    collage = cv2.imread(collage_path, cv2.IMREAD_GRAYSCALE)
    collage_height, collage_width = collage.shape
    print(f"Collage image loaded with shape: {collage_height}x{collage_width}")

    best_match = float('inf')
    best_x, best_y = -1, -1

    # Adjusted bounds to avoid out-of-range errors
    for y in range(collage_height - 99):
        for x in range(collage_width - 99):
            collage_section = collage[y:y+100, x:x+100]

            # Compute Sum of Absolute Differences (SAD)
            difference = np.sum(np.abs(collage_section - portrait_resized))

            if difference < best_match:
                best_match = difference
                best_x, best_y = x, y  # Store best match coordinates
    # Compute password
    password = math.floor(math.pi * (best_x + best_y))
    print(f"Portrait found at: ({best_x}, {best_y})")
    print(f"Password to unlock ZIP: {password}")
    return password

if __name__ == "__main__":
    pi_image_path = "pi_image.png"
    distorted_image_path = "art.png"
    collage_path = "collage.png"

    filter_2x2 = extract_filter(pi_image_path)
    if filter_2x2 is None:
        print("Exiting due to error in Step 1")
    else:
        recovered_portrait = recover_portrait(distorted_image_path, filter_2x2)
        if recovered_portrait is None:
            print("Exiting due to error in Step 2")
        else:
            password = find_portrait_in_collage(recovered_portrait, collage_path)
            if password is None:
                print("Exiting due to error in Step 3")
            else:
                print(f"Final Password: {password}")
