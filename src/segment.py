import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Preprocessing
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return image, edges

# Step 2: Hand Segmentation
def segment_hand(edges):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour corresponds to the hand
    hand_contour = max(contours, key=cv2.contourArea)

    # Create a blank mask to draw the hand contour
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [hand_contour], -1, 255, thickness=cv2.FILLED)

    return mask, hand_contour

# Utility: Plot the results
def plot_results(original, edges, mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Segmented Hand")
    plt.imshow(mask, cmap='gray')

    plt.tight_layout()
    plt.show()

# Main Workflow
if __name__ == "__main__":
    # Path to your hand image
    image_path = '../data/test_ashkan_aligned.png'

    # Preprocess the image
    original, edges = preprocess_image(image_path)

    # Segment the hand
    mask, hand_contour = segment_hand(edges)

    # Plot the results
    plot_results(original, edges, mask)

    # Save the mask for further processing
    cv2.imwrite('segmented_hand.png', mask)
