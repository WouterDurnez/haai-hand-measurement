"""
This script is used to test the reference image processing pipeline.

The pipeline includes:
1. Detecting ArUco markers in the image
2. Aligning the image based on the detected markers
3. Detecting circles in the aligned image
4. Calculating the mm per pixel based on the detected circle diameter

Next steps:
- Segment the hand from the aligned image (using an NN or simple thresholding, masking out the markers, logo and coin)
- Calculate keypoint-based features for the hand (e.g., finger lengths, palm width, etc.)
- Train a model to predict size based on these features
"""

from src.reference.markers import ArucoMarkerHandler
from src.reference.viz import ImagePlotter
from src.reference.circles import CircleHandler


from loguru import logger

def main():

    # Init the plotter
    plotter = ImagePlotter()

    # Process the image
    processor = ArucoMarkerHandler("data/test_ashkan.png")
    processor.process()

    # Draw detected markers
    markers_image = processor.draw_markers(mask=False)
    plotter.plot(markers_image, title="Detected Markers")

    # Align the image
    aligned_image = processor.align_image(save=True)
    plotter.plot(aligned_image, title="Aligned Image")

    # Draw markers on the aligned image
    processor = ArucoMarkerHandler(image=aligned_image)
    processor.detect_markers()
    aligned_markers_image = processor.draw_markers()
    plotter.plot(aligned_markers_image, title="Detected Markers on Aligned Image")

    # Detect circles
    circle_processor = CircleHandler(image=aligned_markers_image)
    circle_processor.detect_circles()
    circles_image = circle_processor.draw_circles()
    plotter.plot(circles_image, title="Detected Circles")

    # Get the circle diameter and calculate mm per pixel
    mm_per_pixel = circle_processor.calculate_mm_per_pixel()
    logger.info(f"Pixels per mm: {1/mm_per_pixel}")

if __name__ == "__main__":
    main()