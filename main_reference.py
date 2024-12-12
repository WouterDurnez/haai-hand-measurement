import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from src.reference.markers import ArucoMarkerHandler
from src.reference.viz import ImagePlotter
from src.reference.circles import CircleHandler


from loguru import logger



if __name__ == "__main__":

    # Init the plotter
    plotter = ImagePlotter()

    # Process the image
    processor = ArucoMarkerHandler("data/test_ashkan.png")
    processor.process()

    # Draw detected markers
    markers_image = processor.draw_markers()
    plotter.plot(markers_image, title="Detected Markers")

    # Align the image
    aligned_image = processor.align_image(save=True)
    plotter.plot(aligned_image, title="Aligned Image")

    # Draw markers on the aligned image
    # processor = ArucoMarkerHandler(image=aligned_image)
    # processor.detect_markers()
    # aligned_markers_image = processor.draw_markers()
    # plotter.plot(aligned_markers_image, title="Detected Markers on Aligned Image")

    # Detect circles
    circle_processor = CircleHandler(image=aligned_image)
    circle_processor.detect_circles()
    circles_image = circle_processor.draw_circles()
    plotter.plot(circles_image, title="Detected Circles")

    # Get the circle diameter and calculate mm per pixel
    mm_per_pixel = circle_processor.calculate_mm_per_pixel()
