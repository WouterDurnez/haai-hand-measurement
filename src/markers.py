from pathlib import Path

import cv2

import numpy as np
import logging
from src.viz import ImagePlotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArucoMarkerHandler:
    """
    A class to process images with ArUco markers, perform perspective transformation,
    and detect circles.
    """

    def __init__(self, image_path: str = None, image: np.ndarray = None):
        """
        Initialize the processor with an image.

        :param image_path: Path to the input image file
        :param image: Input image as a NumPy array
        """

        if image_path is None and image is None or image_path and image:
            logger.error("Please provide either an image path or an image.")

        # Check if image path exists, if provided
        if image_path:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image path {image_path} does not exist.")
                raise FileNotFoundError(f"Image path {image_path} does not exist.")

        # Load the image if needed
        self.original_image = cv2.imread(image_path) if image_path else image

        # Convert the image to grayscale and 3-channel grayscale
        self.gray_image = cv2.cvtColor(self.original_image,
                                       cv2.COLOR_BGR2GRAY)
        self.gray_3channel = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        # Initialize ArUco marker detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict,
                                                self.parameters)

        # Initialize attributes
        self.corners = None
        self.ids = None
        self.centers = None
        self.aligned_image = None

    def detect_markers(self) -> None:
        """
        Detect ArUco markers in the grayscale image.
        Order markers by their IDs.
        """
        # Detect markers
        self.corners, self.ids, _ = self.detector.detectMarkers(
            self.gray_image)

        # Raise error if more or less than 4 markers are detected
        if len(self.ids) != 4:
            logger.error("Please provide an image with exactly 4 markers")
            raise ValueError("Please provide an image with exactly 4 markers")

        # Order the markers by ID (top-left, top-right, bottom-right, bottom-left)
        if self.ids is not None:

            # Flatten the IDs and sort them
            self.ids = self.ids.flatten()
            order = np.argsort(self.ids)
            self.ids = self.ids[order]

            # Reorder the corners based on the sorted IDs
            self.corners = [self.corners[i] for i in order]

            # Calculate marker centers for alignment
            self.centers = [self._get_marker_center(corner) for corner in
                            self.corners]

    def draw_markers(self) -> np.ndarray:
        """
        Draw detected markers on the grayscale image.

        :return: Image with markers drawn
        """
        output_image = self.gray_3channel.copy()

        if self.ids is not None:
            for i, corner in enumerate(self.corners):
                # Convert corner points to integers
                corner = corner[0].astype(np.int32)

                # Draw lines with increased thickness
                cv2.polylines(output_image, [corner], True, (0, 0, 255),
                              thickness=5)

                # Draw marker ID
                cv2.putText(output_image, str(self.ids[i]),self.centers[i].astype(np.int32),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)

        return output_image

    def align_image(self, height_over_width_ratio: float = 1612 / 2466,
                    width: int = 1000,
                    padding: int = 150) -> np.ndarray:
        """
        Perform perspective transformation to align the image.

        :param height_over_width_ratio: Ratio of height to width
        :param width: Desired width of aligned image
        :param padding: Padding added to the transformed image
        """
        # Calculate height based on width and ratio
        height = int(width / height_over_width_ratio)

        # Prepare source and destination points
        src_pts = np.array(self.centers, dtype='float32')
        dst_pts = np.array([[padding, padding],
                            [width + padding, padding],
                            [width + padding, height + padding],
                            [padding, height + padding]],
                           dtype='float32')

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.aligned_image = cv2.warpPerspective(self.original_image, matrix,
                                                 (width + 2 * padding,
                                                  height + 2 * padding))

        return self.aligned_image

    @staticmethod
    def _get_marker_center(corner: np.ndarray) -> np.ndarray:
        """
        Calculate the center of an ArUco marker.

        :param corner: Corner points of the marker
        :return: Center point of the marker
        """
        return np.mean(corner[0], axis=0)

    def process(self) -> None:
        """
        Perform complete image processing workflow:
        1. Detect ArUco markers
        2. Align image
        3. Detect circles
        """
        self.detect_markers()
        self.align_image()



if __name__ == '__main__':

    # Init the plotter
    plotter = ImagePlotter()

    # Process the image
    processor = ArucoMarkerHandler("data/test_ashkan.png")
    processor.process()

    # Draw detected markers
    markers_image = processor.draw_markers()
    plotter.plot(markers_image, title="Detected Markers")

    # Align the image
    aligned_image = processor.align_image()
    plotter.plot(aligned_image, title="Aligned Image")

    # Draw markers on the aligned image
    processor = ArucoMarkerHandler(image=aligned_image)
    processor.detect_markers()
    markers_image = processor.draw_markers()
    plotter.plot(markers_image, title="Detected Markers on Aligned Image")