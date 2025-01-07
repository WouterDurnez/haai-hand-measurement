import uuid
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy.ma.core import absolute

from src.reference.viz import ImagePlotter


class ArucoMarkerHandler:
    """
    A class to process images with ArUco markers, perform perspective transformation,
    and detect circles.
    """

    def __init__(self, image_path: str|Path = None, image: np.ndarray = None):
        """
        Initialize the processor with an image.

        :param image_path: Path to the input image file
        :param image: Input image as a NumPy array
        """

        if image_path is None and image is None or image_path and image:
            logger.error("Please provide either an image path or an image.")

        # Check if image path exists, if provided
        if image_path:
            self.image_path = Path(image_path) if isinstance(image_path, str) else image_path
            if not self.image_path.exists():
                logger.error(f"Image path {self.image_path} does not exist.")
                raise FileNotFoundError(f"Image path {self.image_path} does not exist.")

        # Load the image if needed
        self.original_image = cv2.imread(self.image_path) if hasattr(self, "image_path") else image

        # Get or set the image name and directory
        self.image_name = self.image_path.stem if hasattr(self, 'image_path') else not f"image_{uuid.uuid4()}"
        self.image_dir = self.image_path.parent if hasattr(self, 'image_path') else Path(".")

        # Convert the image to grayscale and 3-channel grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.gray_3channel = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        # Initialize ArUco marker detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

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
        self.corners, self.ids, _ = self.detector.detectMarkers(self.gray_image)

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
            self.centers = [self._get_marker_center(corner) for corner in self.corners]

    def draw_markers(self, outline: bool = True, text:bool = True, mask:bool = True) -> np.ndarray:
        """
        Draw detected markers on the grayscale image.

        :param outline: Draw marker outlines
        :param text: Draw marker IDs
        :param mask: Fill the markers with white color

        :return: Image with markers drawn
        """
        output_image = self.gray_3channel.copy()
        color = (200,200,200)

        if self.ids is not None:
            for i, corner in enumerate(self.corners):
                # Convert corner points to integers
                corner = corner[0].astype(np.int32)

                # Draw lines with increased thickness
                if outline:
                    cv2.polylines(output_image, [corner], True, color, thickness=5)

                # Fill the marker with white color
                if mask:
                    cv2.fillPoly(output_image, [corner], (255, 255, 255))

                # Draw marker ID
                if text:
                    cv2.putText(
                    output_image,
                    str(self.ids[i]),
                    self.centers[i].astype(np.int32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    color,
                    10,
                )

        self.annotated_image = output_image

        return output_image

    def align_image(
        self,
        height_over_width_ratio: float = 1612 / 2466,
        width: int = 1000,
        padding: int = 150,
            include_annotations: bool = True,
        save: bool = False,
    ) -> np.ndarray:
        """
        Perform perspective transformation to align the image.

        :param height_over_width_ratio: Ratio of height to width
        :param width: Desired width of aligned image
        :param padding: Padding added to the transformed image
        :param include_annotations: Include annotations in the aligned image
        :param save: Save the aligned image to disk
        """
        # Calculate height based on width and ratio
        height = int(width / height_over_width_ratio)

        # Prepare source and destination points
        src_pts = np.array(self.centers, dtype="float32")
        dst_pts = np.array(
            [
                [padding, padding],
                [width + padding, padding],
                [width + padding, height + padding],
                [padding, height + padding],
            ],
            dtype="float32",
        )

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if include_annotations:
            if not hasattr(self, 'annotated_image'):
                self.draw_markers()
            image_to_align = self.annotated_image
        else:
            image_to_align = self.gray_3channel

        self.aligned_image = cv2.warpPerspective(
            image_to_align, matrix, (width + 2 * padding, height + 2 * padding)
        )

        # Save the aligned image if needed
        if save:
            file_name_reference = self.image_name + "_aligned.png"
            file_path_reference = self.image_dir / file_name_reference
            cv2.imwrite(str(file_path_reference), self.aligned_image)

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


if __name__ == "__main__":

    # Init the plotter
    plotter = ImagePlotter()

    # Get current directory
    current_dir = Path(__file__).resolve().parent

    # Go two directories up to get to the project root
    project_dir = current_dir.parent.parent

    # Go to the data directory
    data_dir = project_dir / "data"

    # Path to the input image
    image_path = data_dir / "test_ashkan.png"

    # Process the image
    processor = ArucoMarkerHandler(image_path)
    processor.process()

    # Draw detected markers
    markers_image = processor.draw_markers(mask=False)
    plotter.plot(markers_image, title="Detected Markers")

    # Align the image
    aligned_image = processor.align_image()
    plotter.plot(aligned_image, title="Aligned Image")

    # Draw markers on the aligned image
    processor = ArucoMarkerHandler(image=aligned_image)
    processor.detect_markers()
    markers_image = processor.draw_markers()
    plotter.plot(markers_image, title="Detected Markers on Aligned Image")
