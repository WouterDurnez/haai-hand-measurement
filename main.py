import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.markers import ArucoMarkerHandler
from src.viz import ImagePlotter
from src.circles import CircleHandler

class ArucoMarkerProcessor:
    """
    A class to process images with ArUco markers, perform perspective transformation,
    and detect circles.
    """

    def __init__(self, image_path: str):
        """
        Initialize the processor with an image.

        :param image_path: Path to the input image file
        """

        # Load the image and convert it to grayscale
        self.original_image = cv2.imread(image_path)
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
        self.circles = None

    def detect_markers(self) -> None:
        """
        Detect ArUco markers in the grayscale image.
        Order markers by their IDs.
        """
        # Detect markers
        self.corners, self.ids, _ = self.detector.detectMarkers(
            self.gray_image)

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
                cv2.putText(output_image, str(self.ids[i]), tuple(corner[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return output_image

    def align_image(self, height_over_width_ratio: float = 1612 / 2466,
                    width: int = 1000,
                    padding: int = 150) -> None:
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
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                           dtype='float32')

        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.aligned_image = cv2.warpPerspective(self.original_image, matrix,
                                                 (width + padding,
                                                  height + padding))

    def detect_circles(self, min_radius: int = 70, max_radius: int = 100) -> \
    Optional[List[Tuple[int, int, int]]]:
        """
        Detect circles in the aligned image using Hough Circle Transform.

        :param min_radius: Minimum circle radius to detect
        :param max_radius: Maximum circle radius to detect
        :return: List of detected circles or None
        """
        # Convert aligned image to grayscale
        aligned_gray = cv2.cvtColor(self.aligned_image, cv2.COLOR_BGR2GRAY)

        # Detect circles
        self.circles = cv2.HoughCircles(aligned_gray, cv2.HOUGH_GRADIENT, 1,
                                        20,
                                        param1=20, param2=100,
                                        minRadius=min_radius,
                                        maxRadius=max_radius)

        return self.circles

    def draw_circles(self) -> Optional[np.ndarray]:
        """
        Draw detected circles on the aligned image.

        :return: Image with circles drawn, or None if no circles detected
        """
        if self.circles is None:
            return None

        output_image = self.aligned_image.copy()
        circles = np.uint16(np.around(self.circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output_image, center, radius, (0, 255, 0), 3)

        return output_image

    def measure_circle_diameter(self, coin_diameter: float = 25.75) -> \
    Optional[Tuple[float, float]]:
        """
        Measure the diameter of detected circles and calculate pixel-to-mm ratio.

        :param coin_diameter: Diameter of the reference coin in mm
        :return: Tuple of circle diameter in pixels and mm-to-pixel ratio, or None
        """
        if self.circles is None:
            return None

        circle_diameter = 2 * self.circles[0, 0, 2]
        mm_in_pixels = circle_diameter / coin_diameter

        return circle_diameter, mm_in_pixels

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
        self.detect_circles()


def main():
    """
    Example usage of ArucoMarkerProcessor.
    """
    processor = ArucoMarkerProcessor("data/test_ashkan.png")

    # Process the image
    processor.process()

    # Visualize markers
    markers_image = processor.draw_markers()
    cv2.imshow('Detected Markers', markers_image)
    cv2.waitKey(0)

    # Visualize aligned image
    cv2.imshow('Aligned Image', processor.aligned_image)
    cv2.waitKey(0)

    # Visualize circles
    circles_image = processor.draw_circles()
    if circles_image is not None:
        cv2.imshow('Detected Circles', circles_image)
        cv2.waitKey(0)

    # Print measurement results
    measurement = processor.measure_circle_diameter()
    if measurement:
        circle_diameter, mm_in_pixels = measurement
        print(f'Diameter of the circle: {circle_diameter} pixels')
        print(f'1 mm in pixels: {mm_in_pixels}')

    cv2.destroyAllWindows()


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
    aligned_image = processor.align_image()
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
