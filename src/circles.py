import numpy as np
import logging
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TWO_EURO_DIAMETER_MM = 25.75

class CircleHandler:
    """
    Find reference circles in an image.
    """

    def __init__(self, image_path:str|None = None, image:np.ndarray|None = None):
        """
        Initialize the processor with an image.

        :param image_path: Path to the input image file
        :param image: Input image as a NumPy array
        """

        if image_path is None and image is None:
            logger.error("Please provide either an image path or an image.")

        # Load the image if needed
        self.original_image = cv2.imread(image_path) if image_path else image

        # Convert the image to grayscale and 3-channel grayscale
        self.gray_image = cv2.cvtColor(self.original_image,
                                       cv2.COLOR_BGR2GRAY)
        self.gray_3channel = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        # Initialize attributes
        self.circles = None

    def detect_circles(self, params:dict|None = None) -> None:
        """
        Detect circles in the grayscale image using Hough Circle Transform.
        :param params: Optional parameter overrides for the Hough Circle Transform
        :return:
        """

        # Set default parameters if needed
        if params is None:
            params = dict(dp=1, minDist=50, param1=20, param2=100, minRadius=70, maxRadius=100)

        # Detect circles using Hough Circle Transform
        self.circles = cv2.HoughCircles(self.gray_image, cv2.HOUGH_GRADIENT, **params)

        return self.circles

    def draw_circles(self) -> np.ndarray|None:
        """
        Draw detected circles on the original image.

        :return: Image with circles drawn, or None if no circles detected
        """

        if self.circles is None:
            return None

        output_image = self.original_image.copy()
        circles = np.uint16(np.around(self.circles))

        # We should only have one circle
        if len(circles[0]) != 1:
            logger.error("Please provide an image with exactly one reference coin. The image appears to be ambiguous.")
            raise ValueError("Please provide an image with exactly one reference coin. The image appears to be ambiguous.")

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(output_image, center, radius, (0, 255, 0), 3)

        return output_image

    def calculate_mm_per_pixel(self, circle_diameter:float, coin_diameter_mm:float = TWO_EURO_DIAMETER_MM) -> float:
        """
        Calculate the number of millimeters per pixel in the image.

        :param circle_diameter: Diameter of the detected circle in pixels
        :param coin_diameter_mm: Diameter of the reference coin in mm
        :return: Millimeters per pixel in the image
        """

        # Get circle diameter in pixels
        circle_diameter = 2 * self.circles[0, 0, 2]

        return coin_diameter_mm / circle_diameter