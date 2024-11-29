import matplotlib.pyplot as plt
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePlotter:
    """
    A class to plot images using Matplotlib.
    """

    @staticmethod
    def plot(image: np.ndarray, title: str = None) -> None:
        """
        Plot an image using Matplotlib.

        :param image: Image to plot
        :param title: Title of the plot
        """
        logger.info(f"Plotting image (title: {title})")
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()

