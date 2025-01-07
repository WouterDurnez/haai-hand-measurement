# ArUco Marker and Circle Detection

This project processes images with ArUco markers, performs perspective transformation, and detects circles for referencing. It uses OpenCV for image processing and Loguru for logging.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Loguru

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Make sure you have `uv` installed (do it in a virtual environment):
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install uv
    ```

3. Install the required packages using `uv`:
    ```sh
    uv install
    ```

## Usage

1. Place your image in the `data` directory (or use the sample image provided). It needs to be `png` format.

2. Update the image path in `main_reference.py`:
    ```python
    processor = ArucoMarkerHandler("data/your_image.png")
    ```

3. Run the script to see the functionalities demonstrated:
    ```sh
    python main_reference.py
    ```

## Project Structure

- `main_reference.py`: Main script to process the image.
- `src/reference/markers.py`: Contains the `ArucoMarkerHandler` class for detecting and processing ArUco markers.
- `src/reference/viz.py`: Contains the `ImagePlotter` class for visualizing images.
- `src/reference/circles.py`: Contains the `CircleHandler` class for detecting circles.
