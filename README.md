# Drowsiness Detection System

A computer vision and deep learning project that processes a video file, detects faces and eyes, and classifies detected eye images as **Awake** or **Drowsy**.

The project uses a Convolutional Neural Network (CNN) trained on open-eye and closed-eye images, together with OpenCV Haar Cascade classifiers for face and eye detection.

## Features

- Reads and processes a local video file.
- Detects faces and eyes using OpenCV Haar Cascade classifiers.
- Uses a trained CNN to classify eye images.
- Displays **Awake** or **Drowsy** labels on the video.
- Saves face-region frames classified as Drowsy in the `data/` folder.
- Includes scripts for dataset preparation, model training, and model conversion.

> **Input mode:** The current backend uses `cv2.VideoCapture("vid1.mp4")`. It processes a video file and does not use a live webcam.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- h5py

## Project Structure

```text
Drowsiness-Detection-System/
├── backend.py
├── train.py
├── split.py
├── compress_model.py
├── drowsiness.h5
├── face.xml
├── eye.xml
├── vid1.mp4              # Add your input video here
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── .gitkeep
```

## Setup

### 1. Get the project files

Download or clone this repository:

```bash
git clone <your-repository-url>
cd Drowsiness-Detection-System
```

### 2. Add the input video

Place your video in the project root folder and name it exactly:

```text
vid1.mp4
```

The video is not bundled in this ZIP because it was not available among the files I received. Add your own copy before running the backend.

### 3. Install dependencies

It is recommended to create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Run the Project

Make sure these files are present in the project root:

- `backend.py`
- `drowsiness.h5`
- `face.xml`
- `eye.xml`
- `vid1.mp4`

Then run:

```bash
python backend.py
```

The OpenCV window displays the processed video. Press **x** to stop processing.

Drowsy-classified face-region images are saved in the `data/` directory.

## Model Architecture

The saved `drowsiness.h5` model is a Sequential CNN with:

- Input shape: `150 × 150 × 3`
- Three convolutional layers with 32 filters and ReLU activation
- Max-pooling layers
- Flatten layer
- Dense layer with 100 units and ReLU activation
- One sigmoid output unit for binary classification

The training script compiles the model with binary cross-entropy loss and the Adam optimizer, and trains it for 10 epochs.

## Training

The training script expects this dataset folder layout:

```text
train/
├── Open-Eyes/
└── Close-Eyes/

test/
├── Open-Eyes/
└── Close-Eyes/
```

Run the dataset preparation script only after placing the image folders in the expected locations:

```bash
python split.py
```

Then train the model:

```bash
python train.py
```

The training script saves the resulting model as `drowsiness.h5`. This will overwrite the existing model file, so keep a backup if you want to retain the supplied trained model.

To convert the H5 model to Keras format:

```bash
python compress_model.py
```

## Dataset

The dataset is not included in this repository. Add the dataset source and its license information if you share or redistribute it.

## Limitations and Safety

- This is an educational/demo project, not a certified driver-monitoring or vehicle safety system.
- Detection quality can vary with lighting, face angle, video quality, and eye visibility.
- Do not rely on this project as the sole method of preventing drowsy-driving incidents.
- The included Haar Cascade XML files contain license and attribution notices. Preserve those notices and comply with their redistribution terms.

## Future Improvements

- Add live webcam input.
- Add audio alerts when drowsiness is detected.
- Evaluate performance on a separate test dataset and report measured metrics.
- Improve detection under different lighting conditions and face angles.

## Author

**Shivam Pandya**
