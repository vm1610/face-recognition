# Face Recognition

## Project Overview

This project implements a face recognition system using OpenCV. It consists of three main scripts: one for training the face recognizer with images, one for recognizing faces in real-time using a webcam, and one for dataset preparation.

## Features

- **Face Training:** Trains the face recognizer with labeled images.
- **Real-Time Recognition:** Recognizes faces in real-time using a webcam.
- **Dataset Preparation:** Prepares and manages face image datasets.

## Tech Stack

- **Languages:** Python
- **Libraries:** 
  - `opencv-python` for image processing and face recognition
  - `numpy` for numerical operations
  - `PIL` (Pillow) for image handling

## Setup and Usage

1. **Install Dependencies:**
   - Ensure you have Python installed.
   - Install required libraries using pip:
     ```bash
     pip install opencv-python numpy pillow
     ```

2. **Prepare Data:**
   - Place your face images in a directory named `dataset`.
   - The image filenames should be in the format `ID_someName.jpg`, where `ID` is the identifier for each person.

3. **Scripts:**
   - **`face_recog_train.py`**: Train the face recognizer with images from the `dataset` directory and save the trained data to `trainingData.yml`.
     ```bash
     python face_recog_train.py
     ```
   - **`face_recog_test.py`**: Recognize faces in real-time using a webcam. The script loads the trained data and performs face recognition.
     ```bash
     python face_recog_test.py
     ```
   - **`face_recog_dataset.py`**: Script for managing and preparing the dataset. Ensure your images are correctly formatted and placed in the `dataset` directory before running the training script.

4. **Running the Scripts:**
   - Run the dataset preparation script if needed to organize your images.
   - Train the recognizer with the `face_recog_train.py` script.
   - Use the `face_recog_test.py` script to start real-time face recognition with your webcam. Press `q` to quit the video feed.

## Potential Enhancements

- **Enhanced Accuracy:** Improve recognition accuracy by experimenting with different algorithms or adding more training data.
- **Real-Time Performance:** Optimize performance for faster real-time face recognition.
- **User Management:** Implement a user management system to handle multiple users and their face data.
- **Face Detection Improvements:** Incorporate advanced face detection techniques to handle various lighting conditions and angles.
- **Web Interface:** Develop a web interface for easier interaction and management of face recognition tasks.

## License

This project is open-source and distributed under the terms of the MIT License.
