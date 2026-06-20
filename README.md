# Emojinator 🖐️➡️😊

Emojinator is an end-to-end computer vision and deep learning application that recognizes hand gestures in real-time via a webcam and overlays corresponding emojis directly onto the screen. 

The project covers the entire pipeline: custom dataset generation, convolutional neural network (CNN) training, and real-time computer vision inference.

## Features
* **Custom Data Capture:** Built-in tool to record and preprocess your own hand gestures using OpenCV.
* **Background Subtraction:** Uses HSV color filtering, background thresholding, and morphological operations to clean up and isolate hand contours.
* **Deep Learning Classifier:** A Keras/TensorFlow Sequential CNN trained to recognize distinct gesture classes.
* **Real-Time AR Overlay:** Dynamically overlays a transparent emoji onto your video feed based on the recognized gesture.
