## ⚠️ Note

Before running the project, make sure all file paths in **`app.py`** are correctly configured, especially the path to the trained YOLO model (`best.pt`). If your model or any required files are stored in a different location, update the corresponding paths before launching the application. Incorrect file paths may result in errors such as `FileNotFoundError` or the model failing to load.






# Road-Damage-Detection-Using-YOLO11
# 🛣️ Road Damage Detection Using YOLO11

## 📌 Project Overview

This project is an AI-powered Road Damage Detection system developed using the **YOLO11** object detection model and **Streamlit**. The application automatically detects different types of road surface damage from uploaded images, helping support road inspection and maintenance.

The model detects the following road damage classes:

- 🕳️ Pothole
- 🪨 Crack
- ⭕ Manhole

The application performs real-time object detection, displays bounding boxes with confidence scores, reports the total number of detected objects, and allows users to download the annotated prediction image.

> **Note:** Before running the project, ensure that all file paths in **`app.py`** are correctly configured, especially the path to the trained YOLO model (`best.pt`). By default, the application expects the model to be located in the **`model/`** directory. If the model or any required files are stored elsewhere, update the corresponding paths before launching the application. Incorrect file paths may result in errors such as **`FileNotFoundError`** or the model failing to load successfully.

---

## 🚀 Features

- Upload road images through an easy-to-use Streamlit interface.
- Automatically detect potholes, cracks, and manholes.
- Display bounding boxes with confidence scores.
- Show the number of detected objects for each class.
- Download the annotated detection image.
- Fast inference using the trained YOLO11 model.
