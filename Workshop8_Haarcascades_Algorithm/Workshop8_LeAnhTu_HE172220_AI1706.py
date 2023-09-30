import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import warnings

warnings.filterwarnings("ignore")


def detect_and_recognize_faces(image_path):
    dataset_folder = "Workshop8_TuLA_HE172220_AI1706/dataset/"
    names = []
    for folder in os.listdir(dataset_folder):
        for name in os.listdir(os.path.join(dataset_folder, folder))[
            :70
        ]:  # limit only 70 faces per class
            names.append(folder)

    labels = np.unique(names)

    # Load the pre-trained Eigenface model
    eigenface_model_path = "eigen_model.yml"
    eigenface_model = cv2.face.EigenFaceRecognizer_create()
    eigenface_model.read(eigenface_model_path)

    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        "Workshop8_TuLA_HE172220_AI1706/haarcascades/haarcascade_frontalface_default.xml"
    )

    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Process each detected face
    for x, y, w, h in faces:
        # Extract the face ROI
        face_roi = gray_image[y : y + h, x : x + w]

        # Resize the face ROI to match the training data size
        resized_face_roi = cv2.resize(face_roi, (100, 100))

        # Perform face recognition using the Eigenface model
        idx, confidence = eigenface_model.predict(resized_face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Create label text
        label_text = f"{labels[idx]}"

        # Create label text
        # label_text = f"Name: {labels[idx]}"
        # Text coordinates
        text_x = x - 30
        text_y = y - 20

        # Draw label above rectangle
        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Convert the image from BGR to RGB for displaying with tkinter
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(pil_image)

    # Display the image on the tkinter canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    canvas.image = image_tk  # Save a reference to prevent garbage collection
    canvas_width = 800
    canvas_height = 600
    image_width, image_height = pil_image.size
    resize_factor = min(canvas_width / image_width, canvas_height / image_height)

    # Calculate the new image dimensions
    new_image_width = int(image_width * resize_factor)
    new_image_height = int(image_height * resize_factor)

    # Resize the image
    resized_pil_image = pil_image.resize(
        (new_image_width, new_image_height), Image.ANTIALIAS
    )
    image_tk = ImageTk.PhotoImage(resized_pil_image)

    # Display the resized image on the tkinter canvas
    canvas.create_image(
        (canvas_width - new_image_width) // 2,
        (canvas_height - new_image_height) // 2,
        anchor=tk.NW,
        image=image_tk,
    )
    canvas.image = image_tk  # Save a reference to prevent garbage collection


# GUI function
def browse_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        detect_and_recognize_faces(file_path)


# Create the main GUI window
root = tk.Tk()
root.title("Face Recognition GUI")

# Create a button to browse for an image
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

# Create a canvas to display the image
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# Run the Tkinter main loop
root.mainloop()
