import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image


selected_image_path = ""


def select_image():
    global selected_image_path

    # Open a file dialog to choose an image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )

    if file_path:
        selected_image_path = file_path

        # Read the selected image
        img = Image.open(file_path)

        # Resize the image for display
        resized_img = img.resize((800, 500))

        # Create a Tkinter PhotoImage from the resized image
        tk_img = ImageTk.PhotoImage(resized_img)

        # Update the label to show the image
        image_label.configure(image=tk_img)
        image_label.image = tk_img


def detect_faces():
    global selected_image_path

    if selected_image_path:
        # Load the face cascade classifier
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Read the selected image
        img = cv2.imread(selected_image_path, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangles around the detected faces
        for x, y, w, h in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a PIL image from the RGB image array
        pil_img = Image.fromarray(img_rgb)

        # Resize the image for display
        resized_img = pil_img.resize((800, 500))

        # Create a Tkinter PhotoImage from the resized image
        tk_img = ImageTk.PhotoImage(resized_img)

        # Update the label to show the image
        image_label.configure(image=tk_img)
        image_label.image = tk_img


# Create a Tkinter window
root = tk.Tk()
root.title("Face Detection")

# Retrieve the screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position
window_width = 800
window_height = 600
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

# Set the window position
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=20)

# Create a button to trigger face detection
detect_button = tk.Button(root, text="Detect Faces", command=detect_faces)
detect_button.pack(pady=10)

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Start the GUI event loop
root.mainloop()
