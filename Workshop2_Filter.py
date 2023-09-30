import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load the image
image = cv2.imread(
    "C:/Users/FPTSHOP/OneDrive/Documents/AI1706/Summer2023/CPV301/code/lenna.jpg"
)


class ImageFilterApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Filter App")

        # Create widgets
        self.label1 = tk.Label(master, text="Choose an image to filter:")
        self.button1 = tk.Button(master, text="Select Image", command=self.select_image)
        self.label2 = tk.Label(master, text="Choose filter type:")
        self.filter_var = tk.StringVar()
        self.filter_var.set("mean")
        self.radio_mean = tk.Radiobutton(
            master, text="Mean", variable=self.filter_var, value="mean"
        )
        self.radio_median = tk.Radiobutton(
            master, text="Median", variable=self.filter_var, value="median"
        )
        self.radio_gaussian = tk.Radiobutton(
            master, text="Gaussian", variable=self.filter_var, value="gaussian"
        )
        self.kernel_label = tk.Label(master, text="Kernel size:")
        self.kernel_entry = tk.Entry(master)
        self.sigma_label = tk.Label(master, text="Sigma value:")
        self.sigma_entry = tk.Entry(master)
        self.filter_button = tk.Button(
            master, text="Apply Filter", command=self.apply_filter
        )
        self.original_image_label = tk.Label(master, text="Original Image:")
        self.filtered_image_label = tk.Label(master, text="Filtered Image:")

        # Layout widgets
        self.label1.grid(row=0, column=0, sticky="w")
        self.button1.grid(row=0, column=1)
        self.label2.grid(row=1, column=0, sticky="w")
        self.radio_mean.grid(row=1, column=1, sticky="w")
        self.radio_median.grid(row=2, column=1, sticky="w")
        self.radio_gaussian.grid(row=3, column=1, sticky="w")
        self.kernel_label.grid(row=4, column=0, sticky="w")
        self.kernel_entry.grid(row=4, column=1)
        self.sigma_label.grid(row=5, column=0, sticky="w")
        self.sigma_entry.grid(row=5, column=1)
        self.filter_button.grid(row=6, column=0)
        self.original_image_label.grid(row=7, column=0)
        self.filtered_image_label.grid(row=7, column=1)

    def select_image(self):
        # Open file dialog to choose image
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.img = cv2.imread(file_path)
            # Display original image
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_tk = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_tk)
            self.original_image_label.configure(image=img_tk)
            self.original_image_label.image = img_tk

    def apply_filter(self):
        # Get filter type and kernel size / sigma value from user
        filter_type = self.filter_var.get()
        kernel_size = int(self.kernel_entry.get())
        sigma = float(self.sigma_entry.get())

        # Apply chosen filter
        if filter_type == "mean":
            img_filtered = cv2.blur(self.img, (kernel_size, kernel_size))
        elif filter_type == "median":
            img_filtered = cv2.medianBlur(self.img, kernel_size)
        elif filter_type == "gaussian":
            img_filtered = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)

        # Display filtered image
        if hasattr(self, "filtered_image_label"):
            self.filtered_image_label.destroy()
        img_rgb = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)
        img_tk = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_tk)
        self.filtered_image_label = tk.Label(self.master, image=img_tk)
        self.filtered_image_label.image = img_tk
        self.filtered_image_label.grid(row=7, column=1)


def color_balance(image, b_ratio, g_ratio, r_ratio):
    # Split color channels
    b, g, r = cv2.split(image)

    # Apply ratios to each channel
    b = np.clip(b * b_ratio, 0, 255).astype(np.uint8)
    g = np.clip(g * g_ratio, 0, 255).astype(np.uint8)
    r = np.clip(r * r_ratio, 0, 255).astype(np.uint8)

    # Merge the balanced channels
    img_balanced = cv2.merge([b, g, r])

    return img_balanced


def update_color_balance(val):
    # Get slider values
    b_ratio = slider_b.val / 255.0
    g_ratio = slider_g.val / 255.0
    r_ratio = slider_r.val / 255.0
    # Balance the colors
    balanced_image = color_balance(image, b_ratio, g_ratio, r_ratio)
    # Update the image display
    ax_image.imshow(cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()


def reset_view(event):
    # Reset the view of the plot
    ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Reset the slider values
    slider_b.reset()
    slider_g.reset()
    slider_r.reset()
    # Redraw the plot
    fig.canvas.draw_idle()


fig, ax_image = plt.subplots(1, 1, figsize=(5, 4))
ax_image.axis("off")
ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# axes([left, bottom, width, height])
slider_b_ax = plt.axes([0.25, 0.2, 0.65, 0.03])
slider_g_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_r_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
reset_button_ax = plt.axes([0.65, 0.015, 0.25, 0.075])

# create the picture start point color
x, y = 100, 50
bgr = image[y, x]
slider_b_val = bgr[0]
slider_g_val = bgr[1]
slider_r_val = bgr[2]

# Create the sliders with the converted initial values
slider_b = Slider(slider_b_ax, "Blue", 0, 255, valinit=slider_b_val)
slider_g = Slider(slider_g_ax, "Green", 0, 255, valinit=slider_g_val)
slider_r = Slider(slider_r_ax, "Red", 0, 255, valinit=slider_r_val)
reset_button = Button(reset_button_ax, "Reset the Image")

slider_b.on_changed(update_color_balance)
slider_g.on_changed(update_color_balance)
slider_r.on_changed(update_color_balance)
reset_button.on_clicked(reset_view)

# pos of picture
plt.subplots_adjust(left=0.1, bottom=0.3, right=0.925, top=0.9, wspace=0.2)
plt.show()

root = tk.Tk()
app = ImageFilterApp(root)
root.mainloop()
