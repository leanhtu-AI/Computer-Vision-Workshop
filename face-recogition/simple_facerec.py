import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(
            os.path.join(
                "face-recognition/images/",
                "*.*",
            )
        )
        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            print()
            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame, threshold=0.7):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing
        )
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        confidence_scores = []  # List to store the confidence scores

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=threshold
            )
            name = "Unknown"
            confidence = None  # Confidence score for the face

            # Check if a match was found in known_face_encodings
            if True in matches:
                # Find all indices where the match is True
                match_indices = [i for i, match in enumerate(matches) if match]
                distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                # Get the index of the closest match
                best_match_index = np.argmin(distances)

                # Check if the closest match is within the threshold
                if distances[best_match_index] <= threshold:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - distances[best_match_index]

            face_names.append(f"{name}: {confidence:.2f}")
            confidence_scores.append(confidence)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names, confidence_scores
