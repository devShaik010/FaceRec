import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path, pkl_folder='encodings'):
        """
        Load encoding images from path and optionally save them as pickle files.
        
        :param images_path: Path to the folder containing images.
        :param pkl_folder: Folder name to save pickle files (default: 'encodings').
        """
        # Create folder if it doesn't exist
        os.makedirs(pkl_folder, exist_ok=True)

        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("Searching encodings...")

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            # Check if pickle file already exists
            pkl_file = os.path.join(pkl_folder, f"{filename}.pkl")
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    img_encoding = pickle.load(f)
                print(f"Encoding found for {filename}")
            else:
                print(f"Encoding image {filename}")
                # Get encoding
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                # Save encoding as pickle file
                with open(pkl_file, 'wb') as f:
                    pickle.dump(img_encoding, f)
            
            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
