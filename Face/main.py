import cv2
import os
from simple_facerec import SimpleFacerec

# Check if "Images/" folder exists
images_folder = "Images/"
if os.path.exists(images_folder):
    # Load encoding images
    sfr = SimpleFacerec()
    sfr.load_encoding_images(images_folder)

# Code to capture video from webcam (assuming webcam index 2)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()

    if os.path.exists(images_folder):
        # Detect faces and names if encodings were loaded
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Draw red background rectangle for the text
            text = name
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            font_thickness = 2

            # Get text size and baseline
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate coordinates for text and background rectangle
            text_x = x2
            text_y = y1 - 10
            text_bg_color = (0, 0, 255)  # Red color for background
            padding = 5  # Padding around the text

            # Draw filled background rectangle for the text
            cv2.rectangle(frame, (text_x - padding, text_y - text_size[1] - padding),
                          (text_x + text_size[0] + padding, text_y + padding),
                          text_bg_color, cv2.FILLED)

            # Put text on the frame
            cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                        (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 0), 4)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit loop if Esc key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
