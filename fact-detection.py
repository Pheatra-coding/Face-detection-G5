import cv2
import face_recognition

# Initialize known face encodings and their details
known_face_encodings = []
known_faces = {
    "Horth": {"Age": 19},
    "Peaktra": {"Age": 19},
    "Samoun Suon": {"Age": 19},
    "Dy": {"Age": 19},
    "Leak": {"Age": 19},
    "Bopha": {"Age": 19},
    "Kin": {"Age": 19},
    "Knrork": {"Age": 19},
    "Lin": {"Age": 19},
    "Ny": {"Age": 19}
}

# Load and encode known faces
try:
    for i in range(1, 11):
        person_image = face_recognition.load_image_file(f"person{i}.jpg")
        person_encoding = face_recognition.face_encodings(person_image)[0]
        known_face_encodings.append(person_encoding)
except IndexError:
    print("Error: One or more images did not contain any faces.")
    exit()

known_face_names = list(known_faces.keys())  # Get the names from the dictionary

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Flip the frame horizontally to avoid the mirror effect
    frame = cv2.flip(frame, 1)

    # Resize the frame to make the face recognition faster
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Reduce size by a quarter

    # Convert the frame from BGR to RGB (face_recognition works on RGB)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Display only the name
        display_text = name

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face with a green color
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
 