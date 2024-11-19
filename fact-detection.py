import cv2
import face_recognition

# Initialize known face encodings and their details
known_face_encodings = []
known_faces = {
    "Horth": {"Age": 19},
    "Peaktra": {"Age": 19},
    "Samoun Suon": {"Age":19},
    "Dy": {"Age": 19},
    "leak": {"Age": 19},
    "Bopha": {"Age":19},
    "Kin": {"Age":19},
    "Knrork": {"Age":19},
    "lin": {"Age":19},
    "Ny": {"Age":19}
}

# Load and encode known faces
try:
    person1_image = face_recognition.load_image_file("person1.jpg")
    person1_encoding = face_recognition.face_encodings(person1_image)[0]
    known_face_encodings.append(person1_encoding)
    
    person2_image = face_recognition.load_image_file("person2.jpg")
    person2_encoding = face_recognition.face_encodings(person2_image)[0]
    known_face_encodings.append(person2_encoding)
    
    person3_image = face_recognition.load_image_file("person3.jpg")
    person3_encoding = face_recognition.face_encodings(person3_image)[0]
    known_face_encodings.append(person3_encoding)

    person4_image = face_recognition.load_image_file("person4.jpg")
    person4_encoding = face_recognition.face_encodings(person4_image)[0]
    known_face_encodings.append(person4_encoding)
    
    person5_image = face_recognition.load_image_file("person5.jpg")
    person5_encoding = face_recognition.face_encodings(person5_image)[0]
    known_face_encodings.append(person5_encoding)
    
    person6_image = face_recognition.load_image_file("person6.jpg")
    person6_encoding = face_recognition.face_encodings(person6_image)[0]
    known_face_encodings.append(person6_encoding)

    person7_image = face_recognition.load_image_file("person7.jpg")
    person7_encoding = face_recognition.face_encodings(person7_image)[0]
    known_face_encodings.append(person7_encoding)

    person8_image = face_recognition.load_image_file("person8.jpg")
    person8_encoding = face_recognition.face_encodings(person8_image)[0]
    known_face_encodings.append(person8_encoding)

    person9_image = face_recognition.load_image_file("person9.jpg")
    person9_encoding = face_recognition.face_encodings(person9_image)[0]
    known_face_encodings.append(person9_encoding)

    person10_image = face_recognition.load_image_file("person10.jpg")
    person10_encoding = face_recognition.face_encodings(person10_image)[0]
    known_face_encodings.append(person10_encoding)

except IndexError:
    print("Error: One or more images did not contain any faces.")
    exit()

known_face_names = list(known_faces.keys())  # Get the names from the dictionary

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

frame_count = 0  # Initialize frame counter

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Flip the frame horizontally to avoid the mirror effect
    frame = cv2.flip(frame, 1)

    # Resize the frame to make the face recognition faster
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce size by half

    # Skip frames to make the video smoother (process every 5th frame)
    frame_count += 1
    if frame_count % 5 != 0:  # Process every 5th frame
        continue

    # Convert the frame from BGR to RGB (face_recognition works on RGB)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

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

        # Draw a rectangle around the face with a green color
        cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (0, 255, 0), 2)  # Adjusting for resized frame
        cv2.putText(frame, display_text, (left * 2, top * 2 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)  # Adjusting for resized frame

    # Display the result
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()   