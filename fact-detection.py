import os
import cv2
import face_recognition
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import time
import numpy as np

KNOWN_FACES_DIR = "known_faces"  # Folder where known face images are stored

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection")
        self.root.geometry("1000x600")  # Adjusted size for new layout
        self.root.configure(bg="#FFFFFF")

        # Variables
        self.video_capture = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.running = True
        self.recognition_mode = False

        # Configure Grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)

        # Video Frame (Left Side)
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Control Panel (Right Side)
        self.control_frame = tk.Frame(self.root, bg="#FFFFFF", width=300)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Title Label
        self.title_label = tk.Label(
            self.control_frame,
            text="Welcome to Face Detection",
            bg="#FFFFFF",
            fg="black",
            font=("Arial", 16, "bold"),
        )
        self.title_label.pack(pady=20)

        # Recognized Name Label (Large Display for Name)
        self.name_label = tk.Label(
            self.control_frame,
            text="Name: Unknown",
            bg="#FFFFFF",
            fg="purple",
            font=("Arial", 18, "bold"),
        )
        self.name_label.pack(pady=10)

        # Recognized Faces List
        self.recognized_faces_label = tk.Label(
            self.control_frame,
            text="Recognized Faces:",
            bg="#FFFFFF",
            fg="black",
            font=("Arial", 12, "bold"),
        )
        self.recognized_faces_label.pack(pady=10)

        self.recognized_faces_listbox = tk.Listbox(self.control_frame, font=("Arial", 10), height=10)
        self.recognized_faces_listbox.pack(pady=10, fill="both", expand=True)

        # Buttons for Control
        self.recognize_button = tk.Button(
            self.control_frame,
            text="Start Recognition",
            command=self.toggle_recognition,
            font=("Arial", 12),
            bg="#0078D7",
            fg="white",
        )
        self.recognize_button.pack(pady=10, fill="x")

        self.screenshot_button = tk.Button(
            self.control_frame,
            text="Take Screenshot",
            command=self.take_screenshot,
            font=("Arial", 12),
            bg="#0078D7",
            fg="white",
        )
        self.screenshot_button.pack(pady=10, fill="x")

        self.exit_button = tk.Button(
            self.control_frame,
            text="Exit",
            command=self.on_close,
            font=("Arial", 12),
            bg="#D70000",
            fg="white",
        )
        self.exit_button.pack(pady=10, fill="x")

        # Footer Label
        self.footer_label = tk.Label(
            self.control_frame,
            text="Advanced Security System © 2024",
            bg="#FFFFFF",
            fg="#888888",
            font=("Arial", 8),
        )
        self.footer_label.pack(side="bottom", pady=10)

        # Load known faces
        self.load_known_faces()

        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        self.update_frame()

    def load_known_faces(self):
        """Load known face encodings and names."""
        if not os.path.exists(KNOWN_FACES_DIR):
            messagebox.showerror("Error", f"Directory '{KNOWN_FACES_DIR}' not found!")
            return

        for file_name in os.listdir(KNOWN_FACES_DIR):
            file_path = os.path.join(KNOWN_FACES_DIR, file_name)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    name = os.path.splitext(file_name)[0]
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    def toggle_recognition(self):
        """Toggle recognition mode."""
        self.recognition_mode = not self.recognition_mode
        if self.recognition_mode:
            self.recognize_button.config(text="Stop Recognition")
        else:
            self.recognize_button.config(text="Start Recognition")

    def update_frame(self):
        """Update the video frame in real-time."""
        if not self.running:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_recognition()
            return

        start_time = time.time()
        frame = cv2.flip(frame, 1)  # Flip horizontally

        name = "Unknown"
        if self.recognition_mode:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                # Only use the best match (smallest distance)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                # Draw box and label on frame
                cv2.rectangle(frame, (left * 2, top * 2), (right * 2, bottom * 2), (0, 255, 0), 2)
                cv2.putText(frame, name, (left * 2, top * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update name label in the interface
        self.name_label.config(text=f"Name: {name}")

        # Convert frame to ImageTk
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def take_screenshot(self):
        """Take and save a screenshot."""
        if self.video_capture:
            ret, frame = self.video_capture.read()
            if ret:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
                )
                if file_path:
                    cv2.imwrite(file_path, frame)
                    messagebox.showinfo("Success", "Screenshot saved successfully!")

    def stop_recognition(self):
        """Stop video capture and close the app."""
        self.running = False
        if self.video_capture:
            self.video_capture.release()
        self.video_label.config(image="")

    def on_close(self):
        """Handle window close event."""
        self.stop_recognition()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
