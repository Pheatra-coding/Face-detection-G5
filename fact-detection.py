import os
import cv2
import face_recognition
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Constants
KNOWN_FACES_DIR = "known_faces"
EXCEL_LOG_FILE = "face_recognition_log.xlsx"
CONFIDENCE_THRESHOLD = 0.5  # Lower value means stricter matching

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Security Cam")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2E2E2E")

        # Variables
        self.video_capture = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.priority_faces = ["John", "Alice"]  # Specific names to highlight
        self.running = True
        self.recognition_mode = False
        self.fps_times = []
        self.last_logged_name = None
        self.last_logged_time = None
        self.recognized_faces = set()

        # Configure Grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)

        # Video Frame
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Control Panel
        self.control_frame = tk.Frame(self.root, bg="#1E1E1E", width=300)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Control Elements
        self.title_label = tk.Label(
            self.control_frame,
            text="Face Recognition System",
            bg="#1E1E1E",
            fg="white",
            font=("Arial", 16, "bold"),
        )
        self.title_label.pack(pady=20)

        self.message_label = tk.Label(
            self.control_frame,
            text="Click Recognize to start detection.",
            bg="#1E1E1E",
            fg="#00FF00",
            font=("Arial", 12),
        )
        self.message_label.pack(pady=10)

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

        self.recognized_faces_label = tk.Label(
            self.control_frame,
            text="Recognized Faces:",
            bg="#1E1E1E",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.recognized_faces_label.pack(pady=10)

        self.recognized_faces_listbox = tk.Listbox(self.control_frame, font=("Arial", 10), height=10)
        self.recognized_faces_listbox.pack(pady=10, fill="both", expand=True)

        # Footer
        self.footer_label = tk.Label(
            self.control_frame,
            text="Advanced Security System © 2024",
            bg="#1E1E1E",
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
            self.message_label.config(text="Recognition mode: ON", fg="#00FF00")
            self.recognize_button.config(text="Stop Recognition")
        else:
            self.message_label.config(text="Recognition mode: OFF", fg="#FF0000")
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

        if self.recognition_mode:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                name = "Unknown"
                box_color = (0, 0, 255)  # Default to red for unknown faces
                text_color = (0, 0, 255)

                if distances[best_match_index] < CONFIDENCE_THRESHOLD:
                    name = self.known_face_names[best_match_index]
                    if name in self.priority_faces:
                        box_color = (0, 255, 255)  # Yellow for priority faces
                        text_color = (0, 255, 255)
                    else:
                        box_color = (0, 255, 0)  # Green for recognized faces
                        text_color = (0, 255, 0)

                    if name not in self.recognized_faces:
                        self.recognized_faces.add(name)
                        self.recognized_faces_listbox.insert(tk.END, name)
                        self.log_recognition(name)

                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.putText(frame, f"{name} ({distances[best_match_index]:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

        # FPS Calculation
        self.fps_times.append(time.time() - start_time)
        if len(self.fps_times) > 10:  # Average over 10 frames
            self.fps_times.pop(0)
        fps = int(1 / (np.mean(self.fps_times) or 1))
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert to ImageTk
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def log_recognition(self, name):
        """Log recognized face with timestamp to the log file only if it's a new recognition."""
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Initialize dataframe if file doesn't exist
        if not os.path.exists(EXCEL_LOG_FILE):
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
        else:
            df = pd.read_excel(EXCEL_LOG_FILE)

        # Append the new recognition only if it's a new entry
        df = pd.concat([df, pd.DataFrame({"Name": [name], "Date": [current_date], "Time": [current_time]})], ignore_index=True)
        df.to_excel(EXCEL_LOG_FILE, index=False)

    def take_screenshot(self):
        """Take a screenshot of the current frame."""
        ret, frame = self.video_capture.read()
        if ret:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            screenshot_file = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(screenshot_file, frame)
            messagebox.showinfo("Screenshot", f"Screenshot saved as {screenshot_file}")

    def stop_recognition(self):
        """Stop recognition mode."""
        self.running = False
        self.video_capture.release()
        cv2.destroyAllWindows()

    def on_close(self):
        """Handle the window close event."""
        self.stop_recognition()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
