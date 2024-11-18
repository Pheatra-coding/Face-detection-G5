import cv2
import tkinter as tk
from tkinter import Label, Entry, Button, messagebox, Frame
from PIL import Image, ImageTk

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the main window
window = tk.Tk()
window.title("Face Detection Login System")
window.geometry("800x600")
window.configure(bg="#2c3e50")

# Create a label to display the video feed
video_label = Label(window, bg="#2c3e50")
video_label.pack(pady=30)

# Frame for entry fields and buttons (Card-like appearance)
login_frame = Frame(window, bg="#ecf0f1", padx=40, pady=30, bd=2, relief="ridge")
login_frame.pack(pady=20)

# Title label
title_label = Label(login_frame, text="Secure Face Login", font=("Helvetica", 20, "bold"), fg="#34495e", bg="#ecf0f1")
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

# Name label and entry field
name_label = Label(login_frame, text="Name:", font=("Helvetica", 14), fg="#2c3e50", bg="#ecf0f1")
name_label.grid(row=1, column=0, sticky="e", pady=10, padx=5)
name_entry = Entry(login_frame, font=("Helvetica", 14), width=20, bd=1, relief="solid")
name_entry.grid(row=1, column=1, pady=10, padx=5)

# Password label and entry field
password_label = Label(login_frame, text="Password:", font=("Helvetica", 14), fg="#2c3e50", bg="#ecf0f1")
password_label.grid(row=2, column=0, sticky="e", pady=10, padx=5)
password_entry = Entry(login_frame, font=("Helvetica", 14), width=20, bd=1, relief="solid", show="*")
password_entry.grid(row=2, column=1, pady=10, padx=5)

# Login button with hover effect
def on_enter(event):
    login_button.config(bg="#1abc9c", fg="white")

def on_leave(event):
    login_button.config(bg="#16a085", fg="white")

login_button = Button(login_frame, text="Login", font=("Helvetica", 14, "bold"), bg="#16a085", fg="white", bd=0,
                      activebackground="#1abc9c", activeforeground="white", command=lambda: login())
login_button.grid(row=3, column=0, columnspan=2, pady=20)
login_button.bind("<Enter>", on_enter)
login_button.bind("<Leave>", on_leave)

# Start the webcam
cap = cv2.VideoCapture(0)

# Function to handle login
def login():
    global name
    name = name_entry.get()
    password = password_entry.get()
    if name and password:
        messagebox.showinfo("Login Successful", f"Welcome, {name}!")
        detect_face()  # Start face detection
    else:
        messagebox.showwarning("Input Error", "Please enter both name and password.")

# Face detection function
def detect_face():
    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, detect_face)

# Run the application
window.mainloop()

# Release the video capture
cap.release()
cv2.destroyAllWindows()
