import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from collections import deque
import threading
import time
import os

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur d'Émotions")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f5f7fa')
        
        # ==========================================
        # CONFIGUREZ VOS IMAGES ICI
        # ==========================================
        self.emotion_image_paths = {
            'Colère': 'images/colere.png',
            'Dégoût': 'images/degout.png',
            'Peur': 'images/peur.png',
            'Joie': 'images/joie.png',
            'Tristesse': 'images/tristesse.png',
            'Surprise': 'images/surprise.png',
            'Neutre': 'images/neutre.png'
        }
        # ==========================================
        
        # Variables
        self.is_running = False
        self.cap = None
        self.current_emotion = "Neutre"
        self.emotions = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Tristesse', 'Surprise', 'Neutre']
        self.emotion_colors = {
            'Colère': '#ef4444',
            'Dégoût': '#8b5cf6',
            'Peur': '#64748b',
            'Joie': '#f59e0b',
            'Tristesse': '#3b82f6',
            'Surprise': '#10b981',
            'Neutre': '#6b7280'
        }
        self.emotion_history = deque(maxlen=50)
        
        # Dictionary to store loaded images
        self.emotion_images = {}
        
        # Charger le détecteur de visage Haar Cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            messagebox.showerror("Erreur", "Impossible de charger le détecteur de visage")
        
        # Load emotion images
        self.load_emotion_images()
        
        self.setup_ui()
        
    def load_emotion_images(self):
        """Load all emotion images from the specified paths"""
        missing_images = []
        
        for emotion, path in self.emotion_image_paths.items():
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    img.thumbnail((250, 400), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.emotion_images[emotion] = photo
                except Exception as e:
                    missing_images.append(f"{emotion}: {str(e)}")
            else:
                missing_images.append(f"{emotion}: fichier introuvable ({path})")
        
        if missing_images:
            error_msg = "Images manquantes ou erreurs:\n\n" + "\n".join(missing_images)
            messagebox.showwarning("Attention", error_msg)
        
    def setup_ui(self):
        # En-tête simple
        header_frame = tk.Frame(self.root, bg='#ffffff', height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="Détecteur d'Émotions",
            font=('Segoe UI', 24, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        )
        title_label.pack(pady=20)
        
        # Frame principal avec padding
        main_frame = tk.Frame(self.root, bg='#f5f7fa')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Frame gauche - Vidéo (plus large)
        left_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.FLAT, bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        self.video_canvas = tk.Label(left_frame, bg='#000000')
        self.video_canvas.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        # Frame droit - Informations compactes
        right_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.FLAT, bd=0, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        
        # Émotion actuelle (plus proéminente)
        emotion_container = tk.Frame(right_frame, bg='#ffffff')
        emotion_container.pack(pady=25, padx=20, fill=tk.X)
        
        tk.Label(
            emotion_container,
            text="Émotion Détectée",
            font=('Segoe UI', 11),
            bg='#ffffff',
            fg='#6b7280'
        ).pack()
        
        self.emotion_label = tk.Label(
            emotion_container,
            text="Neutre",
            font=('Segoe UI', 36, 'bold'),
            bg='#ffffff',
            fg='#6b7280'
        )
        self.emotion_label.pack(pady=5)
        
        # Image de l'émotion (compacte)
        image_container = tk.Frame(right_frame, bg='#f9fafb', relief=tk.FLAT, bd=0)
        image_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.emotion_image_canvas = tk.Label(
            image_container, 
            bg='#f9fafb',
            text="",
            font=('Segoe UI', 10),
            fg='#9ca3af'
        )
        self.emotion_image_canvas.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        # Display initial neutral emotion image
        if 'Neutre' in self.emotion_images:
            self.display_emotion_image('Neutre')
        
        # Probabilités simplifiées (top 3 uniquement)
        prob_frame = tk.Frame(right_frame, bg='#ffffff')
        prob_frame.pack(pady=20, padx=20, fill=tk.X)
        
        tk.Label(
            prob_frame,
            text="Probabilités",
            font=('Segoe UI', 11, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        ).pack(pady=(0, 15), anchor='w')
        
        self.prob_labels = {}
        self.prob_bars = {}
        
        for emotion in self.emotions:
            emotion_row = tk.Frame(prob_frame, bg='#ffffff')
            emotion_row.pack(fill=tk.X, pady=5)
            
            label = tk.Label(
                emotion_row,
                text=emotion,
                font=('Segoe UI', 9),
                bg='#ffffff',
                fg='#374151',
                width=9,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            bar_bg = tk.Frame(emotion_row, bg='#e5e7eb', height=8)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
            
            bar = tk.Frame(bar_bg, bg=self.emotion_colors[emotion], height=8)
            bar.place(x=0, y=0, relwidth=0, relheight=1)
            self.prob_bars[emotion] = bar
            
            prob_label = tk.Label(
                emotion_row,
                text="0%",
                font=('Segoe UI', 9, 'bold'),
                bg='#ffffff',
                fg='#6b7280',
                width=4
            )
            prob_label.pack(side=tk.RIGHT)
            self.prob_labels[emotion] = prob_label
        
        # Contrôles (footer simplifié)
        control_frame = tk.Frame(self.root, bg='#ffffff', height=80)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        control_frame.pack_propagate(False)
        
        # Ligne de séparation subtile
        tk.Frame(control_frame, bg='#e5e7eb', height=1).pack(fill=tk.X)
        
        button_container = tk.Frame(control_frame, bg='#ffffff')
        button_container.pack(expand=True)
        
        self.start_button = tk.Button(
            button_container,
            text="Démarrer",
            command=self.start_detection,
            font=('Segoe UI', 11, 'bold'),
            bg='#10b981',
            fg='white',
            width=14,
            height=1,
            relief=tk.FLAT,
            cursor='hand2',
            bd=0
        )
        self.start_button.pack(side=tk.LEFT, padx=8)
        
        self.stop_button = tk.Button(
            button_container,
            text="Arrêter",
            command=self.stop_detection,
            font=('Segoe UI', 11, 'bold'),
            bg='#ef4444',
            fg='white',
            width=14,
            height=1,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED,
            bd=0
        )
        self.stop_button.pack(side=tk.LEFT, padx=8)
        
        self.status_label = tk.Label(
            button_container,
            text="Prêt",
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#6b7280'
        )
        self.status_label.pack(side=tk.LEFT, padx=15)
    
    def display_emotion_image(self, emotion):
        """Display the image associated with the detected emotion"""
        if emotion in self.emotion_images:
            self.emotion_image_canvas.config(
                image=self.emotion_images[emotion],
                text='',
                bg='#f9fafb'
            )
            self.emotion_image_canvas.image = self.emotion_images[emotion]
        else:
            self.emotion_image_canvas.config(
                image='',
                text=f"Image non disponible",
                font=('Segoe UI', 10),
                fg='#9ca3af',
                bg='#f9fafb'
            )
        
    def preprocess_face(self, face_img):
        """Prétraiter l'image du visage pour le modèle"""
        face_img = cv2.resize(face_img, (48, 48))
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img
    
    def predict_emotion(self, face_img):
        """Prédire l'émotion à partir d'une image de visage"""
        # Simulation de prédictions (REMPLACEZ PAR VOTRE MODÈLE)
        np.random.seed(int(time.time() * 1000) % 2**32)
        probs = np.random.dirichlet(np.ones(7) * 5)
        predictions = dict(zip(self.emotions, probs))
        
        emotion = max(predictions, key=predictions.get)
        
        return emotion, predictions
    
    def update_probabilities(self, probabilities):
        """Mettre à jour l'affichage des probabilités"""
        for emotion, prob in probabilities.items():
            self.prob_bars[emotion].place(relwidth=prob)
            self.prob_labels[emotion].config(text=f"{int(prob * 100)}%")
    
    def start_detection(self):
        """Démarrer la détection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erreur", "Impossible d'accéder à la webcam")
                return
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED, bg='#9ca3af')
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="En cours...", fg='#10b981')
            
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du démarrage: {str(e)}")
    
    def stop_detection(self):
        """Arrêter la détection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL, bg='#10b981')
        self.stop_button.config(state=tk.DISABLED, bg='#9ca3af')
        self.status_label.config(text="Arrêté", fg='#ef4444')
        self.video_canvas.config(image='')
    
    def detection_loop(self):
        """Boucle principale de détection"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emotion, probabilities = self.predict_emotion(face_roi)
                
                color = self.emotion_colors.get(emotion, '#ffffff')
                color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
                
                cv2.putText(
                    frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2
                )
                
                self.root.after(0, self.update_emotion_display, emotion, probabilities)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            self.root.after(0, self.update_video, photo)
            
            time.sleep(0.03)
    
    def update_video(self, photo):
        """Mettre à jour l'affichage vidéo"""
        self.video_canvas.config(image=photo)
        self.video_canvas.image = photo
    
    def update_emotion_display(self, emotion, probabilities):
        """Mettre à jour l'affichage de l'émotion"""
        self.current_emotion = emotion
        self.emotion_label.config(
            text=emotion,
            fg=self.emotion_colors.get(emotion, '#6b7280')
        )
        self.update_probabilities(probabilities)
        self.display_emotion_image(emotion)

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    
    def on_closing():
        app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()