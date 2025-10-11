import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from collections import deque
import threading
import time

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur d'Émotions en Temps Réel")
        self.root.geometry("1200x700")
        self.root.configure(bg='#1e1e2e')
        
        # Variables
        self.is_running = False
        self.cap = None
        self.current_emotion = "Neutre"
        self.emotions = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Tristesse', 'Surprise', 'Neutre']
        self.emotion_colors = {
            'Colère': '#e74c3c',
            'Dégoût': '#9b59b6',
            'Peur': '#34495e',
            'Joie': '#f39c12',
            'Tristesse': '#3498db',
            'Surprise': '#1abc9c',
            'Neutre': '#95a5a6'
        }
        self.emotion_history = deque(maxlen=50)
        
        # Charger le détecteur de visage Haar Cascade
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            messagebox.showerror("Erreur", "Impossible de charger le détecteur de visage")
        
        self.setup_ui()
        
    def setup_ui(self):
        # En-tête
        header_frame = tk.Frame(self.root, bg='#2d2d44', height=60)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🎭 Détecteur d'Émotions par IA",
            font=('Helvetica', 20, 'bold'),
            bg='#2d2d44',
            fg='#ffffff'
        )
        title_label.pack(pady=15)
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Frame gauche - Vidéo
        left_frame = tk.Frame(main_frame, bg='#2d2d44', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_label = tk.Label(left_frame, text="Vidéo", font=('Helvetica', 12), 
                              bg='#2d2d44', fg='#ffffff')
        video_label.pack(pady=5)
        
        self.video_canvas = tk.Label(left_frame, bg='#000000')
        self.video_canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Frame droit - Informations
        right_frame = tk.Frame(main_frame, bg='#2d2d44', relief=tk.RAISED, bd=2, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Émotion actuelle
        emotion_frame = tk.Frame(right_frame, bg='#2d2d44')
        emotion_frame.pack(pady=20, padx=15, fill=tk.X)
        
        tk.Label(
            emotion_frame,
            text="Émotion Détectée",
            font=('Helvetica', 14, 'bold'),
            bg='#2d2d44',
            fg='#ffffff'
        ).pack()
        
        self.emotion_label = tk.Label(
            emotion_frame,
            text="Neutre",
            font=('Helvetica', 32, 'bold'),
            bg='#2d2d44',
            fg='#95a5a6'
        )
        self.emotion_label.pack(pady=10)
        
        # Canvas pour le graphique circulaire
        self.graph_canvas = tk.Canvas(
            right_frame,
            width=300,
            height=300,
            bg='#2d2d44',
            highlightthickness=0
        )
        self.graph_canvas.pack(pady=20)
        
        # Probabilités des émotions
        prob_frame = tk.Frame(right_frame, bg='#2d2d44')
        prob_frame.pack(pady=10, padx=15, fill=tk.BOTH, expand=True)
        
        tk.Label(
            prob_frame,
            text="Probabilités",
            font=('Helvetica', 12, 'bold'),
            bg='#2d2d44',
            fg='#ffffff'
        ).pack(pady=(0, 10))
        
        self.prob_labels = {}
        self.prob_bars = {}
        
        for emotion in self.emotions:
            emotion_row = tk.Frame(prob_frame, bg='#2d2d44')
            emotion_row.pack(fill=tk.X, pady=3)
            
            label = tk.Label(
                emotion_row,
                text=emotion,
                font=('Helvetica', 9),
                bg='#2d2d44',
                fg='#ffffff',
                width=10,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            bar_frame = tk.Frame(emotion_row, bg='#1e1e2e', height=15)
            bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            bar = tk.Frame(bar_frame, bg=self.emotion_colors[emotion], height=15)
            bar.place(x=0, y=0, relwidth=0, relheight=1)
            self.prob_bars[emotion] = bar
            
            prob_label = tk.Label(
                emotion_row,
                text="0%",
                font=('Helvetica', 9),
                bg='#2d2d44',
                fg='#ffffff',
                width=5
            )
            prob_label.pack(side=tk.RIGHT)
            self.prob_labels[emotion] = prob_label
        
        # Frame des contrôles
        control_frame = tk.Frame(self.root, bg='#2d2d44', height=80)
        control_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        control_frame.pack_propagate(False)
        
        button_frame = tk.Frame(control_frame, bg='#2d2d44')
        button_frame.pack(expand=True)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ Démarrer",
            command=self.start_detection,
            font=('Helvetica', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            width=15,
            height=2,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⬛ Arrêter",
            command=self.stop_detection,
            font=('Helvetica', 12, 'bold'),
            bg='#e74c3c',
            fg='white',
            width=15,
            height=2,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(
            control_frame,
            text="● Prêt",
            font=('Helvetica', 10),
            bg='#2d2d44',
            fg='#95a5a6'
        )
        self.status_label.pack(pady=5)
        
    def preprocess_face(self, face_img):
        """
        Prétraiter l'image du visage pour le modèle
        MODIFIEZ CETTE FONCTION selon les besoins de votre modèle
        """
        # Redimensionner à 48x48 (taille commune pour les modèles d'émotions)
        face_img = cv2.resize(face_img, (48, 48))
        # Convertir en niveaux de gris si nécessaire
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Normaliser
        face_img = face_img / 255.0
        # Ajouter les dimensions pour le batch et le channel
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img
    
    def predict_emotion(self, face_img):
        """
        Prédire l'émotion à partir d'une image de visage
        REMPLACEZ CETTE FONCTION par votre modèle d'IA
        
        Args:
            face_img: Image du visage (numpy array)
            
        Returns:
            tuple: (emotion_name, probabilities_dict)
        """
        # EXEMPLE: Remplacez cette simulation par votre modèle réel
        # processed_face = self.preprocess_face(face_img)
        # predictions = your_model.predict(processed_face)[0]
        
        # Simulation de prédictions (REMPLACEZ PAR VOTRE MODÈLE)
        np.random.seed(int(time.time() * 1000) % 2**32)
        probs = np.random.dirichlet(np.ones(7) * 5)
        predictions = dict(zip(self.emotions, probs))
        
        # Trouver l'émotion dominante
        emotion = max(predictions, key=predictions.get)
        
        return emotion, predictions
    
    def draw_emotion_graph(self, probabilities):
        """Dessiner un graphique circulaire des probabilités"""
        self.graph_canvas.delete("all")
        
        center_x, center_y = 150, 150
        radius = 80
        
        # Dessiner le cercle de fond
        self.graph_canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            fill='#1e1e2e', outline='#3d3d5c', width=2
        )
        
        # Dessiner les segments pour chaque émotion
        start_angle = 0
        for emotion in self.emotions:
            prob = probabilities.get(emotion, 0)
            extent = prob * 360
            
            if prob > 0.01:  # Afficher seulement si significatif
                self.graph_canvas.create_arc(
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius,
                    start=start_angle, extent=extent,
                    fill=self.emotion_colors[emotion], outline='#2d2d44', width=2
                )
            start_angle += extent
        
        # Cercle central
        inner_radius = 50
        self.graph_canvas.create_oval(
            center_x - inner_radius, center_y - inner_radius,
            center_x + inner_radius, center_y + inner_radius,
            fill='#2d2d44', outline='#3d3d5c', width=2
        )
        
        # Texte central
        max_emotion = max(probabilities, key=probabilities.get)
        max_prob = probabilities[max_emotion]
        self.graph_canvas.create_text(
            center_x, center_y,
            text=f"{int(max_prob * 100)}%",
            font=('Helvetica', 18, 'bold'),
            fill='#ffffff'
        )
    
    def update_probabilities(self, probabilities):
        """Mettre à jour l'affichage des probabilités"""
        for emotion, prob in probabilities.items():
            # Mettre à jour la barre
            self.prob_bars[emotion].place(relwidth=prob)
            # Mettre à jour le texte
            self.prob_labels[emotion].config(text=f"{int(prob * 100)}%")
    
    def start_detection(self):
        """Démarrer la détection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erreur", "Impossible d'accéder à la webcam")
                return
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="● En cours...", fg='#27ae60')
            
            # Démarrer le thread de détection
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du démarrage: {str(e)}")
    
    def stop_detection(self):
        """Arrêter la détection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="● Arrêté", fg='#e74c3c')
        self.video_canvas.config(image='')
    
    def detection_loop(self):
        """Boucle principale de détection"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Retourner l'image horizontalement
            frame = cv2.flip(frame, 1)
            
            # Convertir en niveaux de gris pour la détection de visage
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détecter les visages
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            
            # Traiter chaque visage détecté
            for (x, y, w, h) in faces:
                # Extraire le visage
                face_roi = gray[y:y+h, x:x+w]
                
                # Prédire l'émotion
                emotion, probabilities = self.predict_emotion(face_roi)
                
                # Dessiner le rectangle autour du visage
                color = self.emotion_colors.get(emotion, '#ffffff')
                # Convertir couleur hex en BGR
                color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
                
                # Afficher l'émotion
                cv2.putText(
                    frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2
                )
                
                # Mettre à jour l'interface (dans le thread principal)
                self.root.after(0, self.update_emotion_display, emotion, probabilities)
            
            # Convertir pour Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            # Mettre à jour l'affichage vidéo
            self.root.after(0, self.update_video, photo)
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_video(self, photo):
        """Mettre à jour l'affichage vidéo"""
        self.video_canvas.config(image=photo)
        self.video_canvas.image = photo
    
    def update_emotion_display(self, emotion, probabilities):
        """Mettre à jour l'affichage de l'émotion"""
        self.emotion_label.config(
            text=emotion,
            fg=self.emotion_colors.get(emotion, '#ffffff')
        )
        self.update_probabilities(probabilities)
        self.draw_emotion_graph(probabilities)

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    
    # Gérer la fermeture de la fenêtre
    def on_closing():
        app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()