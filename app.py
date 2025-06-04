from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model YOLOv8
model = YOLO('yolov8n.pt')  # Sesuaikan path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Prediksi dengan YOLOv8
        results = model.predict(source=filepath, save=True, project="static", name="results")
        
        # Path hasil prediksi
        output_path = f"static/results/{filename}"
        
        # Analisis hasil
        detected_objects = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence
                })
        
        # Hitung skor aksesibilitas
        accessibility_score = calculate_accessibility_score(detected_objects)
        
        return jsonify({
            'accessible': accessibility_score >= 0.5,
            'score': accessibility_score,
            'objects': detected_objects,
            'original_image': f'/static/uploads/{filename}',
            'processed_image': f'/static/results/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400
def calculate_accessibility_score(detections):
    score = 0.5  # Nilai default
    
    for obj in detections:
        if obj['class'] == 'sidewalk':
            score += 0.5 * obj['confidence']  # Bobot penuh untuk sidewalk
            
    return round(max(0, min(1, score)), 2)  # Clamp antara 0 dan 1


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    app.run(debug=True)