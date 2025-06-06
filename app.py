from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model YOLOv8
model = YOLO('./models/yolov8n_toilet-disable/weights/best.pt')  

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
        results = model.predict(source=filepath, save=True, project="static", name="results", exist_ok=True)

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
        
        # Hitung skor 
        accessibility_score = calculate_accessibility_score(detected_objects)
        
        return jsonify({
            'accessible': accessibility_score >= 70,
            'score': accessibility_score,  
            'objects': detected_objects,
            'original_image': f'/static/uploads/{filename}',
            'processed_image': f'/static/results/{filename}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

def calculate_accessibility_score(detections):
    feature_scores = {
        'accessible-toilet-sign': 20,  
        'grab-bars': 50,              
        'emergency-button': 20,       
        'Toilet': 10                  
    }
    
    total_score = 0
    detected_features = set()  
    
    for obj in detections:
        class_name = obj['class']
        if class_name in feature_scores and class_name not in detected_features:
            total_score += feature_scores[class_name]
            detected_features.add(class_name)
    
    return total_score  

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        uploads_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(uploads_folder):
            file_path = os.path.join(uploads_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        results_folder = 'static/results'
        for filename in os.listdir(results_folder):
            file_path = os.path.join(results_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    app.run(debug=True)