from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load models
model_toilet = YOLO('./models/yolov8n_toilet-disable/weights/best.pt')
model_jpo = YOLO('./models/yolov8n_jpo-disable/weights/best.pt')  # Update with your JPO model path
model_trotoar = YOLO('./models/yolov8n_jpo-disable/weights/best.pt')  # Update with your Trotoar model path

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

    kategori = request.form.get('kategori')  
    if kategori not in ['toilet', 'jpo', 'trotoar']:
        return jsonify({'error': 'Kategori tidak dikenali'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Select model based on category
        if kategori == 'toilet':
            model = model_toilet
        elif kategori == 'jpo':
            model = model_jpo
        else:
            model = model_trotoar

        results = model.predict(source=filepath, save=True, project="static", name="results", exist_ok=True)

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

        # Calculate accessibility score
        accessibility_score = calculate_accessibility_score(detected_objects, kategori)

        return jsonify({
            'accessible': accessibility_score >= 70,
            'score': accessibility_score,
            'kategori': kategori,
            'objects': detected_objects,
            'original_image': f'/static/uploads/{filename}',
            'processed_image': f'/static/results/{os.path.splitext(filename)[0]}.jpg'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

def calculate_accessibility_score(detections, kategori):
    # Score distribution
    if kategori == 'toilet':
        feature_scores = {
            'accessible-toilet-sign': 20,
            'grab-bars': 50,
            'emergency-button': 20,
            'Toilet': 10
        }
    elif kategori == 'jpo':
        feature_scores = {
            'disable_sign': 30,
            'elevator': 40,
            'ramp': 20,
            'tactile-paving': 10
        }
    elif kategori == 'trotoar':
        feature_scores = {
            'tactile-paving': 100
        }
    else:
        feature_scores = {}

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