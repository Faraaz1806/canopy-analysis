from flask import Flask, request, jsonify, render_template, send_file
import os
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Fix threading issues

# Import your existing analysis
from analysis import CanopyAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize your analyzer
analyzer = CanopyAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'tif']

@app.route('/')
def index():
    return render_template('index_simple.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file'})
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file'})
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run your analysis directly
        result = analyzer.analyze_image(filepath)
        
        if result and 'output_path' in result:
            # Read result image and convert to base64
            with open(result['output_path'], 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # Clean up
            os.remove(filepath)
            if os.path.exists(result['output_path']):
                os.remove(result['output_path'])
                
            return jsonify({
                'success': True,
                'height_map_base64': img_data
            })
        else:
            return jsonify({'success': False, 'error': 'Analysis failed to generate output'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/files')
def list_files():
    try:
        # List images from datasets/canopy_height
        image_dir = 'datasets/canopy_height'
        if os.path.exists(image_dir):
            files = []
            for f in os.listdir(image_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    files.append({'path': os.path.join(image_dir, f), 'name': f})
            return jsonify({'success': True, 'files': files[:4]})  # Only 4 files
        return jsonify({'success': True, 'files': []})
    except:
        return jsonify({'success': True, 'files': []})

@app.route('/serve_file')
def serve_file():
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path)
    return "File not found", 404

@app.route('/analyze_existing', methods=['POST'])
def analyze_existing():
    try:
        data = request.get_json()
        filepath = data.get('file_path')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'})
        
        # Run your analysis directly
        result = analyzer.analyze_image(filepath)
        
        if result and 'output_path' in result:
            # Read result image and convert to base64
            with open(result['output_path'], 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            # Clean up result file
            if os.path.exists(result['output_path']):
                os.remove(result['output_path'])
                
            return jsonify({
                'success': True,
                'height_map_base64': img_data
            })
        else:
            return jsonify({'success': False, 'error': 'Analysis failed to generate output'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🌲 Simple Canopy Server Starting...")
    print("📍 http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)