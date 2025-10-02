"""
CANOPY ANALYSIS WEB SERVER
Production-ready Flask server for forest canopy height estimation
Upload image → Get instant canopy analysis!
Uses your existing analysis.py CanopyAnalyzer class
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Fix threading issues
import matplotlib.pyplot as plt
from datetime import datetime
import werkzeug
from werkzeug.utils import secure_filename
import io
import base64
import json
import tempfile
import requests

# Optional Cloudinary support
cloudinary_available = False
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    cloudinary_available = True
except Exception:
    pass

def configure_cloudinary():
    """Configure Cloudinary if credentials are present."""
    if not cloudinary_available:
        return False
    cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME')
    api_key = os.environ.get('CLOUDINARY_API_KEY')
    api_secret = os.environ.get('CLOUDINARY_API_SECRET')
    if not (cloud_name and api_key and api_secret):
        return False
    try:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )
        return True
    except Exception:
        return False

# Import your analysis.py CanopyAnalyzer
from analysis import CanopyAnalyzer, CONFIG

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.secret_key = 'canopy-analysis-server-2024'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('temp_outputs', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}
SUPPORTED_FORMATS = "PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"

# Update CONFIG for web version
CONFIG['OUTPUT_DPI'] = 150  # Reduced for web
CONFIG['VERSION'] = '2.0.0-WEB'

class WebCanopyAnalyzer(CanopyAnalyzer):
    """Web-optimized version of your CanopyAnalyzer"""
    
    def __init__(self, model_path=None):
        super().__init__(model_path)
        print(f"Web Canopy Analyzer v{CONFIG['VERSION']} Initialized")
        print(f"Using your analysis.py CanopyAnalyzer class")
    
    def analyze_web_image_simple(self, image_path):
        """Simplified web analysis - returns complete visualization"""
        print(f"Web Analysis Starting: {os.path.basename(image_path)}")
        
        try:
            # Use your exact preprocessing
            img_resized, img_array, original_size = self.preprocess_image(image_path)
            
            # Use your exact simulation
            print("Running your canopy analysis algorithm...")
            height_map = self.simulate_canopy_analysis(img_array)
            
            # Generate complete analysis report using your method
            print("Creating complete analysis visualization...")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            analysis_result = self.create_analysis_report(img_resized, height_map, image_name, "temp_outputs")
            
            # Read the generated image and convert to base64
            if analysis_result and 'output_path' in analysis_result:
                with open(analysis_result['output_path'], 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                
                # Clean up temp file
                os.remove(analysis_result['output_path'])
                
                print(f"Analysis completed successfully!")
                
                return {
                    'success': True,
                    'height_map_base64': img_data,
                    'coverage_percent': round(analysis_result['coverage_percent'], 2),
                    'mean_height': round(analysis_result['mean_height'], 2),
                    'max_height': round(analysis_result['max_height'], 2),
                    'tree_threshold': round(analysis_result['tree_threshold'], 2),
                    'timestamp': analysis_result['timestamp'],
                    'model_info': self.model_info
                }
            else:
                return {'success': False, 'error': 'Failed to generate analysis report'}
            
        except Exception as e:
            print(f"Web analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Analysis error: {str(e)}'}

# Initialize global analyzer using your analysis.py
analyzer = WebCanopyAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis with comprehensive error handling"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded. Please select a file.'})
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected. Please choose an image file.'})
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': f'Unsupported file format. Please upload: {SUPPORTED_FORMATS}'
            })
        
        # Check file size (additional check)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)     # Seek back to beginning
        
        if file_size > 16 * 1024 * 1024:  # 16MB limit
            return jsonify({
                'success': False, 
                'error': 'File too large. Maximum size is 16MB.'
            })
        
        if file_size == 0:
            return jsonify({
                'success': False, 
                'error': 'Empty file detected. Please upload a valid image.'
            })
        
        # Process file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save file
            print(f" Saving to: {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
            file.save(file_path)
            print(f" File saved: {file_path} ({file_size} bytes)")
            
            # Validate image format by trying to open
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    print(f" Valid image: {img.format} {img.size} {img.mode}")
            except Exception as img_error:
                os.remove(file_path)  # Clean up invalid file
                return jsonify({
                    'success': False, 
                    'error': f'Invalid image file: {str(img_error)}'
                })
            
            # Analyze the image
            print(" Starting analysis...")
            print(f"File path exists: {os.path.exists(file_path)}")
            print(f"Analyzer available: {analyzer.model_info['available']}")
            
            # Skip model availability check for production deployment
            # if not analyzer.model_info['available']:
            #     return jsonify({'success': False, 'error': 'Analysis model not available. Please check server configuration.'})
            
            try:
                results = analyzer.analyze_web_image_simple(file_path)
                print(f" Analysis results: {results}")
                
                # Ensure we have a valid response
                if not results:
                    return jsonify({'success': False, 'error': 'Analysis returned no results'})
                    
                if not isinstance(results, dict):
                    return jsonify({'success': False, 'error': 'Analysis returned invalid format'})
                    
            except Exception as analysis_error:
                print(f" Analysis error: {analysis_error}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': f'Analysis failed: {str(analysis_error)}'})
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(" Cleaned up temp file")
            
            return jsonify(results)
            
        except FileNotFoundError:
            return jsonify({'success': False, 'error': 'File upload failed. Please try again.'})
        except PermissionError:
            return jsonify({'success': False, 'error': 'Permission denied. Please try again.'})
        except Exception as e:
            # Clean up file if exists
            if os.path.exists(file_path):
                os.remove(file_path)
            
            error_msg = str(e)
            if "cannot identify image file" in error_msg.lower():
                return jsonify({'success': False, 'error': 'Invalid image format. Please upload a valid image file.'})
            elif "truncated" in error_msg.lower():
                return jsonify({'success': False, 'error': 'Corrupted image file. Please try another image.'})
            else:
                print(f" Analysis error: {error_msg}")
                return jsonify({'success': False, 'error': f'Analysis failed. Please try with a different image.'})
    
    except Exception as e:
        print(f" Server error: {str(e)}")
        return jsonify({'success': False, 'error': 'Server error. Please refresh and try again.'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': CONFIG['VERSION'],
        'model_available': analyzer.model_info['available'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/files')
def list_files():
    """List available image files in the project"""
    image_files = []
    
    # Check for images in common directories
    search_paths = [
        '.',  # Root directory
        'images',
        'examples',
        'datasets',
        'evaluation'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                for file in os.listdir(path):
                    if allowed_file(file) and not file.startswith('web_result_'):
                        full_path = os.path.join(path, file)
                        if os.path.isfile(full_path):
                            # Get file size
                            size = os.path.getsize(full_path)
                            size_mb = round(size / (1024 * 1024), 2)
                            
                            image_files.append({
                                'name': file,
                                'path': path,
                                'full_path': full_path,
                                'size_mb': size_mb,
                                'url': f'/serve_file?path={full_path.replace(os.sep, "/")}'
                            })
            except Exception as e:
                print(f"Error scanning {path}: {e}")
    
    return jsonify({
        'success': True,
        'files': image_files,
        'count': len(image_files)
    })

@app.route('/serve_file')
def serve_file():
    """Serve a file from project directory"""
    file_path = request.args.get('path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)

@app.route('/cloud_samples')
def cloud_samples():
    """Return list of Cloudinary or manifest-based sample images.

    Priority order:
      1. Live Cloudinary listing if credentials + folder available
      2. Local manifest file
      3. Empty list
    """
    # Try dynamic Cloudinary listing
    folder = os.environ.get('CLOUDINARY_FOLDER', 'canopy_samples')
    if configure_cloudinary():
        try:
            # List resources with given prefix/folder
            result = cloudinary.api.resources(type='upload', prefix=folder, max_results=30)
            resources = result.get('resources', [])
            images = []
            for r in resources:
                # Filter by image format
                if r.get('resource_type') == 'image':
                    images.append({
                        'public_id': r.get('public_id'),
                        'url': r.get('secure_url') or r.get('url'),
                        'format': r.get('format'),
                        'bytes': r.get('bytes'),
                        'filename': r.get('public_id').split('/')[-1]
                    })
            if images:
                return jsonify({'success': True, 'source': 'cloud', 'images': images, 'count': len(images)})
        except Exception as e:
            # Fall through to manifest if cloud listing fails
            print(f"Cloudinary listing failed: {e}")

    # Fallback: local manifest
    manifest_path = os.environ.get('CLOUD_MANIFEST', 'image_manifest.json')
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'success': True, 'source': 'manifest', 'images': data[:30], 'count': len(data)})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Manifest read error: {e}'})

    return jsonify({'success': True, 'images': [], 'count': 0, 'source': 'none'})

@app.route('/cloud_upload', methods=['POST'])
def cloud_upload():
    """Upload a remote image (by URL) directly to Cloudinary and return its info."""
    if not configure_cloudinary():
        return jsonify({'success': False, 'error': 'Cloudinary not configured'})
    try:
        data = request.get_json() or {}
        remote_url = data.get('url')
        folder = os.environ.get('CLOUDINARY_FOLDER', 'canopy_samples')
        if not remote_url:
            return jsonify({'success': False, 'error': 'No url provided'})
        if not remote_url.lower().startswith('http'):
            return jsonify({'success': False, 'error': 'Invalid url'})
        upload_result = cloudinary.uploader.upload(remote_url, folder=folder)
        return jsonify({
            'success': True,
            'public_id': upload_result.get('public_id'),
            'url': upload_result.get('secure_url') or upload_result.get('url'),
            'format': upload_result.get('format'),
            'bytes': upload_result.get('bytes')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload failed: {e}'})

@app.route('/analyze_cloud', methods=['POST'])
def analyze_cloud():
    """Download a remote (Cloudinary) image temporarily and analyze it."""
    try:
        data = request.get_json() or {}
        url = data.get('url')
        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'})
        # Basic validation
        if not url.lower().startswith('http'):
            return jsonify({'success': False, 'error': 'Invalid URL'})
        # Download
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to fetch image (status {r.status_code})'})
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(r.content)
            tmp_path = tmp.name
        # Analyze
        results = analyzer.analyze_web_image_simple(tmp_path)
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Cloud analysis failed: {e}'})

@app.route('/analyze_existing', methods=['POST'])
def analyze_existing_file():
    """Analyze an existing file from the project with error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
            
        file_path = data.get('file_path')
        
        # Validate file path
        if not file_path:
            return jsonify({'success': False, 'error': 'No file path provided'})
            
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found. It may have been moved or deleted.'})
        
        if not allowed_file(file_path):
            return jsonify({
                'success': False, 
                'error': f'Unsupported file format. Supported formats: {SUPPORTED_FORMATS}'
            })
        
        # Validate file size and format
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return jsonify({'success': False, 'error': 'Empty file detected'})
                
            # Validate image
            from PIL import Image
            with Image.open(file_path) as img:
                print(f" Analyzing: {img.format} {img.size} {img.mode}")
                
        except Exception as img_error:
            return jsonify({'success': False, 'error': 'Invalid or corrupted image file'})
        
        print(f" Analyzing existing file: {os.path.basename(file_path)}")
        
        # Analyze the existing image
        results = analyzer.analyze_web_image_simple(file_path)
        
        return jsonify(results)
        
    except Exception as e:
        print(f" Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': 'Analysis failed. Please try with a different image.'})

if __name__ == '__main__':
    import os

    # Prefer dedicated FLASK_PORT (so a Node proxy can use the main PORT on Render/other PaaS)
    port = int(os.environ.get('FLASK_PORT') or os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("=" * 50)
    print("CANOPY ANALYSIS WEB SERVER STARTING...")
    print(f" Version: {CONFIG['VERSION']}")
    print(f" Model: PVTv2 (Epoch {analyzer.model_info['epoch']})")
    print(f" Port: {port}")
    print(f" Debug: {debug_mode}")
    if port == 5000:
        print(f" Local Flask (internal) URL: http://localhost:{port}")
    else:
        print(f" Flask bound to port {port}")
    print("=" * 50)
    
    # Production ready settings
    app.run(
        debug=debug_mode,
        host='0.0.0.0', 
        port=port,
        threaded=True  # Handle multiple requests
    )