"""
 CANOPY ANALYSIS WEB SERVER 
Production-ready Flask server for forest canopy height estimation
Upload image → Get ins Model Information:
• Architecture: PVTv2 (Pyramid Vision Transformer)
• Training epochs: {self.model_info['epoch']}
• Model accuracy: R² = 40%
• Root Mean Square Error: 11.63m
• Mean Absolute Error: 9.87manopy analysis with beautiful visualization!
Uses your existing analysis.py CanopyAnalyzer class
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import werkzeug
from werkzeug.utils import secure_filename
import io
import base64
import json

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

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Update CONFIG for web version
CONFIG['OUTPUT_DPI'] = 150  # Reduced for web
CONFIG['VERSION'] = '2.0.0-WEB'

class WebCanopyAnalyzer(CanopyAnalyzer):
    """Web-optimized version of your CanopyAnalyzer"""
    
    def __init__(self, model_path=None):
        super().__init__(model_path)
        print(f" Web Canopy Analyzer v{CONFIG['VERSION']} Initialized")
        print(f" Using your analysis.py CanopyAnalyzer class")
    
    def create_web_analysis_report(self, original_img, height_map, image_name):
        """Web-optimized version of your create_analysis_report method"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use your exact tree detection logic
        tree_threshold = max(7.0, height_map.mean() + 0.3 * height_map.std())
        tree_mask = height_map > tree_threshold
        coverage_percent = (np.sum(tree_mask) / tree_mask.size) * 100
        
        # Create web-friendly visualization (same style as your analysis.py)
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        fig.suptitle(f'Professional Canopy Analysis Report - {image_name}\n'
                    f'PVTv2 Model (R²=40%, RMSE=11.63m) | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=16, fontweight='bold')
        
        # Original image
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax1.imshow(original_img)
        ax1.set_title("Satellite/Aerial Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Height map with INTENSE GREEN colormap - Same as your analysis.py!
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        im2 = ax2.imshow(height_map, cmap='Greens', vmin=0, vmax=CONFIG['MAX_HEIGHT'])
        ax2.set_title(f"Canopy Height Map (Enhanced Green)\nRange: {height_map.min():.1f} - {height_map.max():.1f}m", 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, label='Height (meters)', shrink=0.8)
        cbar2.ax.tick_params(labelsize=10)
        
        # Tree detection mask - Same overlay style as your analysis.py
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        ax3.imshow(original_img, alpha=0.5)  # Faded background
        ax3.imshow(tree_mask, cmap='Greens', alpha=0.7, vmin=0, vmax=1)  # Bright green trees
        ax3.set_title(f"Tree Detection (Enhanced)\nThreshold: {tree_threshold:.1f}m", 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Height distribution histogram
        ax4 = plt.subplot2grid((2, 3), (1, 0))
        ax4.hist(height_map.flatten(), bins=40, alpha=0.7, color='forestgreen', edgecolor='black')
        ax4.axvline(tree_threshold, color='red', linestyle='--', linewidth=2, label=f'Tree threshold: {tree_threshold:.1f}m')
        ax4.set_xlabel('Height (meters)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Height Distribution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Forest coverage analysis
        ax5 = plt.subplot2grid((2, 3), (1, 1))
        colors = ['forestgreen', 'lightgray']
        labels = [f'Forest\n{coverage_percent:.1f}%', f'Non-forest\n{100-coverage_percent:.1f}%']
        wedges, texts, autotexts = ax5.pie([coverage_percent, 100-coverage_percent], 
                                          labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 10})
        ax5.set_title('Land Cover Analysis', fontsize=12, fontweight='bold')
        
        # Detailed statistics panel (same as your analysis.py)
        ax6 = plt.subplot2grid((2, 3), (1, 2))
        stats_text = f"""
FOREST ANALYSIS SUMMARY

 Coverage Statistics:
• Forest coverage: {coverage_percent:.1f}%
• Total analyzed area: {height_map.size:,} pixels
• Tree pixels detected: {np.sum(tree_mask):,}

 Height Statistics:
• Mean canopy height: {height_map.mean():.2f}m
• Maximum height: {height_map.max():.2f}m
• Minimum height: {height_map.min():.2f}m
• Standard deviation: {height_map.std():.2f}m
• Tree detection threshold: {tree_threshold:.1f}m

 Model Information:
• Architecture: PVTv2 (Pyramid Vision Transformer)
• Training epochs: {self.model_info['epoch']}
• Model accuracy: R² = 35%
• Root Mean Square Error: 11.63m
• Mean Absolute Error: 9.87m

 Analysis Details:
• Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
• Version: {CONFIG['VERSION']}
• Status: WEB DEPLOYMENT READY
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save result for web
        result_filename = f"web_result_{timestamp}.png"
        result_path = os.path.join('static/results', result_filename)
        plt.savefig(result_path, dpi=CONFIG['OUTPUT_DPI'], bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f" Web analysis saved: {result_path}")
        
        # Return web-friendly results
        return {
            'success': True,
            'coverage_percent': round(coverage_percent, 2),
            'mean_height': round(height_map.mean(), 2),
            'max_height': round(height_map.max(), 2),
            'min_height': round(height_map.min(), 2),
            'std_height': round(height_map.std(), 2),
            'tree_threshold': round(tree_threshold, 2),
            'tree_pixels': int(np.sum(tree_mask)),
            'total_pixels': int(height_map.size),
            'result_image': result_filename,
            'timestamp': timestamp,
            'model_info': self.model_info
        }
    
    def analyze_web_image(self, image_path):
        """Web version of your analyze_image method"""
        print(f" Web Analysis Starting: {os.path.basename(image_path)}")
        
        try:
            # Use your exact preprocessing
            img_resized, img_array, original_size = self.preprocess_image(image_path)
            
            # Use your exact simulation
            print(" Running your canopy analysis algorithm...")
            height_map = self.simulate_canopy_analysis(img_array)
            
            # Generate web report
            print(" Creating web-optimized visualization...")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            results = self.create_web_analysis_report(img_resized, height_map, image_name)
            
            print(f" Web analysis completed!")
            return results
            
        except Exception as e:
            print(f" Web analysis failed: {e}")
            return {'success': False, 'error': str(e)}

# Initialize global analyzer using your analysis.py
analyzer = WebCanopyAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index_simple.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            print(f" File saved: {file_path}")
            
            # Analyze the image using your analysis.py methods
            results = analyzer.analyze_web_image(file_path)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

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
        'evaluation',
        'static/results'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                for file in os.listdir(path):
                    if allowed_file(file):
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

@app.route('/analyze_existing', methods=['POST'])
def analyze_existing_file():
    """Analyze an existing file from the project"""
    data = request.get_json()
    file_path = data.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    
    if not allowed_file(file_path):
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    try:
        print(f"Analyzing existing file: {file_path}")
        
        # Analyze the existing image using your analysis.py methods
        results = analyzer.analyze_web_image(file_path)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'})

if __name__ == '__main__':
    print("" * 30)
    print(" CANOPY ANALYSIS WEB SERVER STARTING...")
    print(f" Version: {CONFIG['VERSION']}")
    print(f" Model: PVTv2 (Epoch {analyzer.model_info['epoch']})")
    print(" Server will be available at: http://localhost:5000")
    print("" * 30)
    
    app.run(debug=True, host='0.0.0.0', port=5000)