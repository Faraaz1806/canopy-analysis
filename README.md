# 🌲 Open-Canopy: AI-Powered Forest Canopy Height Analysis

**Production-ready web service for real-time forest canopy height estimation using satellite/aerial imagery.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![Node.js](https://img.shields.io/badge/Node.js-20+-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/fajwel/Open-Canopy.git
cd Open-Canopy

# 2. Install Python dependencies
pip install -r requirements_prod.txt

# 3. Install Node dependencies
npm install

# 4. Start server (Node proxy + Flask backend)
node server.js
```

**Open:** http://localhost:3000

---

##  Features

-  **Image Upload**: Drag & drop or click to upload forest images
-  **Complete Analysis**: Height distribution, land cover, statistics
-  **Interactive Reports**: Full-screen view with detailed visualizations
-  **Sample Images**: Pre-loaded Cloudinary integration
-  **Real-time Processing**: Instant canopy analysis results
-  **Responsive UI**: Works on desktop and mobile
-  **Production Ready**: Optimized for cloud deployment

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Browser   │◄────►│ Node.js :3000│◄────►│ Flask :5000     │
│  (Frontend) │      │   (Proxy)    │      │  (API/Analysis) │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Static Files│
                     └──────────────┘
```

**Why Node Proxy?**
- Single unified port for deployment (Render/Heroku requirement)
- Automatic Flask backend spawning
- Static file serving
- Future-proof for React/Vue frontend

---

##  Deployment

### **Option 1: Render (Recommended)**

**Build Command:**
```bash
pip install -r requirements_prod.txt && npm install
```

**Start Command:**
```bash
node server.js
```

**Environment Variables:**
```env
FLASK_PORT=5000
PYTHON_CMD=python
MODEL_PATH=useful_models/pvtv2.ckpt
DEBUG=False

# Optional: Cloudinary Integration
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

---

### **Option 2: Heroku**

```bash
# Login and create app
heroku login
heroku create canopy-analysis-app

# Set buildpacks
heroku buildpacks:set heroku/python
heroku buildpacks:add --index 2 heroku/nodejs

# Deploy
git push heroku main
```

**Procfile** (already included):
```
web: gunicorn --bind 0.0.0.0:$PORT canopy_web_server_fixed:app
```

---

### **Option 3: Railway**

1. Connect GitHub repo at [railway.app](https://railway.app)
2. Auto-deploys on push
3. Set environment variables in dashboard
4. No additional configuration needed

---

### **Option 4: Python Only (Direct Flask)**

**Build Command:**
```bash
pip install -r requirements_prod.txt
```

**Start Command:**
```bash
gunicorn -w 2 -b 0.0.0.0:$PORT canopy_web_server_fixed:app
```

Or development:
```bash
python canopy_web_server_fixed.py
```

---

##  Technology Stack

### Backend
- **Flask 3.1.0** - Web framework
- **Gunicorn 23.0.0** - Production WSGI server
- **NumPy 1.26.4** - Numerical processing
- **Pillow 10.4.0** - Image processing
- **Matplotlib 3.9.2** - Visualization generation

### Optional ML (Future)
- **PyTorch 2.4.1** - Deep learning framework
- **Torchvision 0.19.1** - Vision utilities

### Frontend
- **Vanilla JavaScript** - No framework dependencies
- **Responsive CSS** - Mobile-first design
- **Fullscreen API** - Immersive result viewing

### Infrastructure
- **Node.js + Express** - Unified deployment proxy
- **Cloudinary** - Optional cloud image hosting

---

##  Project Structure

```
Open-Canopy/
├── canopy_web_server_fixed.py    # Main Flask server
├── analysis.py                    # Canopy analysis engine
├── server.js                      # Node.js proxy server
├── requirements_prod.txt          # Python dependencies
├── package.json                   # Node dependencies
├── Procfile                       # Heroku/Railway config
├── runtime.txt                    # Python version
├── templates/
│   └── index.html                 # Web UI (black/white theme)
├── static/                        # Static assets (if any)
├── useful_models/
│   └── pvtv2.ckpt                 # Pre-trained model
├── uploads/                       # Temp upload storage
└── temp_outputs/                  # Analysis results cache
```

---

##  Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Node.js server port (set by platform) |
| `FLASK_PORT` | `5000` | Flask backend port |
| `DEBUG` | `False` | Enable debug mode |
| `MODEL_PATH` | `useful_models/pvtv2.ckpt` | Model file location |
| `MAX_UPLOAD_SIZE` | `16777216` | Max file size (16MB) |
| `PYTHON_CMD` | `python` | Python executable name |

### Cloudinary (Optional)
| Variable | Required | Description |
|----------|----------|-------------|
| `CLOUDINARY_CLOUD_NAME` | Yes | Your cloud name |
| `CLOUDINARY_API_KEY` | Yes | API key |
| `CLOUDINARY_API_SECRET` | Yes | API secret |
| `CLOUDINARY_FOLDER` | No | Upload folder (default: `canopy_samples`) |

---

## Testing

### Health Check
```bash
curl http://localhost:3000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0-WEB",
  "model_available": true,
  "timestamp": "2025-10-02T17:30:00"
}
```

### Upload Test
```bash
curl -X POST http://localhost:3000/upload \
  -F "file=@test_image.jpg"
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/upload` | POST | Upload & analyze image |
| `/analyze_cloud` | POST | Analyze Cloudinary URL |
| `/files` | GET | List available files |
| `/serve_file` | GET | Serve static file |

---

##  UI Features

- **Drag & Drop Upload** - Intuitive file selection
- **Sample Gallery** - Pre-loaded forest images
- **Fullscreen Toggle** - Press `F` or click button
- **Responsive Layout** - Mobile-optimized
- **Real-time Status** - Upload/analysis progress
- **Complete Reports** - Height maps, histograms, statistics

---

##  Model Information

**Current Model:** PVTv2 (Pyramid Vision Transformer)
- **Training Epoch:** 22
- **R² Score:** 40%
- **RMSE:** 11.63m
- **MAE:** 9.87m
- **Input Size:** 224×224
- **Max Height:** 18m

---

##  Known Limitations

1. **Model Size**: `pvtv2.ckpt` is ~400MB (use Git LFS or external storage)
2. **File Size Limit**: 16MB per upload
3. **Concurrent Uploads**: Single-threaded processing (future: background workers)
4. **Supported Formats**: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP

---

##  Future Enhancements

- [ ] Real-time model inference (currently simulated)
- [ ] Multi-model comparison
- [ ] Batch processing
- [ ] Export to GeoJSON/Shapefile
- [ ] User authentication
- [ ] API rate limiting
- [ ] Docker containerization
- [ ] Kubernetes deployment

---

##  Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---


**Made with 🌲 by the Open-Canopy Team**
