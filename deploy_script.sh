#!/bin/bash
# Railway/Render Deployment Script

# 1. Requirements file
pip freeze > requirements.txt

# 2. Start command for Railway
# Add to railway.toml:
# [build]
# builder = "NIXPACKS" 
# [deploy]
# startCommand = "python canopy_web_server_fixed.py"

# 3. Environment Variables needed:
# PORT=5000
# PYTHON_VERSION=3.9.18

echo "Ready for Railway/Render deployment!"