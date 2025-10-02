// Node.js Express server that can optionally spawn and proxy to the Python Flask backend
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

const app = express();
// If deploying to Render, Render will inject PORT. We run Node on PORT and Flask on FLASK_PORT (or fallback)
let NODE_PORT = process.env.PORT || 3000;
// If the default 3000 is busy locally, fall back to 3010 automatically
const fallbackPort = 3010;
const net = require('net');

function checkPort(port) {
  return new Promise((resolve) => {
    const tester = net.createServer()
      .once('error', () => resolve(false))
      .once('listening', () => tester.once('close', () => resolve(true)).close())
      .listen(port);
  });
}

const decideNodePort = async () => {
  if (process.env.PORT) return process.env.PORT;
  const free = await checkPort(NODE_PORT);
  if (free) return NODE_PORT;
  const freeFallback = await checkPort(fallbackPort);
  if (freeFallback) return fallbackPort;
  return NODE_PORT; // last resort - will throw if also busy
};

let FLASK_PORT = process.env.FLASK_PORT || 5000;
const PYTHON_CMD = process.env.PYTHON_CMD || 'python';
const FLASK_ENTRY = process.env.FLASK_ENTRY || 'canopy_web_server_fixed.py';
let flaskProcess = null;

function waitForFlask(retries = 40, interval = 500) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      const req = http.get({ host: '127.0.0.1', port: FLASK_PORT, path: '/health', timeout: 300 }, (res) => {
        if (res.statusCode === 200) {
          console.log('[OK] Flask backend is up');
          res.resume();
          return resolve();
        }
        res.resume();
        if (n <= 0) return reject(new Error('Flask did not become ready in time (bad status)'));
        setTimeout(() => attempt(n - 1), interval);
      });
      req.on('error', () => {
        if (n <= 0) return reject(new Error('Flask did not become ready in time (no response)'));
        setTimeout(() => attempt(n - 1), interval);
      });
    };
    attempt(retries);
  });
}

function startFlask() {
  console.log(`[INFO] Spawning Flask backend on port ${FLASK_PORT} using ${PYTHON_CMD} ${FLASK_ENTRY}`);
  flaskProcess = spawn(PYTHON_CMD, [FLASK_ENTRY], {
    env: { ...process.env, FLASK_PORT: String(FLASK_PORT) },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  flaskProcess.stdout.on('data', (d) => process.stdout.write(`[FLASK] ${d}`));
  flaskProcess.stderr.on('data', (d) => process.stderr.write(`[FLASK ERR] ${d}`));
  flaskProcess.on('exit', (code) => {
    console.log(`[FLASK] exited with code ${code}`);
  });
  return waitForFlask();
}

// Serve static files (if any)
app.use(express.static(path.join(__dirname, 'static')));

// Proxy API requests to Flask backend
app.use('/upload', createProxyMiddleware({ target: `http://localhost:${FLASK_PORT}`, changeOrigin: true }));
app.use('/analyze_cloud', createProxyMiddleware({ target: `http://localhost:${FLASK_PORT}`, changeOrigin: true }));
app.use('/analyze_existing', createProxyMiddleware({ target: `http://localhost:${FLASK_PORT}`, changeOrigin: true }));
app.use('/files', createProxyMiddleware({ target: `http://localhost:${FLASK_PORT}`, changeOrigin: true }));
app.use('/serve_file', createProxyMiddleware({ target: `http://localhost:${FLASK_PORT}`, changeOrigin: true }));
app.use('/health', createProxyMiddleware({ target: `http://localhost:${FLASK_PORT}`, changeOrigin: true }));

// Serve the frontend HTML directly
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

async function bootstrap() {
  NODE_PORT = await decideNodePort();
  if (!process.env.FLASK_PORT) {
    // Keep flask separated if same as node
    if (String(FLASK_PORT) === String(NODE_PORT)) {
      FLASK_PORT = parseInt(NODE_PORT) + 100; // offset
    }
  }
  process.env.FLASK_PORT = String(FLASK_PORT);
  try {
    await startFlask();
  } catch (e) {
    console.error('[WARN] Could not confirm Flask readiness:', e.message);
  }
  app.listen(NODE_PORT, () => {
    console.log(`Node.js server running on port ${NODE_PORT}`);
    console.log(`Proxying API requests to Flask backend on port ${FLASK_PORT}`);
  });
}

bootstrap();

// Local usage without env vars:
//   node server.js
// Visit: http://localhost:3000
// Set PYTHON_CMD if your python executable differs.
// Deployment (Render) example:
//   Build Command: pip install -r requirements_prod.txt && npm install
//   Start Command: node server.js
// Node will spawn Flask automatically.
