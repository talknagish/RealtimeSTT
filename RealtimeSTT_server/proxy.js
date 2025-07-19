const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve static files (the HTML client)
app.use(express.static(path.join(__dirname)));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Proxy server is running' });
});

// Proxy WebSocket connections to the STT server
app.use('/ws', createProxyMiddleware({
  target: 'https://stt.talknagish.com',
  changeOrigin: true,
  ws: true,
  secure: false, // Disable SSL verification
  pathRewrite: {
    '^/ws': '', // Remove /ws prefix when forwarding
  },
  onError: (err, req, res) => {
    console.error('Proxy error:', err);
  },
  onProxyReqWs: (proxyReq, req, socket) => {
    console.log('Proxying WebSocket connection to:', req.url);
  },
  onProxyRes: (proxyRes, req, res) => {
    console.log('Proxy response:', proxyRes.statusCode);
  }
}));

// Also handle direct WebSocket upgrade requests
app.use('/control', createProxyMiddleware({
  target: 'wss://stt.talknagish.com',
  changeOrigin: true,
  ws: true,
  secure: false,
  onError: (err, req, res) => {
    console.error('Control proxy error:', err);
  }
}));

app.use('/data', createProxyMiddleware({
  target: 'wss://stt.talknagish.com',
  changeOrigin: true,
  ws: true,
  secure: false,
  onError: (err, req, res) => {
    console.error('Data proxy error:', err);
  }
}));

app.listen(PORT, () => {
  console.log(`🚀 Proxy server running on http://localhost:${PORT}`);
  console.log(`📱 Open http://localhost:${PORT} in your browser`);
  console.log(`🔗 WebSocket proxy: ws://localhost:${PORT}/ws`);
  console.log(`🔗 Direct control: ws://localhost:${PORT}/control`);
  console.log(`🔗 Direct data: ws://localhost:${PORT}/data`);
}); 