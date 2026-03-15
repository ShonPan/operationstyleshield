import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5002',
        changeOrigin: true,
        // Ensure SSE streams are not buffered by the proxy
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes, req, res) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              // Disable any proxy-level buffering for SSE
              res.setHeader('X-Accel-Buffering', 'no')
              res.setHeader('Cache-Control', 'no-cache, no-transform')
            }
          })
        },
      },
    },
  },
})
