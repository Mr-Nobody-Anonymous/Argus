import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    base: './',
    plugins: [react()],
    build: {
        rollupOptions: {
            output: {
                manualChunks(id) {
                    if (id.includes('node_modules')) {
                        if (id.includes('react') || id.includes('react-dom') || id.includes('react-router-dom')) {
                            return 'vendor-react';
                        }
                        if (id.includes('@mui') || id.includes('@emotion')) {
                            return 'vendor-mui';
                        }
                        if (id.includes('recharts')) {
                            return 'vendor-charts';
                        }
                        if (id.includes('lucide-react')) {
                            return 'vendor-icons';
                        }
                    }
                },
            },
        },
        chunkSizeWarningLimit: 1200,
    },
    server: {
        port: 3000,
        host: true,
        // Allow reverse-proxied / tunnelled hosts (cloud dev sandboxes, ngrok,
        // LAN access) to load the dev server. Vite blocks unknown hosts by default.
        allowedHosts: true,
        proxy: {
            // `ws: true` is required so the dev server proxies the WebSocket
            // upgrade for /api/ws/stream/{camera_id} instead of only plain HTTP.
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                ws: true
            },
            '/snapshots': {
                target: 'http://localhost:8000',
                changeOrigin: true
            }
        }
    }
})
