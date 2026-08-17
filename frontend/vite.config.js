import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
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
