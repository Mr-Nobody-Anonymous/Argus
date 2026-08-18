import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    build: {
        rollupOptions: {
            output: {
                // Split the vendor libraries out of the app bundle. A single
                // 1.1 MB chunk had to be re-downloaded in full after any code
                // change and made the first paint slow on a control-room
                // machine; these split by change frequency, so MUI and charts
                // stay cached while app code iterates.
                manualChunks: {
                    react: ['react', 'react-dom', 'react-router-dom'],
                    mui: ['@mui/material', '@mui/icons-material', '@emotion/react', '@emotion/styled'],
                    charts: ['recharts'],
                },
            },
        },
        chunkSizeWarningLimit: 700,
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
