/**
 * Live Video Player Component
 * Renders camera feed with AI bounding box overlays via WebSocket
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

const VideoContainer = styled(Box)(({ theme }) => ({
    position: 'relative',
    width: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    borderRadius: theme.spacing(1),
    overflow: 'hidden',
}));

const OverlayCanvas = styled('canvas')({
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
});

const VideoElement = styled('img')({
    width: '100%',
    height: 'auto',
    display: 'block',
});


export default function LiveVideoPlayer({ cameraId, detections: propDetections }) {
    const [detections, setDetections] = useState([]);
    const [connected, setConnected] = useState(false);
    const [loading, setLoading] = useState(true);
    const wsRef = useRef(null);
    const canvasRef = useRef(null);
    const imgRef = useRef(null);

    // Initialize WebSocket connection
    useEffect(() => {
        let isMounted = true;
        
        const connectWebSocket = () => {
            const ws = new WebSocket(`ws://localhost:8000/ws/stream/${cameraId}`);
            
            ws.onopen = () => {
                if (isMounted) {
                    setConnected(true);
                    setLoading(false);
                }
            };
            
            ws.onmessage = (event) => {
                if (!isMounted) return;
                
                const arrayBuffer = event.data;
                const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                const url = URL.createObjectURL(blob);
                
                if (imgRef.current) {
                    imgRef.current.src = url;
                    URL.revokeObjectURL(url); // Clean up
                }
            };
            
            ws.onerror = () => {
                if (isMounted) {
                    setConnected(false);
                }
            };
            
            ws.onclose = () => {
                if (isMounted) {
                    setConnected(false);
                    setLoading(true);
                    // Reconnect after 3 seconds
                    setTimeout(connectWebSocket, 3000);
                }
            };
            
            wsRef.current = ws;
        };

        connectWebSocket();

        return () => {
            isMounted = false;
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [cameraId]);

    // Update detections from props or WebSocket
    useEffect(() => {
        if (propDetections) {
            setDetections(propDetections);
        }
    }, [propDetections]);

    // Draw bounding boxes overlay
    const drawOverlays = useCallback(() => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        
        if (!canvas || !img || !detections.length) return;
        
        const ctx = canvas.getContext('2d');
        const rect = img.getBoundingClientRect();
        
        // Set canvas size to match video
        canvas.width = rect.width;
        canvas.height = rect.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw each detection
        detections.forEach((det) => {
            const bbox = det.bbox;
            const scaleX = canvas.width / img.naturalWidth;
            const scaleY = canvas.height / img.naturalHeight;
            
            const x1 = bbox.x1 * scaleX;
            const y1 = bbox.y1 * scaleY;
            const x2 = bbox.x2 * scaleX;
            const y2 = bbox.y2 * scaleY;
            
            // Draw bounding box
            ctx.strokeStyle = getColorForClass(det.class);
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw label background
            ctx.fillStyle = getColorForClass(det.class);
            ctx.font = '14px Arial';
            const label = `${det.class} ${det.confidence.toFixed(2)}`;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);
            
            // Draw label text
            ctx.fillStyle = '#fff';
            ctx.fillText(label, x1 + 5, y1 - 5);
        });
    }, [detections]);

    // Redraw on detections change
    useEffect(() => {
        drawOverlays();
    }, [detections, drawOverlays]);

    const getColorForClass = (className) => {
        const colors = {
            person: '#00ff00',
            car: '#ff0000',
            truck: '#ff6600',
            bicycle: '#0066ff',
            motorcycle: '#0099ff',
            bus: '#ff3300',
        };
        return colors[className] || '#ffff00';
    };

    return (
        <VideoContainer>
            {loading && (
                <Box 
                    display="flex" 
                    alignItems="center" 
                    justifyContent="center" 
                    minHeight={200}
                >
                    <CircularProgress />
                </Box>
            )}
            
            <VideoElement 
                ref={imgRef} 
                style={{ display: loading ? 'none' : 'block' }}
                alt={`Camera ${cameraId}`}
            />
            
            <OverlayCanvas ref={canvasRef} />
            
            {!connected && !loading && (
                <Box 
                    position="absolute" 
                    top={16} 
                    right={16} 
                    bgcolor="error.main" 
                    color="white"
                    px={1}
                    borderRadius={1}
                >
                    <Typography variant="caption">Disconnected</Typography>
                </Box>
            )}
        </VideoContainer>
    );
}