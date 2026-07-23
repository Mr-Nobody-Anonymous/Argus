/**
 * Live Video Player Component
 * Renders camera feed with AI bounding box overlays via WebSocket
 * 
 * Protocol:
 * - Binary message (Blob): JPEG-encoded video frame
 * - Text message (JSON): Detection metadata {camera_id, detections, timestamp}
 *   Each detection: {track_id, class, confidence, bbox: {x1, y1, x2, y2}}
 * - Zone polygons are rendered from zoneAPI if provided via zones prop
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


export default function LiveVideoPlayer({ cameraId, detections: propDetections, zones = [] }) {
    const [detections, setDetections] = useState([]);
    const [connected, setConnected] = useState(false);
    const [loading, setLoading] = useState(true);
    const wsRef = useRef(null);
    const canvasRef = useRef(null);
    const imgRef = useRef(null);
    const animationFrameRef = useRef(null);

    // Initialize WebSocket connection
    useEffect(() => {
        let isMounted = true;

        const connectWebSocket = () => {
            // Connect to the API-prefixed WebSocket endpoint
            // The Vite dev server does not proxy WebSocket, so we connect directly
            const ws = new WebSocket(`ws://localhost:8000/api/ws/stream/${cameraId}`);

            ws.onopen = () => {
                if (isMounted) {
                    setConnected(true);
                    setLoading(false);
                }
            };

            ws.onmessage = (event) => {
                if (!isMounted) return;

                if (event.data instanceof Blob) {
                    // ── Binary message: JPEG frame ──
                    const blob = event.data;
                    const url = URL.createObjectURL(blob);
                    if (imgRef.current) {
                        imgRef.current.onload = () => {
                            URL.revokeObjectURL(url);
                            // Repaint overlays after new frame loads
                            if (animationFrameRef.current) {
                                cancelAnimationFrame(animationFrameRef.current);
                            }
                            animationFrameRef.current = requestAnimationFrame(drawOverlays);
                        };
                        imgRef.current.src = url;
                    }
                } else if (typeof event.data === 'string') {
                    // ── Text message: JSON detection metadata ──
                    try {
                        const metadata = JSON.parse(event.data);
                        if (metadata.detections) {
                            setDetections(metadata.detections);
                        }
                        if (metadata.error) {
                            console.warn(`WebSocket error for camera ${cameraId}:`, metadata.error);
                        }
                    } catch (err) {
                        console.warn('Failed to parse WebSocket metadata:', err);
                    }
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
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [cameraId]);

    // Update detections from props (if passed from parent)
    useEffect(() => {
        if (propDetections) {
            setDetections(propDetections);
        }
    }, [propDetections]);

    // Draw bounding boxes, labels, and zone polygons onto the canvas overlay
    const drawOverlays = useCallback(() => {
        const canvas = canvasRef.current;
        const img = imgRef.current;

        if (!canvas || !img) return;

        const ctx = canvas.getContext('2d');
        const rect = img.getBoundingClientRect();

        // Set canvas size to match video element
        canvas.width = rect.width;
        canvas.height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // ── 1. Draw zone polygons ──
        if (zones && zones.length > 0) {
            zones.forEach((zone) => {
                const coords = zone.coordinates || [];
                if (coords.length < 2) return;

                const scaleX = canvas.width / (img.naturalWidth || 1);
                const scaleY = canvas.height / (img.naturalHeight || 1);

                ctx.beginPath();
                if (zone.type === 'rectangle' && coords.length === 2) {
                    // Rectangle: two corner points
                    const x1 = coords[0][0] * scaleX;
                    const y1 = coords[0][1] * scaleY;
                    const x2 = coords[1][0] * scaleX;
                    const y2 = coords[1][1] * scaleY;
                    ctx.rect(x1, y1, x2 - x1, y2 - y1);
                } else {
                    // Polygon: array of [x, y] vertices
                    coords.forEach((point, i) => {
                        const px = point[0] * scaleX;
                        const py = point[1] * scaleY;
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    });
                    ctx.closePath();
                }

                ctx.strokeStyle = 'rgba(255, 165, 0, 0.7)';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 3]);
                ctx.stroke();

                // Zone label
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(255, 165, 0, 0.2)';
                ctx.fill();

                ctx.fillStyle = 'rgba(255, 165, 0, 0.9)';
                ctx.font = '12px Arial';
                ctx.fillText(zone.name || 'Zone', coords[0][0] * scaleX + 5, coords[0][1] * scaleY - 5);
            });
        }

        // ── 2. Draw detection bounding boxes and labels ──
        if (detections && detections.length > 0) {
            detections.forEach((det) => {
                const bbox = det.bbox;
                if (!bbox) return;

                const scaleX = canvas.width / (img.naturalWidth || 1);
                const scaleY = canvas.height / (img.naturalHeight || 1);

                // Support both {x1, y1, x2, y2} object format and [x1, y1, x2, y2] array format
                let x1, y1, x2, y2;
                if (Array.isArray(bbox)) {
                    [x1, y1, x2, y2] = bbox;
                } else {
                    x1 = bbox.x1 || bbox[0];
                    y1 = bbox.y1 || bbox[1];
                    x2 = bbox.x2 || bbox[2];
                    y2 = bbox.y2 || bbox[3];
                }

                const rx1 = x1 * scaleX;
                const ry1 = y1 * scaleY;
                const rx2 = x2 * scaleX;
                const ry2 = y2 * scaleY;

                const color = getColorForClass(det.class);

                // Draw bounding box
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
                ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);

                // Draw label background
                const trackLabel = det.track_id ? `#${det.track_id}` : '';
                const label = `${det.class}${trackLabel} ${(det.confidence || 0).toFixed(2)}`;
                ctx.font = 'bold 13px Consolas, monospace';
                const textWidth = ctx.measureText(label).width;

                ctx.fillStyle = color;
                ctx.fillRect(rx1, ry1 - 22, textWidth + 12, 22);

                // Draw label text
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, rx1 + 6, ry1 - 7);

                // Draw small corner indicators for track points
                if (det.track_id) {
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(rx1 + 4, ry1 + 4, 3, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(rx2 - 4, ry2 - 4, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }
    }, [detections, zones]);

    // Redraw whenever detections or zones change, and on canvas resize
    useEffect(() => {
        drawOverlays();
    }, [detections, zones, drawOverlays]);

    const getColorForClass = (className) => {
        const colors = {
            person: '#00ff00',
            car: '#ff0000',
            truck: '#ff6600',
            bicycle: '#0066ff',
            motorcycle: '#0099ff',
            bus: '#ff3300',
            dog: '#00ff88',
            cat: '#ff00ff',
            backpack: '#ffff00',
            suitcase: '#ffaa00',
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
