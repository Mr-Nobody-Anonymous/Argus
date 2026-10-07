/**
 * A single video-wall tile: live MJPEG-over-WebSocket frames with detection
 * overlays drawn on a canvas above them.
 *
 * Protocol (unchanged, defined by the backend):
 *   binary Blob -> JPEG frame
 *   text JSON   -> {camera_id, detections:[{track_id, class, confidence,
 *                   bbox:{x1,y1,x2,y2}}], timestamp}
 *
 * Three things this component is careful about:
 *
 * **Overlay geometry.** Boxes arrive in source-frame pixels. The canvas is
 * sized to the element's *displayed* size and coordinates are scaled by
 * naturalWidth/Height, so overlays stay locked to the image at any tile size.
 * The canvas is also sized in device pixels and scaled back down, otherwise
 * strokes blur on HiDPI screens.
 *
 * **Liveness is measured, not assumed.** An open WebSocket does not mean
 * frames are arriving - a stalled camera holds the socket open indefinitely.
 * The tile tracks the last frame's arrival time and degrades LIVE -> STALE ->
 * OFFLINE on its own clock, so a frozen feed announces itself instead of
 * quietly showing an old picture.
 *
 * **Reconnect backs off.** A fixed 3 s retry against a downed backend is a
 * request flood from every tile at once; this backs off to 15 s and stops
 * entirely once unmounted.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Box, Typography, Tooltip, IconButton } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { Fullscreen, VideocamOff, CenterFocusStrong } from '@mui/icons-material';
import { buildStreamUrl } from '../services/api';
import { C, MONO } from '../theme';
import { StatusDot, Tag, HudReticle } from './ui';

// A feed is stale if no frame has arrived within this window, and considered
// offline beyond the second. Both are generous relative to the 4-10 fps the
// backend actually produces on CPU.
const STALE_AFTER_MS = 2500;
const OFFLINE_AFTER_MS = 8000;

const CLASS_COLORS = {
    person: C.signal,
    car: C.accent,
    truck: C.accent,
    bus: C.accent,
    motorcycle: C.accent,
    bicycle: C.accent,
    backpack: C.warn,
    handbag: C.warn,
    suitcase: C.warn,
};
const colorForClass = (cls) => CLASS_COLORS[cls] || C.ok;

export default function CameraTile({
    camera,
    zones = [],
    height = 260,
    onExpand,
    showOverlays = true,
    compact = false,
}) {
    const cameraId = camera?.id;
    const [detections, setDetections] = useState([]);
    const [phase, setPhase] = useState('connecting'); // connecting|live|stale|offline
    const [fps, setFps] = useState(null);

    const wsRef = useRef(null);
    const imgRef = useRef(null);
    const canvasRef = useRef(null);
    const rafRef = useRef(null);
    const lastFrameAt = useRef(0);
    const frameTimes = useRef([]);
    const objectUrlRef = useRef(null);
    const retryRef = useRef(0);
    const retryTimer = useRef(null);
    const overlaysOn = useRef(showOverlays);
    overlaysOn.current = showOverlays;

    // ── Drawing ──────────────────────────────────────────────────────────────
    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img || !img.naturalWidth) return;

        const rect = img.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        // Match the backing store to device pixels; keep CSS size logical.
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(rect.width * dpr) ||
            canvas.height !== Math.round(rect.height * dpr)) {
            canvas.width = Math.round(rect.width * dpr);
            canvas.height = Math.round(rect.height * dpr);
            canvas.style.width = `${rect.width}px`;
            canvas.style.height = `${rect.height}px`;
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        if (!overlaysOn.current) return;

        // The frame is rendered with object-fit: contain, so the picture does
        // not fill the element - it is letterboxed. Scaling by the element
        // rect would smear every box across the black bars and misplace it by
        // the offset. Compute the actual drawn content box instead.
        const scale = Math.min(rect.width / img.naturalWidth,
            rect.height / img.naturalHeight);
        const drawnW = img.naturalWidth * scale;
        const drawnH = img.naturalHeight * scale;
        const offX = (rect.width - drawnW) / 2;
        const offY = (rect.height - drawnH) / 2;
        // Map a source-frame pixel to a canvas pixel.
        const px = (v) => offX + v * scale;
        const py = (v) => offY + v * scale;
        const sx = scale;
        const sy = scale;

        // Zones first, so detections draw on top of them.
        zones.forEach((zone) => {
            const coords = zone.coordinates || [];
            if (coords.length < 2) return;
            ctx.beginPath();
            if (zone.type === 'rectangle' && coords.length === 2) {
                ctx.rect(px(coords[0][0]), py(coords[0][1]),
                    (coords[1][0] - coords[0][0]) * sx,
                    (coords[1][1] - coords[0][1]) * sy);
            } else if (zone.type === 'line') {
                coords.forEach((pt, i) => (i ? ctx.lineTo(px(pt[0]), py(pt[1]))
                    : ctx.moveTo(px(pt[0]), py(pt[1]))));
            } else {
                coords.forEach((pt, i) => (i ? ctx.lineTo(px(pt[0]), py(pt[1]))
                    : ctx.moveTo(px(pt[0]), py(pt[1]))));
                ctx.closePath();
            }
            ctx.strokeStyle = alpha(C.warn, 0.85);
            ctx.lineWidth = 1.5;
            ctx.setLineDash([6, 4]);
            ctx.stroke();
            if (zone.type !== 'line') {
                ctx.fillStyle = alpha(C.warn, 0.08);
                ctx.fill();
            }
            ctx.setLineDash([]);
            if (zone.name) {
                ctx.font = `600 10px ${MONO}`;
                ctx.fillStyle = alpha(C.warn, 0.95);
                ctx.fillText(String(zone.name).toUpperCase(),
                    px(coords[0][0]) + 4, py(coords[0][1]) - 5);
            }
        });

        // Detections as corner brackets rather than full rectangles: they
        // occlude far less of the subject, which matters when an operator is
        // trying to identify the person inside the box.
        detections.forEach((det) => {
            const b = det.bbox || {};
            const x = px(b.x1 ?? 0);
            const y = py(b.y1 ?? 0);
            const w = ((b.x2 ?? 0) - (b.x1 ?? 0)) * sx;
            const h = ((b.y2 ?? 0) - (b.y1 ?? 0)) * sy;
            if (w <= 0 || h <= 0) return;

            const col = colorForClass(det.class);
            const arm = Math.max(8, Math.min(w, h) * 0.22);

            ctx.strokeStyle = col;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'miter';
            ctx.beginPath();
            ctx.moveTo(x, y + arm); ctx.lineTo(x, y); ctx.lineTo(x + arm, y);
            ctx.moveTo(x + w - arm, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + arm);
            ctx.moveTo(x + w, y + h - arm); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - arm, y + h);
            ctx.moveTo(x + arm, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - arm);
            ctx.stroke();

            ctx.fillStyle = alpha(col, 0.06);
            ctx.fillRect(x, y, w, h);

            if (!compact) {
                const conf = Number.isFinite(det.confidence)
                    ? ` ${Math.round(det.confidence * 100)}%` : '';
                const id = det.track_id != null ? `#${det.track_id} ` : '';
                const text = `${id}${String(det.class || 'object').toUpperCase()}${conf}`;
                ctx.font = `700 10px ${MONO}`;
                const tw = ctx.measureText(text).width;
                const ty = y > 16 ? y - 15 : y + h + 3;
                ctx.fillStyle = alpha('#000', 0.82);
                ctx.fillRect(x, ty, tw + 10, 14);
                ctx.fillStyle = col;
                ctx.fillRect(x, ty, 2, 14);
                ctx.fillStyle = col;
                ctx.fillText(text, x + 6, ty + 10.5);
            }
        });
    }, [detections, zones, compact]);

    // The transport effect must NOT depend on `draw`: draw is recreated on
    // every detection update, and listing it as a dependency tore down and
    // rebuilt the WebSocket ~20 times a second. The socket reconnected so
    // fast it almost never delivered a frame, leaving the overlay blank.
    // Hold the latest draw in a ref so the transport can call it without
    // re-subscribing.
    const drawRef = useRef(draw);
    useEffect(() => { drawRef.current = draw; draw(); }, [draw]);

    // Redraw on resize; the tile is inside a responsive grid.
    useEffect(() => {
        const onResize = () => {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = requestAnimationFrame(() => drawRef.current());
        };
        window.addEventListener('resize', onResize);
        return () => window.removeEventListener('resize', onResize);
    }, []);

    // ── Transport ────────────────────────────────────────────────────────────
    useEffect(() => {
        if (!cameraId) return undefined;
        let alive = true;

        const connect = () => {
            if (!alive) return;
            let ws;
            try {
                ws = new WebSocket(buildStreamUrl(cameraId));
            } catch {
                schedule();
                return;
            }
            ws.binaryType = 'blob';
            wsRef.current = ws;

            ws.onopen = () => { if (alive) retryRef.current = 0; };

            ws.onmessage = (evt) => {
                if (!alive) return;
                if (evt.data instanceof Blob) {
                    const url = URL.createObjectURL(evt.data);
                    const img = imgRef.current;
                    if (!img) { URL.revokeObjectURL(url); return; }
                    img.onload = () => {
                        // Revoke the *previous* url only after the new frame is
                        // decoded, or the visible image can be torn down mid-paint.
                        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
                        objectUrlRef.current = url;
                        const now = performance.now();
                        lastFrameAt.current = now;
                        frameTimes.current.push(now);
                        if (frameTimes.current.length > 12) frameTimes.current.shift();
                        cancelAnimationFrame(rafRef.current);
                        rafRef.current = requestAnimationFrame(() => drawRef.current());
                    };
                    img.onerror = () => URL.revokeObjectURL(url);
                    img.src = url;
                } else if (typeof evt.data === 'string') {
                    try {
                        const meta = JSON.parse(evt.data);
                        if (Array.isArray(meta.detections)) setDetections(meta.detections);
                    } catch { /* a malformed frame must not kill the stream */ }
                }
            };

            ws.onerror = () => { try { ws.close(); } catch { /* already closing */ } };
            ws.onclose = () => { if (alive) schedule(); };
        };

        const schedule = () => {
            if (!alive) return;
            retryRef.current = Math.min(retryRef.current + 1, 5);
            const delay = Math.min(1000 * 2 ** (retryRef.current - 1), 15000);
            clearTimeout(retryTimer.current);
            retryTimer.current = setTimeout(connect, delay);
        };

        connect();
        return () => {
            alive = false;
            clearTimeout(retryTimer.current);
            cancelAnimationFrame(rafRef.current);
            try { wsRef.current?.close(); } catch { /* noop */ }
            if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
        };
    }, [cameraId]);

    // Liveness clock, independent of socket state.
    useEffect(() => {
        const tick = setInterval(() => {
            const since = performance.now() - lastFrameAt.current;
            if (!lastFrameAt.current) return; // still waiting for frame one
            setPhase(since > OFFLINE_AFTER_MS ? 'offline'
                : since > STALE_AFTER_MS ? 'stale' : 'live');

            const t = frameTimes.current;
            if (t.length > 2 && since < STALE_AFTER_MS) {
                const span = (t[t.length - 1] - t[0]) / 1000;
                setFps(span > 0 ? (t.length - 1) / span : null);
            } else {
                setFps(null);
            }
        }, 700);
        return () => clearInterval(tick);
    }, []);

    const phaseMeta = {
        connecting: { color: C.textFaint, label: 'CONNECTING' },
        live: { color: C.ok, label: 'LIVE' },
        stale: { color: C.warn, label: 'STALE' },
        offline: { color: C.danger, label: 'NO SIGNAL' },
    }[phase];

    const people = detections.filter((d) => d.class === 'person').length;

    return (
        <Box
            sx={{
                position: 'relative',
                height,
                borderRadius: 2,
                overflow: 'hidden',
                background: '#000',
                border: `1px solid ${phase === 'live' ? alpha(C.signal, 0.4) : C.line}`,
                boxShadow: phase === 'live' ? `0 0 16px ${alpha(C.signal, 0.12)}` : 'none',
                transition: 'border-color .3s, box-shadow .3s',
                '&:hover': {
                    borderColor: C.signal,
                    boxShadow: `0 0 24px ${alpha(C.signal, 0.35)}, 0 8px 30px ${alpha('#000', 0.8)}`,
                },
                '&:hover .tile-actions': { opacity: 1 },
            }}
        >
            <HudReticle color={phase === 'live' ? C.signal : C.lineHi} size={10} stroke={1.5} opacity={phase === 'live' ? 0.75 : 0.4} />

            <Box
                component="img"
                ref={imgRef}
                alt={`${camera?.name || 'camera'} live feed`}
                sx={{
                    width: '100%', height: '100%', objectFit: 'contain',
                    display: 'block',
                    opacity: phase === 'offline' ? 0.25 : 1,
                    filter: phase === 'stale' ? 'grayscale(0.65)' : 'none',
                    transition: 'opacity .4s, filter .4s',
                }}
            />
            <Box
                component="canvas"
                ref={canvasRef}
                sx={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
            />

            {/* Scanline sweep: a purely cosmetic cue that the tile is live.
                Suppressed when it is not, so it can never imply liveness. */}
            {phase === 'live' && (
                <Box sx={{
                    position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden',
                    '&::after': {
                        content: '""', position: 'absolute', top: 0, bottom: 0, width: '35%',
                        background: `linear-gradient(90deg, transparent, ${alpha(C.signal, 0.05)}, transparent)`,
                        animation: 'argus-sweep 5.5s linear infinite',
                    },
                }} />
            )}

            {/* Header strip */}
            <Box sx={{
                position: 'absolute', top: 0, left: 0, right: 0,
                display: 'flex', alignItems: 'center', gap: 1, px: 1.25, py: 0.9,
                background: `linear-gradient(180deg, ${alpha('#000', 0.88)}, transparent)`,
                zIndex: 2,
            }}>
                <StatusDot color={phaseMeta.color} pulse={phase === 'live'} title={phaseMeta.label} />
                <Typography sx={{
                    fontFamily: MONO, fontSize: 11, fontWeight: 700,
                    color: C.text, textShadow: '0 1px 3px #000', minWidth: 0,
                }} noWrap>
                    {camera?.name || `CAM ${cameraId}`}
                </Typography>
                {phase === 'live' && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.4, px: 0.6, py: 0.1, borderRadius: 0.5, bgcolor: alpha(C.critical, 0.18), border: `1px solid ${alpha(C.critical, 0.4)}` }}>
                        <Box sx={{ width: 5, height: 5, borderRadius: '50%', bgcolor: C.critical, animation: 'argus-pulse 1.2s infinite' }} />
                        <Typography sx={{ fontFamily: MONO, fontSize: 8.5, fontWeight: 800, color: C.critical, letterSpacing: '0.08em' }}>
                            REC
                        </Typography>
                    </Box>
                )}
                {camera?.location_tag && (
                    <Typography sx={{
                        fontFamily: MONO, fontSize: 9.5, color: C.textDim,
                        textShadow: '0 1px 3px #000',
                    }} noWrap>
                        {camera.location_tag}
                    </Typography>
                )}
                <Box sx={{ flex: 1 }} />
                <Tag label={phaseMeta.label} color={phaseMeta.color} />
            </Box>

            {/* Footer telemetry */}
            <Box sx={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                display: 'flex', alignItems: 'center', gap: 1, px: 1.25, py: 0.9,
                background: `linear-gradient(0deg, ${alpha('#000', 0.9)}, transparent)`,
            }}>
                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim }}>
                    {fps != null ? `${fps.toFixed(1)} FPS` : '—.— FPS'}
                </Typography>
                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }}>·</Typography>
                <Tooltip title="Objects currently tracked in frame">
                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: detections.length ? C.signal : C.textFaint }}>
                        {detections.length} TRACKED
                    </Typography>
                </Tooltip>
                {people > 0 && (
                    <>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }}>·</Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.signal }}>
                            {people} PERSON{people > 1 ? 'S' : ''}
                        </Typography>
                    </>
                )}
            </Box>

            {/* Hover actions */}
            <Box className="tile-actions" sx={{
                position: 'absolute', top: 34, right: 8, display: 'flex',
                flexDirection: 'column', gap: 0.5, opacity: 0, transition: 'opacity .2s',
            }}>
                {onExpand && (
                    <Tooltip title="Expand feed" placement="left">
                        <IconButton size="small" onClick={() => onExpand(camera)}
                            sx={{
                                bgcolor: alpha('#000', 0.6), color: C.text,
                                '&:hover': { bgcolor: alpha(C.signal, 0.25) },
                            }}>
                            <Fullscreen sx={{ fontSize: 16 }} />
                        </IconButton>
                    </Tooltip>
                )}
            </Box>

            {phase === 'offline' && (
                <Box sx={{
                    position: 'absolute', inset: 0, display: 'flex',
                    flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 1,
                }}>
                    <VideocamOff sx={{ fontSize: 30, color: alpha(C.danger, 0.8) }} />
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.danger, letterSpacing: '0.12em' }}>
                        NO SIGNAL
                    </Typography>
                    <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint }}>
                        reconnecting…
                    </Typography>
                </Box>
            )}

            {phase === 'connecting' && !lastFrameAt.current && (
                <Box sx={{
                    position: 'absolute', inset: 0, display: 'flex',
                    flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 1,
                }}>
                    <CenterFocusStrong sx={{
                        fontSize: 26, color: alpha(C.signal, 0.7),
                        animation: 'argus-pulse 1.6s ease-in-out infinite',
                    }} />
                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, letterSpacing: '0.12em' }}>
                        ACQUIRING FEED
                    </Typography>
                </Box>
            )}
        </Box>
    );
}
