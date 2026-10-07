/**
 * CityOS - Intersection Intelligence dashboard.
 *
 * A live digital twin of each camera's intersection: road users rendered on a
 * top-down canvas with trajectories and velocity vectors, safety alerts
 * (wrong-way / near-miss / VRU), traffic-flow charts, and a signal-optimiser
 * panel with adaptive recommendations and manual override.
 *
 * Polling cadence: twin 1 s (it is the live model), alerts 4 s.
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box, Typography, Grid, Card, CardContent, Chip, ToggleButtonGroup,
    ToggleButton, Tooltip, CircularProgress, Alert as MuiAlert, Divider,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
    ResponsiveContainer, Legend,
} from 'recharts';
import { cityosAPI, cameraAPI } from '../services/api';
import { C, MONO, glass, priorityColor } from '../theme';
import { StatusDot, HudReticle } from '../components/ui';

// ── Road-user visual language ────────────────────────────────────────────────
const CATEGORY_STYLE = {
    vehicle:    { color: '#f43f5e', label: 'Vehicle',    size: 7 },
    truck:      { color: '#fb923c', label: 'Truck',      size: 9 },
    bus:        { color: '#fbbf24', label: 'Bus',        size: 9 },
    motorcycle: { color: '#f472b6', label: 'Moto',       size: 6 },
    cyclist:    { color: '#34d399', label: 'Cyclist',    size: 5 },
    pedestrian: { color: '#22d3ee', label: 'Pedestrian', size: 4 },
    other:      { color: '#8b9bb0', label: 'Other',      size: 5 },
};

const SIGNAL_STATE_COLOR = {
    green: C.ok,
    yellow: C.warn,
    all_red: C.danger,
};

function timeLabel(ts) {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
}

// ── Digital twin canvas ──────────────────────────────────────────────────────
function TwinCanvas({ objects, signal }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;

        // Background - tactical dark void
        ctx.fillStyle = '#030712';
        ctx.fillRect(0, 0, W, H);

        const cx = W / 2;
        const cy = H / 2;

        // Subtle radar range concentric rings
        ctx.strokeStyle = alpha(C.signal, 0.08);
        ctx.lineWidth = 1;
        [60, 120, 180, 240].forEach((r) => {
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
        });

        // Roads: two crossing corridors through the middle
        const roadW = Math.min(W, H) * 0.22;
        ctx.fillStyle = '#0b1120';
        ctx.fillRect(cx - roadW / 2, 0, roadW, H);          // N-S road
        ctx.fillRect(0, cy - roadW / 2, W, roadW);          // E-W road

        // Fine grid lines on road
        ctx.strokeStyle = alpha(C.signal, 0.12);
        ctx.lineWidth = 0.75;
        ctx.strokeRect(cx - roadW / 2, 0, roadW, H);
        ctx.strokeRect(0, cy - roadW / 2, W, roadW);

        // Center lane markings
        ctx.strokeStyle = alpha(C.signal, 0.35);
        ctx.setLineDash([8, 10]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
        ctx.moveTo(0, cy); ctx.lineTo(W, cy);
        ctx.stroke();
        ctx.setLineDash([]);

        // Intersection box highlight with cyber glow
        ctx.strokeStyle = alpha(C.signal, 0.5);
        ctx.lineWidth = 1.5;
        ctx.strokeRect(cx - roadW / 2, cy - roadW / 2, roadW, roadW);

        // Crosswalk stripes at the four edges of the junction
        ctx.fillStyle = alpha(C.signal, 0.25);
        const stripe = 4;
        for (let i = -3; i <= 3; i++) {
            const off = i * stripe * 2.4;
            ctx.fillRect(cx + off, cy - roadW / 2 - 14, stripe, 10);
            ctx.fillRect(cx + off, cy + roadW / 2 + 4, stripe, 10);
            ctx.fillRect(cx - roadW / 2 - 14, cy + off, 10, stripe);
            ctx.fillRect(cx + roadW / 2 + 4, cy + off, 10, stripe);
        }

        // Radar Crosshair Ticks & Angle markers
        ctx.strokeStyle = alpha(C.signal, 0.3);
        ctx.lineWidth = 1;
        const tickLen = 8;
        // North
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, tickLen); ctx.stroke();
        // South
        ctx.beginPath(); ctx.moveTo(cx, H); ctx.lineTo(cx, H - tickLen); ctx.stroke();
        // West
        ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(tickLen, cy); ctx.stroke();
        // East
        ctx.beginPath(); ctx.moveTo(W, cy); ctx.lineTo(W - tickLen, cy); ctx.stroke();

        // Approach labels in tactical monospace font
        ctx.fillStyle = C.signal;
        ctx.font = `700 ${Math.round(W * 0.019)}px ${MONO}`;
        ctx.textAlign = 'center';
        ctx.fillText('N ▲', cx, 20);
        ctx.fillText('S ▼', cx, H - 8);
        ctx.fillText('◀ W', 20, cy + 5);
        ctx.fillText('E ▶', W - 20, cy + 5);

        // Center Signal State Indicator with multi-ring pulse
        if (signal) {
            const sc = SIGNAL_STATE_COLOR[signal.state] || C.textDim;
            ctx.beginPath();
            ctx.arc(cx, cy, 18, 0, Math.PI * 2);
            ctx.fillStyle = alpha(sc, 0.15);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(cx, cy, 14, 0, Math.PI * 2);
            ctx.strokeStyle = sc;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = sc;
            ctx.font = `800 ${Math.round(W * 0.022)}px ${MONO}`;
            ctx.fillText(signal.phase === 'NS' ? '↕' : '↔', cx, cy + 6);
        }

        // Road users: trajectory trail, velocity vector, body, and targeting brackets
        for (const obj of objects || []) {
            const style = CATEGORY_STYLE[obj.category] || CATEGORY_STYLE.other;
            const px = obj.position.x * W;
            const py = obj.position.y * H;

            // Trajectory trail
            const traj = obj.trajectory || [];
            if (traj.length > 1) {
                ctx.strokeStyle = alpha(style.color, 0.45);
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                traj.forEach(([tx, ty], idx) => {
                    const x = tx * W;
                    const y = ty * H;
                    if (idx === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            // Velocity vector (heading arrow, length ~ speed)
            if (obj.speed_mps > 0.2 && obj.heading && obj.heading !== '-') {
                const compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
                const idx = compass.indexOf(obj.heading);
                if (idx >= 0) {
                    const angle = (idx * 45 * Math.PI) / 180;
                    const len = Math.min(32, 8 + obj.speed_mps * 2.4);
                    const hx = px + Math.sin(angle) * len;
                    const hy = py - Math.cos(angle) * len;
                    ctx.strokeStyle = alpha(style.color, 0.95);
                    ctx.lineWidth = 1.75;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(hx, hy);
                    ctx.stroke();
                    // arrowhead
                    ctx.fillStyle = alpha(style.color, 0.95);
                    ctx.beginPath();
                    ctx.arc(hx, hy, 2.5, 0, Math.PI * 2);
                    ctx.fill();
                }
            }

            // Radar targeting reticle around vehicle
            const boxSize = style.size + 4;
            ctx.strokeStyle = alpha(style.color, 0.7);
            ctx.lineWidth = 1;
            const cLen = 3;
            // top-left
            ctx.beginPath(); ctx.moveTo(px - boxSize, py - boxSize + cLen); ctx.lineTo(px - boxSize, py - boxSize); ctx.lineTo(px - boxSize + cLen, py - boxSize); ctx.stroke();
            // top-right
            ctx.beginPath(); ctx.moveTo(px + boxSize - cLen, py - boxSize); ctx.lineTo(px + boxSize, py - boxSize); ctx.lineTo(px + boxSize, py - boxSize + cLen); ctx.stroke();
            // bottom-left
            ctx.beginPath(); ctx.moveTo(px - boxSize, py + boxSize - cLen); ctx.lineTo(px - boxSize, py + boxSize); ctx.lineTo(px - boxSize + cLen, py + boxSize); ctx.stroke();
            // bottom-right
            ctx.beginPath(); ctx.moveTo(px + boxSize - cLen, py + boxSize); ctx.lineTo(px + boxSize, py + boxSize); ctx.lineTo(px + boxSize, py - boxSize + cLen); ctx.stroke();

            // Object Body with intense glowing shadow
            ctx.beginPath();
            ctx.arc(px, py, style.size, 0, Math.PI * 2);
            ctx.fillStyle = style.color;
            ctx.shadowColor = style.color;
            ctx.shadowBlur = 10;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Track id & speed badge
            if (obj.category !== 'pedestrian') {
                ctx.fillStyle = alpha(C.text, 0.9);
                ctx.font = `600 ${Math.round(W * 0.015)}px ${MONO}`;
                ctx.textAlign = 'left';
                const speedText = obj.speed_mps ? ` ${Math.round(obj.speed_mps * 3.6)}km/h` : '';
                ctx.fillText(`#${obj.track_id}${speedText}`, px + boxSize + 3, py - 3);
            }
        }

        // Top-left HUD telemetry banner on canvas
        ctx.fillStyle = alpha(C.void, 0.85);
        ctx.fillRect(8, 8, 220, 38);
        ctx.strokeStyle = alpha(C.signal, 0.4);
        ctx.lineWidth = 1;
        ctx.strokeRect(8, 8, 220, 38);

        ctx.fillStyle = C.signal;
        ctx.font = `700 9px ${MONO}`;
        ctx.textAlign = 'left';
        ctx.fillText('LIVE DIGITAL TWIN // SENSOR FUSION', 14, 20);
        ctx.fillStyle = C.textFaint;
        ctx.font = `500 8px ${MONO}`;
        ctx.fillText('COORD: LOCAL ENU // HOMOGRAPHY MATRIX: LOCKED', 14, 32);
        ctx.fillText(`OBJECTS TRACKED: ${objects?.length || 0}`, 14, 42);

    }, [objects, signal]);

    return (
        <Box sx={{
            position: 'relative',
            border: `1px solid ${alpha(C.signal, 0.35)}`,
            borderRadius: 2,
            overflow: 'hidden',
            background: '#030712',
            boxShadow: `0 8px 32px rgba(0,0,0,0.6), 0 0 24px ${alpha(C.signal, 0.12)}`,
        }}>
            <HudReticle color={C.signal} size={12} stroke={2} opacity={0.8} />
            <canvas
                ref={canvasRef}
                width={720}
                height={480}
                style={{ width: '100%', height: 'auto', display: 'block' }}
            />
            {/* Legend */}
            <Box sx={{
                position: 'absolute', top: 10, right: 10,
                background: alpha(C.panel, 0.88),
                backdropFilter: 'blur(8px)',
                border: `1px solid ${C.line}`,
                borderRadius: 1.5, p: 1,
                display: 'flex', flexDirection: 'column', gap: 0.5,
                zIndex: 2,
            }}>
                {Object.entries(CATEGORY_STYLE).map(([key, s]) => (
                    <Box key={key} sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
                        <Typography sx={{ fontFamily: MONO, fontSize: 9.5, fontWeight: 700, color: C.textDim, letterSpacing: '0.04em' }}>
                            {s.label.toUpperCase()}
                        </Typography>
                    </Box>
                ))}
            </Box>
        </Box>
    );
}

// ── Tactical Stat card ──────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color }) {
    const valColor = color || C.signal;
    return (
        <Card sx={{
            ...glass(0.75, valColor),
            borderRadius: 2,
            position: 'relative',
            overflow: 'hidden',
            transition: 'all 0.2s ease',
            '&:hover': {
                borderColor: alpha(valColor, 0.5),
                boxShadow: `0 8px 24px rgba(0,0,0,0.5), 0 0 16px ${alpha(valColor, 0.15)}`,
            },
            '&::before': {
                content: '""',
                position: 'absolute',
                top: 0, left: 0, right: 0, height: '2px',
                background: `linear-gradient(90deg, transparent, ${valColor}, transparent)`,
            },
        }}>
            <HudReticle color={valColor} size={6} stroke={1.5} opacity={0.4} />
            <CardContent sx={{ py: 1.5, px: 2 }}>
                <Typography variant="overline" sx={{ letterSpacing: '0.12em', color: C.textDim }}>{label}</Typography>
                <Typography sx={{
                    fontFamily: MONO, fontSize: 28, fontWeight: 800,
                    color: valColor, lineHeight: 1.1,
                    textShadow: `0 0 16px ${alpha(valColor, 0.35)}`,
                    fontVariantNumeric: 'tabular-nums',
                }}>
                    {value}
                </Typography>
                {sub && (
                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, mt: 0.4, letterSpacing: '0.04em' }}>
                        {sub}
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
}

// ── Signal panel ─────────────────────────────────────────────────────────────
function SignalPanel({ cameraId, signal, rec, demand, onChanged }) {
    const [busy, setBusy] = useState(false);

    const setMode = async (mode) => {
        setBusy(true);
        try {
            await cityosAPI.setSignalMode(cameraId, mode);
            onChanged?.();
        } catch { /* surfaced via next poll */ }
        setBusy(false);
    };

    const forcePhase = async (phase) => {
        setBusy(true);
        try {
            await cityosAPI.forcePhase(cameraId, phase);
            onChanged?.();
        } catch { /* surfaced via next poll */ }
        setBusy(false);
    };

    if (!signal) return null;
    const stateColor = SIGNAL_STATE_COLOR[signal.state] || C.textDim;

    return (
        <Card sx={{ ...glass(), borderRadius: 2 }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Typography variant="overline">Signal Optimiser</Typography>
                    <Chip
                        size="small"
                        label={signal.mode?.toUpperCase()}
                        sx={{ color: C.accent, borderColor: alpha(C.accent, 0.4) }}
                        variant="outlined"
                    />
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 1.5 }}>
                    <StatusDot color={stateColor} pulse={signal.state === 'green'} size={10} />
                    <Typography sx={{ fontFamily: MONO, fontSize: 15, fontWeight: 700, color: C.text }}>
                        {signal.phase} · {signal.state.replace('_', '-').toUpperCase()}
                    </Typography>
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint }}>
                        {signal.seconds_in_state}s
                    </Typography>
                </Box>

                {rec && (
                    <Box sx={{
                        mt: 1.5, p: 1, borderRadius: 1,
                        background: alpha(C.accent, 0.07),
                        border: `1px solid ${alpha(C.accent, 0.25)}`,
                    }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.accent, letterSpacing: '0.08em' }}>
                            ADAPTIVE RECOMMENDATION
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 11.5, color: C.text, mt: 0.4 }}>
                            {rec.action?.replace('_', ' ')} · suggested green {rec.suggested_green_s}s
                        </Typography>
                        <Typography sx={{ fontSize: 10.5, color: C.textFaint }}>
                            {rec.reason}
                        </Typography>
                    </Box>
                )}

                {demand && Object.keys(demand).length > 0 && (
                    <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mt: 1.25 }}>
                        {Object.entries(demand).map(([approach, val]) => (
                            <Chip key={approach} size="small" variant="outlined"
                                label={`${approach}: ${val}`}
                                sx={{ color: C.textDim, borderColor: C.line }} />
                        ))}
                    </Box>
                )}

                <Divider sx={{ my: 1.5, borderColor: C.line }} />

                <Typography variant="overline" sx={{ mb: 0.75, display: 'block' }}>Mode</Typography>
                <ToggleButtonGroup
                    exclusive size="small" fullWidth
                    value={signal.mode}
                    onChange={(_, v) => v && setMode(v)}
                    disabled={busy}
                    sx={{ '& .MuiToggleButton-root': { fontFamily: MONO, fontSize: 10.5 } }}
                >
                    <ToggleButton value="fixed">FIXED</ToggleButton>
                    <ToggleButton value="adaptive">ADAPTIVE</ToggleButton>
                    <ToggleButton value="manual">MANUAL</ToggleButton>
                </ToggleButtonGroup>

                <Typography variant="overline" sx={{ mt: 1.5, mb: 0.75, display: 'block' }}>
                    Manual Override
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <ToggleButtonGroup
                        exclusive size="small" fullWidth
                        value={null}
                        onChange={(_, v) => v && forcePhase(v)}
                        disabled={busy}
                        sx={{ '& .MuiToggleButton-root': { fontFamily: MONO, fontSize: 10.5 } }}
                    >
                        <ToggleButton value="NS">FORCE NS GREEN</ToggleButton>
                        <ToggleButton value="EW">FORCE EW GREEN</ToggleButton>
                    </ToggleButtonGroup>
                </Box>
            </CardContent>
        </Card>
    );
}

// ── Alerts feed ──────────────────────────────────────────────────────────────
function AlertsFeed({ alerts }) {
    if (!alerts?.length) {
        return (
            <Box sx={{ textAlign: 'center', py: 4 }}>
                <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint }}>
                    NO SAFETY EVENTS RECORDED
                </Typography>
                <Typography sx={{ fontSize: 11, color: C.textFaint, mt: 0.5 }}>
                    Wrong-way, near-miss and VRU conflicts appear here in real time.
                </Typography>
            </Box>
        );
    }
    return (
        <Box sx={{ maxHeight: 340, overflowY: 'auto', pr: 0.5 }}>
            {alerts.map((a) => {
                const pc = priorityColor(a.severity);
                return (
                    <Box key={a.id} sx={{
                        p: 1.1, mb: 0.75, borderRadius: 1.25,
                        border: `1px solid ${alpha(pc, 0.3)}`,
                        background: alpha(pc, 0.06),
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                            <StatusDot color={pc} pulse size={7} />
                            <Typography sx={{
                                fontFamily: MONO, fontSize: 9.5, letterSpacing: '0.1em',
                                color: pc, textTransform: 'uppercase',
                            }}>
                                {a.type.replace('_', ' ')} · {a.severity}
                            </Typography>
                            <Box sx={{ flex: 1 }} />
                            <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint }}>
                                {timeLabel(a.timestamp)}
                            </Typography>
                        </Box>
                        <Typography sx={{ fontSize: 12, color: C.text, mt: 0.5 }}>
                            {a.message}
                        </Typography>
                    </Box>
                );
            })}
        </Box>
    );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function CityOSDashboard() {
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState(null);
    const [twin, setTwin] = useState(null);
    const [alerts, setAlerts] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);

    // Load cameras once, pick the first as default.
    useEffect(() => {
        let alive = true;
        cameraAPI.getAll()
            .then(({ data }) => {
                if (!alive) return;
                const list = data.cameras || [];
                setCameras(list);
                setSelectedCamera((cur) => cur ?? (list[0]?.id ?? null));
            })
            .catch(() => alive && setError('Could not load cameras'))
            .finally(() => alive && setLoading(false));
        return () => { alive = false; };
    }, []);

    // Live twin poll (1 s).
    useEffect(() => {
        if (selectedCamera == null) return undefined;
        let alive = true;
        const poll = () => {
            cityosAPI.twin(selectedCamera)
                .then(({ data }) => alive && setTwin(data))
                .catch(() => alive && setTwin(null));
        };
        poll();
        const t = setInterval(poll, 1000);
        return () => { alive = false; clearInterval(t); };
    }, [selectedCamera]);

    // Alerts poll (4 s).
    useEffect(() => {
        let alive = true;
        const poll = () => {
            cityosAPI.alerts(undefined, 40)
                .then(({ data }) => alive && setAlerts(data.alerts || []))
                .catch(() => {});
        };
        poll();
        const t = setInterval(poll, 4000);
        return () => { alive = false; clearInterval(t); };
    }, []);

    const refreshSignal = useCallback(() => {
        if (selectedCamera == null) return;
        cityosAPI.twin(selectedCamera)
            .then(({ data }) => setTwin(data))
            .catch(() => {});
    }, [selectedCamera]);

    if (loading) {
        return (
            <Box sx={{ display: 'grid', placeItems: 'center', minHeight: 300 }}>
                <CircularProgress size={28} sx={{ color: C.signal }} />
            </Box>
        );
    }

    const counts = twin?.counts_by_category || {};
    const safety = twin?.safety || {};
    const edge = twin?.edge_node || {};
    const volumeSeries = (twin?.flow?.volume_series || []).map((d) => ({
        ...d,
        t: timeLabel(d.minute),
    }));

    return (
        <Box>
            {error && <MuiAlert severity="warning" sx={{ mb: 2 }}>{error}</MuiAlert>}

            {/* Header row: intersection selector + live status */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap', mb: 2 }}>
                <Typography variant="overline" sx={{ letterSpacing: '0.16em' }}>
                    INTERSECTION
                </Typography>
                <ToggleButtonGroup
                    exclusive size="small"
                    value={selectedCamera}
                    onChange={(_, v) => v != null && setSelectedCamera(v)}
                >
                    {cameras.map((c) => (
                        <ToggleButton key={c.id} value={c.id}
                            sx={{ fontFamily: MONO, fontSize: 10.5, px: 1.5 }}>
                            CAM {c.id} · {String(c.name || '').slice(0, 14)}
                        </ToggleButton>
                    ))}
                </ToggleButtonGroup>
                {twin && (
                    <Chip size="small" variant="outlined"
                        label={`${twin.intersection_id}`}
                        sx={{ color: C.signalDim, borderColor: C.line }} />
                )}
                <Box sx={{ flex: 1 }} />
                {twin && (
                    <Tooltip title="Edge-node ingest health">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.85 }}>
                            <StatusDot
                                color={(edge.last_ingest_age_s ?? 99) < 5 ? C.ok : C.warn}
                                pulse size={7}
                            />
                            <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim }}>
                                EDGE · {edge.frames_ingested ?? 0} FRAMES ·{' '}
                                {edge.ingest_latency_ms ?? 0} MS
                            </Typography>
                        </Box>
                    </Tooltip>
                )}
            </Box>

            {!twin ? (
                <Card sx={{ ...glass(), borderRadius: 2, p: 4, textAlign: 'center' }}>
                    <Typography sx={{ fontFamily: MONO, fontSize: 12, color: C.textDim }}>
                        NO DIGITAL TWIN YET FOR THIS CAMERA
                    </Typography>
                    <Typography sx={{ fontSize: 12, color: C.textFaint, mt: 1 }}>
                        The intersection model builds up as soon as the camera pipeline
                        processes frames with tracked road users.
                    </Typography>
                </Card>
            ) : (
                <Grid container spacing={2}>
                    {/* Left column: twin + volume chart */}
                    <Grid item xs={12} lg={8}>
                        <TwinCanvas objects={twin.objects} signal={twin.signal} />

                        {/* Category counts */}
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
                            {Object.entries(CATEGORY_STYLE).map(([key, s]) => (
                                <Chip key={key} size="small"
                                    label={`${s.label}: ${counts[key] ?? 0}`}
                                    sx={{
                                        fontFamily: MONO, fontSize: 11,
                                        color: s.color,
                                        borderColor: alpha(s.color, 0.4),
                                        background: alpha(s.color, 0.07),
                                    }}
                                    variant="outlined"
                                />
                            ))}
                            <Chip size="small"
                                label={`VRU ACTIVE: ${twin.perception_stats?.vru_active ?? 0}`}
                                sx={{
                                    fontFamily: MONO, fontSize: 11, color: C.ok,
                                    borderColor: alpha(C.ok, 0.4),
                                }}
                                variant="outlined"
                            />
                        </Box>

                        {/* Volume chart */}
                        <Card sx={{ ...glass(), borderRadius: 2, mt: 2 }}>
                            <CardContent>
                                <Typography variant="overline">
                                    Traffic Volume (per minute, last 15 min)
                                </Typography>
                                <ResponsiveContainer width="100%" height={220}>
                                    <AreaChart data={volumeSeries}>
                                        <defs>
                                            <linearGradient id="gVeh" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.45} />
                                                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.03} />
                                            </linearGradient>
                                            <linearGradient id="gPed" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.45} />
                                                <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.03} />
                                            </linearGradient>
                                            <linearGradient id="gCyc" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#34d399" stopOpacity={0.4} />
                                                <stop offset="100%" stopColor="#34d399" stopOpacity={0.03} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid stroke={C.line} strokeDasharray="3 3" />
                                        <XAxis dataKey="t" tick={{ fill: C.textFaint, fontSize: 10, fontFamily: MONO }} />
                                        <YAxis allowDecimals={false} tick={{ fill: C.textFaint, fontSize: 10, fontFamily: MONO }} width={28} />
                                        <RTooltip
                                            contentStyle={{
                                                background: C.panel, border: `1px solid ${C.lineHi}`,
                                                fontFamily: MONO, fontSize: 11,
                                            }}
                                        />
                                        <Legend wrapperStyle={{ fontFamily: MONO, fontSize: 10 }} />
                                        <Area type="monotone" dataKey="vehicle" name="Vehicles" stroke="#f43f5e" fill="url(#gVeh)" strokeWidth={1.5} />
                                        <Area type="monotone" dataKey="pedestrian" name="Pedestrians" stroke="#22d3ee" fill="url(#gPed)" strokeWidth={1.5} />
                                        <Area type="monotone" dataKey="cyclist" name="Cyclists" stroke="#34d399" fill="url(#gCyc)" strokeWidth={1.5} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Turning movements */}
                        <Card sx={{ ...glass(), borderRadius: 2, mt: 2 }}>
                            <CardContent>
                                <Typography variant="overline">Turning Movements</Typography>
                                {Object.keys(twin.flow?.turning_matrix || {}).length === 0 ? (
                                    <Typography sx={{ fontSize: 12, color: C.textFaint, py: 1.5 }}>
                                        Accumulates as tracked road users complete trips through
                                        the intersection (entry approach → exit approach).
                                    </Typography>
                                ) : (
                                    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mt: 1 }}>
                                        {Object.entries(twin.flow.turning_matrix).map(([entry, exits]) => (
                                            <Box key={entry}>
                                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.signal, letterSpacing: '0.08em' }}>
                                                    FROM {entry.toUpperCase()}
                                                </Typography>
                                                {Object.entries(exits).map(([exit, n]) => (
                                                    <Typography key={exit} sx={{ fontFamily: MONO, fontSize: 11, color: C.textDim }}>
                                                        → {exit}: <span style={{ color: C.text }}>{n}</span>
                                                    </Typography>
                                                ))}
                                            </Box>
                                        ))}
                                    </Box>
                                )}
                            </CardContent>
                        </Card>
                    </Grid>

                    {/* Right column: stats, signal, alerts */}
                    <Grid item xs={12} lg={4}>
                        <Grid container spacing={1.5}>
                            <Grid item xs={6}>
                                <StatCard label="Active Objects" value={twin.perception_stats?.active_objects ?? 0}
                                    sub={`${twin.perception_stats?.total_observed ?? 0} observed total`} color={C.signal} />
                            </Grid>
                            <Grid item xs={6}>
                                <StatCard label="Wrong-Way" value={safety.wrong_way ?? 0}
                                    sub="events this session" color={(safety.wrong_way ?? 0) > 0 ? C.critical : C.text} />
                            </Grid>
                            <Grid item xs={6}>
                                <StatCard label="Near Misses" value={safety.near_miss ?? 0}
                                    sub="TTC conflicts detected" color={(safety.near_miss ?? 0) > 0 ? C.danger : C.text} />
                            </Grid>
                            <Grid item xs={6}>
                                <StatCard label="VRU Conflicts" value={safety.vru_conflict ?? 0}
                                    sub="Vision Zero focus" color={(safety.vru_conflict ?? 0) > 0 ? C.warn : C.text} />
                            </Grid>
                        </Grid>

                        <Box sx={{ mt: 2 }}>
                            <SignalPanel
                                cameraId={selectedCamera}
                                signal={twin.signal}
                                rec={twin.signal_recommendation}
                                demand={twin.demand_by_approach}
                                onChanged={refreshSignal}
                            />
                        </Box>

                        <Card sx={{ ...glass(), borderRadius: 2, mt: 2 }}>
                            <CardContent>
                                <Typography variant="overline" sx={{ mb: 1, display: 'block' }}>
                                    Safety Alerts
                                </Typography>
                                <AlertsFeed alerts={alerts.filter(
                                    (a) => String(a.intersection_id) === String(twin.intersection_id)
                                )} />
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            )}
        </Box>
    );
}