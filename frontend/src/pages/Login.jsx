/**
 * Argus Tactical Defense Operations - Security Clearance Terminal
 *
 * Cybernetic authentication gateway with interactive radar telemetry,
 * AES-256-GCM biometric posture indicators, and aerospace HUD aesthetics.
 */
import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Paper,
    TextField,
    Button,
    Typography,
    Alert,
    CircularProgress,
    Stack,
    InputAdornment,
    Divider,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Shield,
    Lock,
    Person,
    Key,
    VpnKey,
    Security,
    VerifiedUser,
    CheckCircleOutline,
} from '@mui/icons-material';

import { authAPI } from '../services/api';
import { C, DISPLAY, MONO, glass } from '../theme';
import { HudReticle, StatusDot } from '../components/ui';

function RadarBackground() {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationFrameId;
        let angle = 0;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        const particles = Array.from({ length: 42 }, () => ({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            size: Math.random() * 2 + 1,
            alpha: Math.random() * 0.4 + 0.1,
            speedY: Math.random() * 0.4 - 0.2,
        }));

        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            // Faint radar concentric circles
            ctx.strokeStyle = 'rgba(0, 242, 254, 0.04)';
            ctx.lineWidth = 1;
            [120, 240, 360, 480, 620].forEach((r) => {
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.stroke();
            });

            // Grid crosshairs
            ctx.beginPath();
            ctx.moveTo(cx, 0);
            ctx.lineTo(cx, canvas.height);
            ctx.moveTo(0, cy);
            ctx.lineTo(canvas.width, cy);
            ctx.stroke();

            // Radar sweep sector
            angle += 0.015;
            const grad = ctx.createConicGradient(angle, cx, cy);
            grad.addColorStop(0, 'rgba(0, 242, 254, 0.08)');
            grad.addColorStop(0.12, 'rgba(0, 242, 254, 0.0)');
            grad.addColorStop(1, 'rgba(0, 242, 254, 0.0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(cx, cy, 620, 0, Math.PI * 2);
            ctx.fill();

            // Subtle telemetry particles
            particles.forEach((p) => {
                ctx.fillStyle = `rgba(0, 242, 254, ${p.alpha})`;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                p.y += p.speedY;
                if (p.y < 0) p.y = canvas.height;
                if (p.y > canvas.height) p.y = 0;
            });

            animationFrameId = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 0,
            }}
        />
    );
}

export default function Login({ onSuccess }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const [busy, setBusy] = useState(false);

    const submit = async (event) => {
        event.preventDefault();
        setBusy(true);
        setError(null);
        try {
            const data = await authAPI.login(username, password);
            onSuccess?.(data.user);
        } catch (err) {
            const status = err.response?.status;
            if (status === 429) {
                setError(
                    err.response?.data?.detail ||
                    'SECURITY LOCKOUT: Excessive failed attempts detected. Rate limiter active.'
                );
            } else if (status === 401) {
                setError('AUTHENTICATION FAILED: Invalid clearance credentials.');
            } else {
                setError(err.response?.data?.detail || 'COMMUNICATION ERROR: Unable to reach Argus API.');
            }
        } finally {
            setBusy(false);
        }
    };

    return (
        <Box
            sx={{
                minHeight: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                bgcolor: C.void,
                p: 2,
                overflow: 'hidden',
            }}
        >
            <RadarBackground />

            {/* Top Telemetry Header */}
            <Box
                sx={{
                    position: 'absolute',
                    top: 20,
                    left: 24,
                    right: 24,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    zIndex: 2,
                    pointerEvents: 'none',
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <StatusDot color={C.signal} size={7} />
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.signal, letterSpacing: '0.12em' }}>
                        ARGUS // AI SURVEILLANCE & DEFENSE OS
                    </Typography>
                </Box>
                <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 2 }}>
                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim, letterSpacing: '0.08em' }}>
                        CIPHER: AES-256-GCM
                    </Typography>
                    <Divider orientation="vertical" flexItem sx={{ borderColor: C.line, my: 0.5 }} />
                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.ok, letterSpacing: '0.08em' }}>
                        AUDIT: SHA-256 CHAINED
                    </Typography>
                </Box>
            </Box>

            {/* Central Terminal Card */}
            <Paper
                elevation={12}
                sx={{
                    position: 'relative',
                    zIndex: 2,
                    p: { xs: 3, sm: 4.5 },
                    width: '100%',
                    maxWidth: 440,
                    ...glass(0.85, C.signal),
                    borderRadius: 2,
                    border: `1px solid ${alpha(C.signal, 0.35)}`,
                    boxShadow: `0 0 40px ${alpha(C.signal, 0.12)}, 0 24px 60px rgba(0, 0, 0, 0.8)`,
                }}
            >
                <HudReticle color={C.signal} size={14} stroke={2} opacity={0.8} />

                {/* Tactical Shield Hologram */}
                <Stack spacing={1.5} alignItems="center" sx={{ mb: 3.5, textAlign: 'center' }}>
                    <Box
                        sx={{
                            position: 'relative',
                            width: 64,
                            height: 64,
                            borderRadius: '50%',
                            display: 'grid',
                            placeItems: 'center',
                            background: `radial-gradient(circle, ${alpha(C.signal, 0.22)} 0%, transparent 70%)`,
                            border: `1px solid ${alpha(C.signal, 0.5)}`,
                            boxShadow: `0 0 24px ${alpha(C.signal, 0.3)}`,
                            '&::before': {
                                content: '""',
                                position: 'absolute',
                                inset: -4,
                                borderRadius: '50%',
                                border: `1px dashed ${alpha(C.signal, 0.35)}`,
                                animation: 'argus-radar-sweep 12s linear infinite',
                            },
                        }}
                    >
                        <Shield sx={{ fontSize: 32, color: C.signal, filter: `drop-shadow(0 0 8px ${C.signal})` }} />
                    </Box>

                    <Box>
                        <Typography
                            variant="h5"
                            sx={{
                                fontFamily: DISPLAY,
                                fontWeight: 700,
                                letterSpacing: '0.12em',
                                color: C.text,
                                textTransform: 'uppercase',
                                lineHeight: 1.1,
                            }}
                        >
                            Argus Tactical Deck
                        </Typography>
                        <Typography
                            sx={{
                                fontFamily: MONO,
                                fontSize: 10.5,
                                color: C.signalDim,
                                letterSpacing: '0.14em',
                                textTransform: 'uppercase',
                                mt: 0.6,
                            }}
                        >
                            CLEARANCE LEVEL ACCESS GATEWAY // v2.1
                        </Typography>
                    </Box>
                </Stack>

                {error && (
                    <Alert
                        severity="error"
                        sx={{
                            mb: 2.5,
                            fontFamily: MONO,
                            fontSize: 11.5,
                            background: alpha(C.danger, 0.15),
                            border: `1px solid ${alpha(C.danger, 0.5)}`,
                            color: '#ff8095',
                            '& .MuiAlert-icon': { color: C.danger },
                        }}
                    >
                        {error}
                    </Alert>
                )}

                <form onSubmit={submit}>
                    <Stack spacing={2.2}>
                        <TextField
                            label="IDENTIFIER / USERNAME"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            autoFocus
                            fullWidth
                            autoComplete="username"
                            disabled={busy}
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <Person sx={{ fontSize: 18, color: C.signalDim }} />
                                    </InputAdornment>
                                ),
                            }}
                            InputLabelProps={{
                                sx: { fontFamily: MONO, fontSize: 11, letterSpacing: '0.08em', color: C.textDim },
                            }}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    fontFamily: MONO,
                                    fontSize: 13,
                                    background: alpha(C.void, 0.6),
                                    '& fieldset': { borderColor: C.line },
                                    '&:hover fieldset': { borderColor: alpha(C.signal, 0.6) },
                                    '&.Mui-focused fieldset': {
                                        borderColor: C.signal,
                                        boxShadow: `0 0 14px ${alpha(C.signal, 0.25)}`,
                                    },
                                },
                            }}
                        />

                        <TextField
                            label="ACCESS KEY / PASSWORD"
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            fullWidth
                            autoComplete="current-password"
                            disabled={busy}
                            InputProps={{
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <Lock sx={{ fontSize: 18, color: C.signalDim }} />
                                    </InputAdornment>
                                ),
                            }}
                            InputLabelProps={{
                                sx: { fontFamily: MONO, fontSize: 11, letterSpacing: '0.08em', color: C.textDim },
                            }}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    fontFamily: MONO,
                                    fontSize: 13,
                                    background: alpha(C.void, 0.6),
                                    '& fieldset': { borderColor: C.line },
                                    '&:hover fieldset': { borderColor: alpha(C.signal, 0.6) },
                                    '&.Mui-focused fieldset': {
                                        borderColor: C.signal,
                                        boxShadow: `0 0 14px ${alpha(C.signal, 0.25)}`,
                                    },
                                },
                            }}
                        />

                        <Button
                            type="submit"
                            variant="contained"
                            size="large"
                            disabled={busy || !username || !password}
                            sx={{
                                py: 1.35,
                                mt: 1,
                                fontFamily: DISPLAY,
                                fontSize: 14,
                                fontWeight: 700,
                                letterSpacing: '0.12em',
                                background: `linear-gradient(135deg, ${C.signal} 0%, #0284c7 100%)`,
                                color: '#020617',
                                textTransform: 'uppercase',
                                boxShadow: `0 0 20px ${alpha(C.signal, 0.4)}`,
                                transition: 'all 0.25s ease',
                                '&:hover': {
                                    background: `linear-gradient(135deg, #38bdf8 0%, #0369a1 100%)`,
                                    boxShadow: `0 0 30px ${alpha(C.signal, 0.65)}`,
                                    transform: 'translateY(-1px)',
                                },
                            }}
                            startIcon={busy ? <CircularProgress size={18} sx={{ color: '#020617' }} /> : <VpnKey />}
                        >
                            {busy ? 'VERIFYING CREDENTIALS…' : 'AUTHENTICATE ACCESS'}
                        </Button>
                    </Stack>
                </form>

                {/* Security Posture Telemetry */}
                <Box sx={{ mt: 3, pt: 2, borderTop: `1px solid ${C.line}` }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint, letterSpacing: '0.06em' }}>
                            CLEARANCE TIERS:
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.signal, letterSpacing: '0.06em' }}>
                            VIEWER · OPERATOR · ADMIN
                        </Typography>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', justifyContent: 'center', mt: 1.5 }}>
                        <Box sx={{ px: 0.75, py: 0.25, borderRadius: 0.75, background: alpha(C.signal, 0.08), border: `1px solid ${alpha(C.signal, 0.2)}` }}>
                            <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.signal }}>
                                [AES-256 BIOMETRICS]
                            </Typography>
                        </Box>
                        <Box sx={{ px: 0.75, py: 0.25, borderRadius: 0.75, background: alpha(C.ok, 0.08), border: `1px solid ${alpha(C.ok, 0.2)}` }}>
                            <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.ok }}>
                                [SHA-256 TAMPER-PROOF]
                            </Typography>
                        </Box>
                        <Box sx={{ px: 0.75, py: 0.25, borderRadius: 0.75, background: alpha(C.accent, 0.08), border: `1px solid ${alpha(C.accent, 0.2)}` }}>
                            <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.accent }}>
                                [ZERO DEFAULT CREDS]
                            </Typography>
                        </Box>
                    </Box>
                </Box>
            </Paper>

            {/* Bottom Status Bar */}
            <Box
                sx={{
                    position: 'absolute',
                    bottom: 16,
                    left: 24,
                    right: 24,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    zIndex: 2,
                    pointerEvents: 'none',
                }}
            >
                <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint }}>
                    LOC: EDGE COMMAND NODE // ONLINE
                </Typography>
                <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint }}>
                    STANDBY MODE // READY FOR OPERATOR INPUT
                </Typography>
            </Box>
        </Box>
    );
}
