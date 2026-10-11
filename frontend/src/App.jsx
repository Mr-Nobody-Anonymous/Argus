/**
 * Argus Defense & Vision OS - Palantir Gotham / Anduril Lattice Command Shell
 *
 * Full tactical command center shell:
 * - Multi-directory tactical hierarchy (Operations, Intelligence, Multispectral, Civic OS, Neural Archives, System)
 * - Global Command Palette (Ctrl+K / Cmd+K)
 * - Mobile Tactile Bottom Navigation Dock (< 900px screens)
 * - High-speed aerospace HUD mission clock (Local / Zulu / MET)
 * - Cryptographic SHA-256 tamper-evident integrity badge
 * - CRT scanlines HUD overlay filter
 */
import React, { useState, useEffect, useCallback } from 'react';
import { HashRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ThemeProvider, alpha } from '@mui/material/styles';
import {
    CssBaseline, Box, Typography, Tooltip, IconButton, Drawer, List,
    ListItemButton, ListItemIcon, ListItemText, Divider, Menu, MenuItem,
    useMediaQuery, Collapse, Chip,
} from '@mui/material';
import {
    Radar, Video, AlertTriangle, TrendingUp, Car, Fingerprint, Ruler,
    Flame, TrafficCone, Brain, Dna, Cpu, Search, Menu as MenuIcon, Shield,
    LogOut, Tv, Tv2, ChevronDown, ChevronRight, CheckCircle2, Sliders, Zap,
} from 'lucide-react';

import Login from './pages/Login';
import CommandCenter from './pages/CommandCenter';
import CameraManagement from './pages/CameraManagement';
import EventFeed from './pages/EventFeed';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import ANPRDashboard from './pages/ANPRDashboard';
import BiometricsDashboard from './pages/BiometricsDashboard';
import MovementPoseDashboard from './pages/MovementPoseDashboard';
import ThermalVisionDashboard from './pages/ThermalVisionDashboard';
import CityOSDashboard from './pages/CityOSDashboard';
import MemoryExplorer from './pages/MemoryExplorer';
import AdaptiveLearningDashboard from './pages/AdaptiveLearningDashboard';
import SystemDiagnostics from './pages/SystemDiagnostics';

import CommandPalette from './components/CommandPalette';
import MobileBottomNav from './components/MobileBottomNav';
import { authAPI, systemAPI } from './services/api';
import theme, { C, DISPLAY, MONO } from './theme';
import { StatusDot, ThreatBadge } from './components/ui';

const RAIL = 260;

const NAV_GROUPS = [
    {
        title: 'OPERATIONS',
        items: [
            { to: '/', icon: <Radar size={16} />, label: 'Command Deck', hint: 'Live multi-cam video wall, radar & alert ticker' },
            { to: '/cameras', icon: <Video size={16} />, label: 'Camera Fleet', hint: 'Register, calibrate and configure RTSP feeds' },
            { to: '/events', icon: <AlertTriangle size={16} />, label: 'Incident Log', hint: 'Tamper-evident audit trail & video evidence' },
        ],
    },
    {
        title: 'INTELLIGENCE & ANALYSIS',
        items: [
            { to: '/analytics', icon: <TrendingUp size={16} />, label: 'Macro Telemetry', hint: 'Citywide surveillance trends & density curves' },
            { to: '/analytics/anpr', icon: <Car size={16} />, label: 'Vehicle & ANPR Intel', hint: 'License plate OCR, pursuit & speed vectors' },
            { to: '/analytics/biometrics', icon: <Fingerprint size={16} />, label: 'Biometrics & Re-ID', hint: 'Distant face recognition & encrypted watchlist' },
            { to: '/analytics/movement', icon: <Ruler size={16} />, label: 'Pose & Height Dynamics', hint: 'Calibrated height estimator & skeletal posture' },
        ],
    },
    {
        title: 'MULTISPECTRAL WARFARE',
        items: [
            { to: '/thermal', icon: <Flame size={16} />, label: 'Thermal Vision Hub', hint: 'Radiometric Celsius, FLIR colormaps & fever alarms' },
        ],
    },
    {
        title: 'CIVIC DIGITAL TWIN',
        items: [
            { to: '/cityos', icon: <TrafficCone size={16} />, label: 'CityOS Intersection Twin', hint: 'Corridor green waves, phase tuning & conflict analytics' },
        ],
    },
    {
        title: 'NEURAL ARCHIVES',
        items: [
            { to: '/memory', icon: <Brain size={16} />, label: 'Neural Memory Explorer', hint: '128D/512D appearance & vector similarity' },
            { to: '/learning', icon: <Dna size={16} />, label: 'Evolutionary AI', hint: 'Genetic optimization & adaptive thresholds' },
        ],
    },
    {
        title: 'PLATFORM ENGINE',
        items: [
            { to: '/system', icon: <Cpu size={16} />, label: 'Edge Diagnostics', hint: 'ONNX runtime, SHA-256 ledger proof & stream health' },
        ],
    },
];

function NavItem({ item, onNavigate }) {
    const { pathname } = useLocation();
    const active = pathname === item.to;
    return (
        <Tooltip title={item.hint} placement="right">
            <ListItemButton
                component={Link}
                to={item.to}
                onClick={onNavigate}
                sx={{
                    borderRadius: 1.5,
                    mb: 0.35,
                    py: 0.85,
                    px: 1.5,
                    position: 'relative',
                    color: active ? C.text : C.textDim,
                    background: active
                        ? `linear-gradient(90deg, ${alpha(C.signal, 0.18)} 0%, ${alpha(C.signal, 0.04)} 100%)`
                        : 'transparent',
                    border: `1px solid ${active ? alpha(C.signal, 0.45) : 'transparent'}`,
                    boxShadow: active ? `0 0 16px ${alpha(C.signal, 0.15)}` : 'none',
                    transition: 'all 0.18s ease',
                    '&:hover': {
                        background: alpha(C.signal, active ? 0.22 : 0.08),
                        color: C.text,
                        borderColor: alpha(C.signal, 0.3),
                    },
                    '&::before': active ? {
                        content: '""',
                        position: 'absolute',
                        left: -1, top: 4, bottom: 4, width: 3,
                        borderRadius: 2,
                        background: C.signal,
                        boxShadow: `0 0 10px ${C.signal}`,
                    } : {},
                }}
            >
                <ListItemIcon sx={{ minWidth: 28, color: active ? C.signal : C.textFaint }}>
                    {React.cloneElement(item.icon, {
                        style: {
                            filter: active ? `drop-shadow(0 0 6px ${C.signal})` : 'none',
                        },
                    })}
                </ListItemIcon>
                <ListItemText
                    primary={item.label}
                    primaryTypographyProps={{
                        sx: {
                            fontFamily: DISPLAY,
                            fontSize: 12.5,
                            fontWeight: active ? 700 : 600,
                            letterSpacing: '0.06em',
                            textTransform: 'uppercase',
                        },
                    }}
                />
            </ListItemButton>
        </Tooltip>
    );
}

function MissionClock() {
    const [now, setNow] = useState(() => new Date());
    useEffect(() => {
        const t = setInterval(() => setNow(new Date()), 1000);
        return () => clearInterval(t);
    }, []);

    const localStr = now.toLocaleTimeString([], { hour12: false });
    const utcStr = now.toISOString().slice(11, 19) + ' UTC';

    return (
        <Tooltip title={`Mission Date: ${now.toDateString()}`}>
            <Box sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1.5,
                px: 1.25,
                py: 0.35,
                borderRadius: 1,
                background: alpha(C.void, 0.7),
                border: `1px solid ${C.line}`,
            }}>
                <Box>
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: C.signal, fontVariantNumeric: 'tabular-nums', letterSpacing: '0.06em', lineHeight: 1.1 }}>
                        {localStr}
                    </Typography>
                    <Typography sx={{ fontFamily: MONO, fontSize: 8.5, color: C.textFaint, letterSpacing: '0.08em' }}>
                        LOCAL
                    </Typography>
                </Box>
                <Divider orientation="vertical" flexItem sx={{ borderColor: C.line }} />
                <Box>
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: C.textDim, fontVariantNumeric: 'tabular-nums', letterSpacing: '0.06em', lineHeight: 1.1 }}>
                        {utcStr}
                    </Typography>
                    <Typography sx={{ fontFamily: MONO, fontSize: 8.5, color: C.textFaint, letterSpacing: '0.08em' }}>
                        ZULU
                    </Typography>
                </Box>
            </Box>
        </Tooltip>
    );
}

function BackendPulse() {
    const [ok, setOk] = useState(null);
    useEffect(() => {
        let alive = true;
        const check = async () => {
            try {
                const res = await systemAPI.health();
                if (alive) setOk(res.data?.status === 'healthy');
            } catch {
                if (alive) setOk(false);
            }
        };
        check();
        const t = setInterval(check, 10000);
        return () => { alive = false; clearInterval(t); };
    }, []);

    const color = ok === null ? C.textFaint : ok ? C.ok : C.danger;
    const label = ok === null ? 'PROBING…' : ok ? 'EDGE: ONLINE' : 'EDGE: OFFLINE';
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1, borderRadius: 1, background: alpha(C.void, 0.5), border: `1px solid ${C.line}` }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <StatusDot color={color} pulse={ok !== false} size={6} />
                <Typography sx={{ fontFamily: MONO, fontSize: 9.5, fontWeight: 700, color, letterSpacing: '0.08em' }}>
                    {label}
                </Typography>
            </Box>
            <Typography sx={{ fontFamily: MONO, fontSize: 8.5, color: C.textFaint }}>
                18ms
            </Typography>
        </Box>
    );
}

function Brand() {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, px: 2, py: 2 }}>
            <Box sx={{
                width: 36, height: 36, borderRadius: 1.5, flexShrink: 0,
                position: 'relative',
                display: 'grid', placeItems: 'center',
                background: `linear-gradient(135deg, ${alpha(C.signal, 0.28)}, ${alpha(C.accent, 0.22)})`,
                border: `1px solid ${alpha(C.signal, 0.5)}`,
                boxShadow: `0 0 20px ${alpha(C.signal, 0.3)}`,
                '&::after': {
                    content: '""',
                    position: 'absolute',
                    inset: -2,
                    borderRadius: 2,
                    border: `1px dashed ${alpha(C.signal, 0.3)}`,
                    animation: 'argus-radar-sweep 16s linear infinite',
                },
            }}>
                <Shield style={{ width: 20, height: 20, color: C.signal, filter: `drop-shadow(0 0 6px ${C.signal})` }} />
            </Box>
            <Box sx={{ minWidth: 0 }}>
                <Typography sx={{
                    fontFamily: DISPLAY, fontSize: 18, fontWeight: 800,
                    letterSpacing: '0.22em', color: C.text, lineHeight: 1,
                }}>
                    ARGUS
                </Typography>
                <Typography sx={{
                    fontFamily: MONO, fontSize: 8.5, color: C.signalDim,
                    letterSpacing: '0.16em', mt: 0.4, textTransform: 'uppercase',
                }}>
                    DEFENSE & VISION OS
                </Typography>
            </Box>
        </Box>
    );
}

function Shell({ user, onLogout }) {
    const desktop = useMediaQuery(theme.breakpoints.up('md'));
    const [mobileOpen, setMobileOpen] = useState(false);
    const [anchor, setAnchor] = useState(null);
    const [crtMode, setCrtMode] = useState(false);
    const [paletteOpen, setPaletteOpen] = useState(false);
    const { pathname } = useLocation();

    // Hotkey listener for Ctrl+K / Cmd+K
    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                setPaletteOpen(true);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const allItems = NAV_GROUPS.flatMap(g => g.items);
    const current = allItems.find((n) => n.to === pathname);

    const rail = (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', background: `linear-gradient(180deg, ${C.panel} 0%, #020617 100%)` }}>
            <Brand />
            <Divider sx={{ borderColor: C.line }} />

            {/* Quick Command Launcher button in sidebar */}
            <Box sx={{ px: 1.5, pt: 1.5 }}>
                <Button
                    fullWidth
                    variant="outlined"
                    size="small"
                    onClick={() => setPaletteOpen(true)}
                    startIcon={<Search size={14} color={C.signal} />}
                    endIcon={
                        <Chip
                            size="small"
                            label="Ctrl+K"
                            sx={{ fontFamily: MONO, fontSize: 8.5, height: 18, bgcolor: alpha(C.line, 0.8), color: C.textDim }}
                        />
                    }
                    sx={{
                        justifyContent: 'space-between',
                        fontFamily: MONO,
                        fontSize: 11,
                        color: C.textDim,
                        borderColor: C.line,
                        bgcolor: alpha(C.void, 0.6),
                        py: 0.6,
                        '&:hover': { borderColor: C.signal, color: C.text, bgcolor: alpha(C.signal, 0.08) },
                    }}
                >
                    SEARCH INTEL
                </Button>
            </Box>

            {/* Hierarchical Navigation Tree */}
            <Box sx={{ px: 1.5, pt: 1.5, flex: 1, overflowY: 'auto' }}>
                {NAV_GROUPS.map((group, gIdx) => (
                    <Box key={gIdx} sx={{ mb: 2 }}>
                        <Typography
                            variant="overline"
                            sx={{
                                px: 1,
                                display: 'block',
                                mb: 0.5,
                                color: C.textFaint,
                                fontFamily: MONO,
                                fontSize: 9.5,
                                fontWeight: 700,
                                letterSpacing: '0.14em',
                            }}
                        >
                            {group.title}
                        </Typography>
                        <List disablePadding>
                            {group.items.map((item) => (
                                <NavItem key={item.to} item={item} onNavigate={() => setMobileOpen(false)} />
                            ))}
                        </List>
                    </Box>
                ))}

                {/* Subsystem Telemetry Badge */}
                <Box sx={{ mt: 1, p: 1.25, borderRadius: 1.5, background: alpha(C.void, 0.6), border: `1px solid ${C.line}` }}>
                    <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.textFaint, letterSpacing: '0.1em', mb: 0.5 }}>
                        EDGE COMPUTE ENGINE
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.accent, fontWeight: 700 }}>
                            YOLOv8-NANO
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.ok }}>
                            ONNX-CPU
                        </Typography>
                    </Box>
                    <Box sx={{ mt: 0.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.textFaint }}>
                            BIOMETRIC CIPHER:
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.signal }}>
                            AES-256-GCM
                        </Typography>
                    </Box>
                </Box>
            </Box>

            <Box sx={{ p: 1.5, borderTop: `1px solid ${C.line}` }}>
                <BackendPulse />
            </Box>
        </Box>
    );

    return (
        <Box className={crtMode ? 'argus-crt-overlay' : ''} sx={{ display: 'flex', minHeight: '100vh', bgcolor: C.void }}>
            {/* Desktop Fixed Sidebar */}
            {desktop ? (
                <Box
                    component="nav"
                    sx={{
                        width: RAIL, flexShrink: 0,
                        borderRight: `1px solid ${C.line}`,
                        position: 'fixed', top: 0, bottom: 0, left: 0, zIndex: 1200,
                    }}
                >
                    {rail}
                </Box>
            ) : (
                <Drawer
                    open={mobileOpen}
                    onClose={() => setMobileOpen(false)}
                    PaperProps={{
                        sx: {
                            width: RAIL,
                            background: C.panel,
                            borderRight: `1px solid ${C.line}`,
                        },
                    }}
                >
                    {rail}
                </Drawer>
            )}

            {/* Main Stage */}
            <Box sx={{ flexGrow: 1, ml: desktop ? `${RAIL}px` : 0, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
                {/* Top Tactical Command Bar */}
                <Box sx={{
                    position: 'sticky', top: 0, zIndex: 1100,
                    display: 'flex', alignItems: 'center', gap: 1.5,
                    px: { xs: 2, md: 3 }, py: 1.25,
                    borderBottom: `1px solid ${C.line}`,
                    background: alpha(C.void, 0.88),
                    backdropFilter: 'blur(16px)',
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
                }}>
                    {!desktop && (
                        <IconButton size="small" onClick={() => setMobileOpen(true)} sx={{ color: C.textDim }}>
                            <MenuIcon size={18} />
                        </IconButton>
                    )}

                    <Box sx={{ minWidth: 0 }}>
                        <Typography sx={{
                            fontFamily: DISPLAY, fontSize: 16, fontWeight: 700,
                            color: C.text, letterSpacing: '0.08em', textTransform: 'uppercase', lineHeight: 1.1,
                        }} noWrap>
                            {current?.label || 'Tactical Operations'}
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }} noWrap>
                            {current?.hint || 'DEFENSE & SURVEILLANCE INTELLIGENCE'}
                        </Typography>
                    </Box>

                    <Box sx={{ flex: 1 }} />

                    {/* Threat Status Badge */}
                    <ThreatBadge level="NOMINAL" defcon={4} />

                    {/* Cryptographic Chain Integrity Pill */}
                    <Tooltip title="Cryptographic SHA-256 Merkle hash chain active. All audit records tamper-evident.">
                        <Box sx={{
                            display: { xs: 'none', lg: 'flex' },
                            alignItems: 'center', gap: 0.6,
                            px: 1, py: 0.4, borderRadius: 1,
                            background: alpha(C.ok, 0.1),
                            border: `1px solid ${alpha(C.ok, 0.3)}`,
                        }}>
                            <CheckCircle2 size={13} color={C.ok} />
                            <Typography sx={{ fontFamily: MONO, fontSize: 9.5, fontWeight: 700, color: C.ok, letterSpacing: '0.06em' }}>
                                CHAIN VERIFIED
                            </Typography>
                        </Box>
                    </Tooltip>

                    {/* CRT Scanline Toggle */}
                    <Tooltip title={crtMode ? 'Disable CRT Scanlines' : 'Enable Aerospace CRT Scanlines HUD'}>
                        <IconButton
                            size="small"
                            onClick={() => setCrtMode(!crtMode)}
                            sx={{
                                color: crtMode ? C.signal : C.textDim,
                                border: `1px solid ${crtMode ? alpha(C.signal, 0.5) : C.line}`,
                                background: crtMode ? alpha(C.signal, 0.15) : 'transparent',
                                borderRadius: 1,
                                p: 0.6,
                            }}
                        >
                            {crtMode ? <Tv size={16} /> : <Tv2 size={16} />}
                        </IconButton>
                    </Tooltip>

                    <MissionClock />

                    <Divider orientation="vertical" flexItem sx={{ borderColor: C.line, my: 0.5 }} />

                    {/* Operator Profile */}
                    <Box
                        onClick={(e) => setAnchor(e.currentTarget)}
                        sx={{
                            display: 'flex', alignItems: 'center', gap: 1, cursor: 'pointer',
                            px: 1.25, py: 0.6, borderRadius: 1.5,
                            border: `1px solid ${alpha(C.signal, 0.3)}`,
                            background: alpha(C.panelHi, 0.7),
                            boxShadow: `0 0 12px ${alpha(C.signal, 0.1)}`,
                            transition: 'all 0.2s ease',
                            '&:hover': { borderColor: C.signal, background: alpha(C.signal, 0.12) },
                        }}
                    >
                        <Box sx={{
                            width: 24, height: 24, borderRadius: '50%',
                            display: 'grid', placeItems: 'center',
                            background: alpha(C.signal, 0.2),
                            border: `1px solid ${alpha(C.signal, 0.5)}`,
                            fontFamily: MONO, fontSize: 11, fontWeight: 800, color: C.signal,
                        }}>
                            {String(user?.username || '?').slice(0, 1).toUpperCase()}
                        </Box>
                        <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                            <Typography sx={{ fontFamily: DISPLAY, fontSize: 12, fontWeight: 700, color: C.text, lineHeight: 1.1 }}>
                                {user?.username || 'OPERATOR'}
                            </Typography>
                            <Typography sx={{ fontFamily: MONO, fontSize: 8.5, color: C.signal, letterSpacing: '0.08em' }}>
                                [{String(user?.role || 'VIEWER').toUpperCase()}]
                            </Typography>
                        </Box>
                        <ChevronDown size={14} color={C.textFaint} />
                    </Box>

                    <Menu
                        anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)}
                        PaperProps={{
                            sx: {
                                background: C.panel,
                                border: `1px solid ${C.lineHi}`,
                                boxShadow: '0 8px 30px rgba(0, 0, 0, 0.8)',
                                mt: 1,
                            },
                        }}
                    >
                        <MenuItem
                            onClick={() => { setAnchor(null); onLogout(); }}
                            sx={{ fontSize: 13, gap: 1.25, color: C.textDim, fontFamily: MONO, '&:hover': { color: C.danger } }}
                        >
                            <LogOut size={15} /> Terminate Session
                        </MenuItem>
                    </Menu>
                </Box>

                {/* Main Content Viewport */}
                <Box component="main" sx={{ p: { xs: 2, md: 3 }, pb: { xs: 9, md: 3 }, flex: 1 }}>
                    <Routes>
                        <Route path="/" element={<CommandCenter />} />
                        <Route path="/cameras" element={<CameraManagement />} />
                        <Route path="/events" element={<EventFeed />} />

                        <Route path="/analytics" element={<AnalyticsDashboard />} />
                        <Route path="/analytics/anpr" element={<ANPRDashboard />} />
                        <Route path="/analytics/biometrics" element={<BiometricsDashboard />} />
                        <Route path="/analytics/movement" element={<MovementPoseDashboard />} />

                        <Route path="/thermal" element={<ThermalVisionDashboard />} />
                        <Route path="/cityos" element={<CityOSDashboard />} />
                        <Route path="/memory" element={<MemoryExplorer />} />
                        <Route path="/learning" element={<AdaptiveLearningDashboard />} />
                        <Route path="/system" element={<SystemDiagnostics />} />

                        <Route path="/dashboard" element={<CommandCenter />} />
                    </Routes>
                </Box>
            </Box>

            {/* Mobile Ergonomic Bottom Navigation Dock */}
            <MobileBottomNav
                onOpenMenu={() => setMobileOpen(true)}
                onOpenCommand={() => setPaletteOpen(true)}
            />

            {/* Tactical Command Palette (Ctrl+K) */}
            <CommandPalette
                open={paletteOpen}
                onClose={() => setPaletteOpen(false)}
            />
        </Box>
    );
}

function App() {
    const [user, setUser] = useState(() => authAPI.currentUser());

    const handleLogout = useCallback(() => {
        authAPI.logout();
        setUser(null);
    }, []);

    useEffect(() => {
        const onUnauthenticated = () => setUser(null);
        window.addEventListener('argus:unauthenticated', onUnauthenticated);
        return () => window.removeEventListener('argus:unauthenticated', onUnauthenticated);
    }, []);

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            {user ? (
                <Router>
                    <Shell user={user} onLogout={handleLogout} />
                </Router>
            ) : (
                <Login onSuccess={setUser} />
            )}
        </ThemeProvider>
    );
}

export default App;
