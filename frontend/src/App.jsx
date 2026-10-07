import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';
import {
    CssBaseline, Box, Typography, Tooltip, IconButton, Drawer, List,
    ListItemButton, ListItemIcon, ListItemText, Divider, Menu, MenuItem,
    useMediaQuery,
} from '@mui/material';
import {
    Videocam, Event, Analytics, Psychology, Search, Radar, Logout,
    Menu as MenuIcon, Shield, ExpandMore, Traffic, Tv, TvOff,
    VerifiedUser, NotificationsActive, Memory, SettingsInputAntenna,
} from '@mui/icons-material';

import Login from './pages/Login';
import CommandCenter from './pages/CommandCenter';
import CameraManagement from './pages/CameraManagement';
import EventFeed from './pages/EventFeed';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import AdaptiveLearningDashboard from './pages/AdaptiveLearningDashboard';
import MemoryExplorer from './pages/MemoryExplorer';
import CityOSDashboard from './pages/CityOSDashboard';
import { authAPI, systemAPI } from './services/api';
import theme, { C, DISPLAY, MONO } from './theme';
import { StatusDot, ThreatBadge, HudReticle } from './components/ui';

const RAIL = 240;

const NAV = [
    { to: '/', icon: <Radar />, label: 'Command Deck', hint: 'Live video wall, radar and tactical alert feed' },
    { to: '/cityos', icon: <Traffic />, label: 'CityOS Twin', hint: 'Intersection digital twin and traffic intelligence' },
    { to: '/cameras', icon: <Videocam />, label: 'Cameras', hint: 'Register, calibrate and configure feeds' },
    { to: '/events', icon: <Event />, label: 'Event Log', hint: 'Audit trail and incident triage' },
    { to: '/analytics', icon: <Analytics />, label: 'Telemetry', hint: 'Trends, density and spatial distribution' },
    { to: '/memory', icon: <Search />, label: 'Neural Memory', hint: 'Search stored face and vehicle vectors' },
    { to: '/learning', icon: <Psychology />, label: 'Evolutionary AI', hint: 'Adaptive thresholds and genetic engine' },
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
                    mb: 0.5,
                    py: 1,
                    position: 'relative',
                    color: active ? C.text : C.textDim,
                    background: active ? `linear-gradient(90deg, ${alpha(C.signal, 0.15)} 0%, ${alpha(C.signal, 0.03)} 100%)` : 'transparent',
                    border: `1px solid ${active ? alpha(C.signal, 0.4) : 'transparent'}`,
                    boxShadow: active ? `0 0 16px ${alpha(C.signal, 0.15)}` : 'none',
                    transition: 'all 0.2s ease',
                    '&:hover': {
                        background: alpha(C.signal, active ? 0.2 : 0.08),
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
                <ListItemIcon sx={{ minWidth: 32, color: active ? C.signal : C.textFaint }}>
                    {React.cloneElement(item.icon, {
                        sx: {
                            fontSize: 18,
                            filter: active ? `drop-shadow(0 0 6px ${C.signal})` : 'none',
                        },
                    })}
                </ListItemIcon>
                <ListItemText
                    primary={item.label}
                    primaryTypographyProps={{
                        sx: {
                            fontFamily: DISPLAY,
                            fontSize: 13,
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

/** Dual Wall-clock readout: Local and UTC time in aerospace HUD format. */
function MissionClock() {
    const [now, setNow] = useState(() => new Date());
    useEffect(() => {
        const t = setInterval(() => setNow(new Date()), 1000);
        return () => clearInterval(t);
    }, []);

    const localStr = now.toLocaleTimeString([], { hour12: false });
    const utcStr = now.toISOString().slice(11, 19) + ' UTC';

    return (
        <Tooltip title={`Current date: ${now.toDateString()}`}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, px: 1, py: 0.35, borderRadius: 1, background: alpha(C.void, 0.6), border: `1px solid ${C.line}` }}>
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

/** Backend reachability with edge sensor pulse. */
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
    const label = ok === null ? 'PROBING…' : ok ? 'EDGE NODE: ACTIVE' : 'EDGE NODE: OFFLINE';
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 1, borderRadius: 1, background: alpha(C.void, 0.5), border: `1px solid ${C.line}` }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <StatusDot color={color} pulse={ok !== false} size={7} />
                <Typography sx={{ fontFamily: MONO, fontSize: 9.5, fontWeight: 700, color, letterSpacing: '0.08em' }}>
                    {label}
                </Typography>
            </Box>
            <Typography sx={{ fontFamily: MONO, fontSize: 8.5, color: C.textFaint }}>
                1000ms
            </Typography>
        </Box>
    );
}

function Brand() {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, px: 2, py: 2.25 }}>
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
                <Shield sx={{ fontSize: 20, color: C.signal, filter: `drop-shadow(0 0 6px ${C.signal})` }} />
            </Box>
            <Box sx={{ minWidth: 0 }}>
                <Typography sx={{
                    fontFamily: DISPLAY, fontSize: 17, fontWeight: 800,
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
    const { pathname } = useLocation();

    const current = NAV.find((n) => n.to === pathname);

    const rail = (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', background: `linear-gradient(180deg, ${C.panel} 0%, #030712 100%)` }}>
            <Brand />
            <Divider sx={{ borderColor: C.line }} />
            <Box sx={{ px: 1.5, pt: 2, flex: 1, overflowY: 'auto' }}>
                <Typography variant="overline" sx={{ px: 1, display: 'block', mb: 1, color: C.textFaint, letterSpacing: '0.14em' }}>
                    TACTICAL SURFACES
                </Typography>
                <List disablePadding>
                    {NAV.map((item) => (
                        <NavItem key={item.to} item={item} onNavigate={() => setMobileOpen(false)} />
                    ))}
                </List>

                {/* Telemetry Quick Status Widget in Sidebar */}
                <Box sx={{ mt: 3, p: 1.25, borderRadius: 1.5, background: alpha(C.void, 0.6), border: `1px solid ${C.line}` }}>
                    <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.textFaint, letterSpacing: '0.1em', mb: 0.5 }}>
                        AI INFERENCE MODEL
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.accent, fontWeight: 700 }}>
                            YOLOv8-NANO
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.ok }}>
                            ONNX-CPU
                        </Typography>
                    </Box>
                    <Box sx={{ mt: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.textFaint }}>
                            CIPHER / BIOMETRICS:
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
                            <MenuIcon sx={{ fontSize: 20 }} />
                        </IconButton>
                    )}
                    <Box sx={{ minWidth: 0 }}>
                        <Typography sx={{
                            fontFamily: DISPLAY, fontSize: 16, fontWeight: 700,
                            color: C.text, letterSpacing: '0.08em', textTransform: 'uppercase', lineHeight: 1.1,
                        }} noWrap>
                            {current?.label || 'Command Deck'}
                        </Typography>
                        <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }} noWrap>
                            {current?.hint || 'SURVEILLANCE & EDGE COMPUTATION'}
                        </Typography>
                    </Box>

                    <Box sx={{ flex: 1 }} />

                    {/* Threat Status Badge */}
                    <ThreatBadge level="NOMINAL" defcon={4} />

                    {/* Cryptographic Chain Integrity Pill */}
                    <Tooltip title="Cryptographic SHA-256 hash chaining active. Audit events are tamper-evident.">
                        <Box sx={{
                            display: { xs: 'none', lg: 'flex' },
                            alignItems: 'center', gap: 0.6,
                            px: 1, py: 0.4, borderRadius: 1,
                            background: alpha(C.ok, 0.1),
                            border: `1px solid ${alpha(C.ok, 0.3)}`,
                        }}>
                            <VerifiedUser sx={{ fontSize: 13, color: C.ok }} />
                            <Typography sx={{ fontFamily: MONO, fontSize: 9.5, fontWeight: 700, color: C.ok, letterSpacing: '0.06em' }}>
                                CHAIN INTACT
                            </Typography>
                        </Box>
                    </Tooltip>

                    {/* CRT Scanline Toggle */}
                    <Tooltip title={crtMode ? 'Disable CRT Scanlines' : 'Enable CRT Scanlines HUD'}>
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
                            {crtMode ? <Tv sx={{ fontSize: 16 }} /> : <TvOff sx={{ fontSize: 16 }} />}
                        </IconButton>
                    </Tooltip>

                    <MissionClock />

                    <Divider orientation="vertical" flexItem sx={{ borderColor: C.line, my: 0.5 }} />

                    {/* User Clearance Badge */}
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
                        <ExpandMore sx={{ fontSize: 15, color: C.textFaint }} />
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
                            <Logout sx={{ fontSize: 16 }} /> Terminate Session
                        </MenuItem>
                    </Menu>
                </Box>

                {/* Main Content Area */}
                <Box component="main" sx={{ p: { xs: 2, md: 3 }, flex: 1 }}>
                    <Routes>
                        <Route path="/" element={<CommandCenter />} />
                        <Route path="/cityos" element={<CityOSDashboard />} />
                        <Route path="/cameras" element={<CameraManagement />} />
                        <Route path="/events" element={<EventFeed />} />
                        <Route path="/analytics" element={<AnalyticsDashboard />} />
                        <Route path="/learning" element={<AdaptiveLearningDashboard />} />
                        <Route path="/memory" element={<MemoryExplorer />} />
                        <Route path="/dashboard" element={<CommandCenter />} />
                    </Routes>
                </Box>
            </Box>
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
