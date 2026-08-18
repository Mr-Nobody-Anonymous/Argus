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
    Menu as MenuIcon, ShieldOutlined, ExpandMore,
} from '@mui/icons-material';

import Login from './pages/Login';
import CommandCenter from './pages/CommandCenter';
import CameraManagement from './pages/CameraManagement';
import EventFeed from './pages/EventFeed';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import AdaptiveLearningDashboard from './pages/AdaptiveLearningDashboard';
import MemoryExplorer from './pages/MemoryExplorer';
import { authAPI, systemAPI } from './services/api';
import theme, { C, MONO } from './theme';
import { StatusDot } from './components/ui';

const RAIL = 232;

const NAV = [
    { to: '/', icon: <Radar />, label: 'Command', hint: 'Live video wall and alert feed' },
    { to: '/cameras', icon: <Videocam />, label: 'Cameras', hint: 'Register and configure feeds' },
    { to: '/events', icon: <Event />, label: 'Events', hint: 'Event history and triage' },
    { to: '/analytics', icon: <Analytics />, label: 'Analytics', hint: 'Trends and distributions' },
    { to: '/memory', icon: <Search />, label: 'Memory', hint: 'Search stored observations' },
    { to: '/learning', icon: <Psychology />, label: 'Learning', hint: 'Adaptive thresholds' },
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
                    position: 'relative',
                    color: active ? C.text : C.textDim,
                    background: active ? alpha(C.signal, 0.1) : 'transparent',
                    '&:hover': { background: alpha(C.signal, active ? 0.14 : 0.06), color: C.text },
                    '&::before': active ? {
                        content: '""',
                        position: 'absolute',
                        left: 0, top: 8, bottom: 8, width: 2,
                        borderRadius: 2,
                        background: C.signal,
                        boxShadow: `0 0 8px ${C.signal}`,
                    } : {},
                }}
            >
                <ListItemIcon sx={{ minWidth: 34, color: active ? C.signal : C.textFaint }}>
                    {React.cloneElement(item.icon, { sx: { fontSize: 18 } })}
                </ListItemIcon>
                <ListItemText
                    primary={item.label}
                    primaryTypographyProps={{
                        sx: {
                            fontFamily: MONO,
                            fontSize: 11.5,
                            fontWeight: active ? 700 : 500,
                            letterSpacing: '0.06em',
                            textTransform: 'uppercase',
                        },
                    }}
                />
            </ListItemButton>
        </Tooltip>
    );
}

/** Wall-clock readout. A control room needs the time on screen. */
function Clock() {
    const [now, setNow] = useState(() => new Date());
    useEffect(() => {
        const t = setInterval(() => setNow(new Date()), 1000);
        return () => clearInterval(t);
    }, []);
    return (
        <Tooltip title={now.toDateString()}>
            <Typography sx={{
                fontFamily: MONO, fontSize: 12, color: C.textDim,
                fontVariantNumeric: 'tabular-nums', letterSpacing: '0.05em',
            }}>
                {now.toLocaleTimeString([], { hour12: false })}
            </Typography>
        </Tooltip>
    );
}

/** Backend reachability, polled independently of any page. */
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
    const label = ok === null ? 'CHECKING' : ok ? 'BACKEND OK' : 'BACKEND DOWN';
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.85 }}>
            <StatusDot color={color} pulse={ok !== false} size={7} />
            <Typography sx={{ fontFamily: MONO, fontSize: 10, color, letterSpacing: '0.08em' }}>
                {label}
            </Typography>
        </Box>
    );
}

function Brand() {
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, px: 1.5, py: 2 }}>
            <Box sx={{
                width: 32, height: 32, borderRadius: 1.5, flexShrink: 0,
                display: 'grid', placeItems: 'center',
                background: `linear-gradient(135deg, ${alpha(C.signal, 0.22)}, ${alpha(C.accent, 0.18)})`,
                border: `1px solid ${alpha(C.signal, 0.4)}`,
                boxShadow: `0 0 18px ${alpha(C.signal, 0.18)}`,
            }}>
                <ShieldOutlined sx={{ fontSize: 17, color: C.signal }} />
            </Box>
            <Box sx={{ minWidth: 0 }}>
                <Typography sx={{
                    fontFamily: MONO, fontSize: 15, fontWeight: 800,
                    letterSpacing: '0.22em', color: C.text, lineHeight: 1,
                }}>
                    ARGUS
                </Typography>
                <Typography sx={{
                    fontFamily: MONO, fontSize: 8.5, color: C.textFaint,
                    letterSpacing: '0.16em', mt: 0.4,
                }}>
                    VIDEO ANALYTICS
                </Typography>
            </Box>
        </Box>
    );
}

function Shell({ user, onLogout }) {
    const desktop = useMediaQuery(theme.breakpoints.up('md'));
    const [mobileOpen, setMobileOpen] = useState(false);
    const [anchor, setAnchor] = useState(null);
    const { pathname } = useLocation();

    const current = NAV.find((n) => n.to === pathname);

    const rail = (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Brand />
            <Divider sx={{ borderColor: C.line }} />
            <Box sx={{ px: 1.25, pt: 2, flex: 1, overflowY: 'auto' }}>
                <Typography variant="overline" sx={{ px: 1, display: 'block', mb: 0.75 }}>
                    Operations
                </Typography>
                <List disablePadding>
                    {NAV.map((item) => (
                        <NavItem key={item.to} item={item} onNavigate={() => setMobileOpen(false)} />
                    ))}
                </List>
            </Box>
            <Box sx={{ p: 1.5, borderTop: `1px solid ${C.line}` }}>
                <BackendPulse />
            </Box>
        </Box>
    );

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            {desktop ? (
                <Box
                    component="nav"
                    sx={{
                        width: RAIL, flexShrink: 0,
                        borderRight: `1px solid ${C.line}`,
                        background: `linear-gradient(180deg, ${C.panel} 0%, ${C.void} 100%)`,
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
                            background: `linear-gradient(180deg, ${C.panel} 0%, ${C.void} 100%)`,
                            borderRight: `1px solid ${C.line}`,
                        },
                    }}
                >
                    {rail}
                </Drawer>
            )}

            <Box sx={{ flexGrow: 1, ml: desktop ? `${RAIL}px` : 0, minWidth: 0 }}>
                {/* Top bar */}
                <Box sx={{
                    position: 'sticky', top: 0, zIndex: 1100,
                    display: 'flex', alignItems: 'center', gap: 1.5,
                    px: { xs: 2, md: 3 }, py: 1.5,
                    borderBottom: `1px solid ${C.line}`,
                    background: alpha(C.void, 0.85),
                    backdropFilter: 'blur(12px)',
                }}>
                    {!desktop && (
                        <IconButton size="small" onClick={() => setMobileOpen(true)} sx={{ color: C.textDim }}>
                            <MenuIcon sx={{ fontSize: 20 }} />
                        </IconButton>
                    )}
                    <Box sx={{ minWidth: 0 }}>
                        <Typography sx={{
                            fontFamily: MONO, fontSize: 13, fontWeight: 700,
                            color: C.text, letterSpacing: '0.1em', textTransform: 'uppercase',
                        }} noWrap>
                            {current?.label || 'Argus'}
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: C.textFaint }} noWrap>
                            {current?.hint || ''}
                        </Typography>
                    </Box>

                    <Box sx={{ flex: 1 }} />
                    <Clock />
                    <Divider orientation="vertical" flexItem sx={{ borderColor: C.line, my: 0.5 }} />

                    <Box
                        onClick={(e) => setAnchor(e.currentTarget)}
                        sx={{
                            display: 'flex', alignItems: 'center', gap: 0.85, cursor: 'pointer',
                            px: 1, py: 0.5, borderRadius: 1.5,
                            border: `1px solid ${C.line}`,
                            '&:hover': { borderColor: C.lineHi, background: alpha(C.panelHi, 0.7) },
                        }}
                    >
                        <Box sx={{
                            width: 24, height: 24, borderRadius: '50%',
                            display: 'grid', placeItems: 'center',
                            background: alpha(C.accent, 0.16),
                            border: `1px solid ${alpha(C.accent, 0.35)}`,
                            fontFamily: MONO, fontSize: 10, fontWeight: 700, color: C.accent,
                        }}>
                            {String(user?.username || '?').slice(0, 1).toUpperCase()}
                        </Box>
                        <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                            <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.text, lineHeight: 1.2 }}>
                                {user?.username || 'user'}
                            </Typography>
                            <Typography sx={{ fontFamily: MONO, fontSize: 9, color: C.textFaint, letterSpacing: '0.08em' }}>
                                {String(user?.role || '—').toUpperCase()}
                            </Typography>
                        </Box>
                        <ExpandMore sx={{ fontSize: 16, color: C.textFaint }} />
                    </Box>

                    <Menu
                        anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)}
                        PaperProps={{ sx: { background: C.panel, border: `1px solid ${C.line}`, mt: 1 } }}
                    >
                        <MenuItem
                            onClick={() => { setAnchor(null); onLogout(); }}
                            sx={{ fontSize: 13, gap: 1.25, color: C.textDim, '&:hover': { color: C.danger } }}
                        >
                            <Logout sx={{ fontSize: 16 }} /> Sign out
                        </MenuItem>
                    </Menu>
                </Box>

                <Box component="main" sx={{ p: { xs: 2, md: 3 } }}>
                    <Routes>
                        <Route path="/" element={<CommandCenter />} />
                        <Route path="/cameras" element={<CameraManagement />} />
                        <Route path="/events" element={<EventFeed />} />
                        <Route path="/analytics" element={<AnalyticsDashboard />} />
                        <Route path="/learning" element={<AdaptiveLearningDashboard />} />
                        <Route path="/memory" element={<MemoryExplorer />} />
                        {/* The dashboard used to live here; keep the old path working. */}
                        <Route path="/dashboard" element={<CommandCenter />} />
                    </Routes>
                </Box>
            </Box>
        </Box>
    );
}

function App() {
    // Restore the session from sessionStorage so a page refresh does not force a
    // re-login while the token is still valid.
    const [user, setUser] = useState(() => authAPI.currentUser());

    const handleLogout = useCallback(() => {
        authAPI.logout();
        setUser(null);
    }, []);

    // The axios interceptor emits this when a refresh fails, which is the only
    // reliable signal that the session is genuinely over (as opposed to one
    // request racing an expiry).
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
