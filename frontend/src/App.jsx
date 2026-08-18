import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
    CssBaseline,
    AppBar,
    Toolbar,
    Typography,
    Container,
    Box,
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    ListItemButton,
    Divider,
    Chip,
} from '@mui/material';
import {
    Videocam,
    Event,
    Analytics,
    Shield,
    Psychology,
    Search,
    People,
    Radar,
    Timeline,
    Logout,
} from '@mui/icons-material';
import { Button, Tooltip } from '@mui/material';

import Login from './pages/Login';
import CameraManagement from './pages/CameraManagement';
import EventFeed from './pages/EventFeed';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import SurveillanceDashboard from './pages/SurveillanceDashboard';
import AdaptiveLearningDashboard from './pages/AdaptiveLearningDashboard';
import MemoryExplorer from './pages/MemoryExplorer';
import { authAPI } from './services/api';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#00bcd4',  // Cyan for tech/surveillance feel
        },
        secondary: {
            main: '#ff4081',
        },
        background: {
            default: '#000000',  // True black for surveillance monitors
            paper: '#0a0a0a',    // Dark gray for panels
        },
        info: {
            main: '#00e5ff',
        },
        warning: {
            main: '#ff9800',
        },
        error: {
            main: '#ff5252',
        },
    },
    components: {
        MuiAppBar: {
            styleOverrides: {
                root: {
                    background: 'linear-gradient(90deg, #0d47a1 0%, #1565c0 100%)',
                    borderBottom: '1px solid rgba(0, 229, 255, 0.2)',
                }
            }
        },
        MuiDrawer: {
            styleOverrides: {
                paper: {
                    background: 'linear-gradient(180deg, #0a1929 0%, #000000 100%)',
                }
            }
        }
    }
});

const drawerWidth = 240;

function NavItem({ to, icon, label }) {
    const location = useLocation();
    const selected = location.pathname === to;

    return (
        <ListItem disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
                component={Link}
                to={to}
                selected={selected}
                sx={{ borderRadius: 2, mx: 1 }}
            >
                <ListItemIcon sx={{ minWidth: 40 }}>
                    {icon}
                </ListItemIcon>
                <ListItemText primary={label} />
            </ListItemButton>
        </ListItem>
    );
}

function Shell({ user, onLogout }) {
    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
                <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="h6" noWrap component="div">
                        Argus
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <Tooltip title={`Role: ${user?.role || 'unknown'}`}>
                            <Chip
                                icon={<Shield />}
                                label={`${user?.username || 'user'} · ${user?.role || '—'}`}
                                color="secondary"
                                variant="outlined"
                                sx={{ color: 'common.white', borderColor: 'rgba(255,255,255,0.35)' }}
                            />
                        </Tooltip>
                        <Button
                            size="small"
                            startIcon={<Logout />}
                            onClick={onLogout}
                            sx={{ color: 'common.white' }}
                        >
                            Sign out
                        </Button>
                    </Box>
                </Toolbar>
            </AppBar>

            <Drawer
                variant="permanent"
                sx={{
                    width: drawerWidth,
                    flexShrink: 0,
                    [`& .MuiDrawer-paper`]: {
                        width: drawerWidth,
                        boxSizing: 'border-box',
                        borderRight: '1px solid rgba(255,255,255,0.08)',
                    },
                }}
            >
                <Toolbar />
                <Box sx={{ overflow: 'auto', p: 1.5 }}>
                    <Typography variant="overline" sx={{ px: 2, color: 'text.secondary' }}>
                        Navigation
                    </Typography>
                    <List>
                        <NavItem to="/dashboard" icon={<Radar />} label="Dashboard" />
                        <NavItem to="/" icon={<Videocam />} label="Cameras" />
                        <NavItem to="/events" icon={<Event />} label="Events" />
                        <NavItem to="/analytics" icon={<Analytics />} label="Analytics" />
                        <NavItem to="/learning" icon={<Psychology />} label="Adaptive Learning" />
                        <NavItem to="/memory" icon={<Search />} label="Memory" />
                    </List>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="caption" sx={{ px: 2, color: 'text.secondary' }}>
                        Backend and frontend are wired through the Vite proxy at `/api`.
                    </Typography>
                </Box>
            </Drawer>

            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: { xs: 2, md: 3 },
                    bgcolor: 'background.default',
                }}
            >
                <Toolbar />
                <Container maxWidth="xl" sx={{ py: 1 }}>
                    <Routes>
                        <Route path="/dashboard" element={<SurveillanceDashboard />} />
                        <Route path="/" element={<CameraManagement />} />
                        <Route path="/events" element={<EventFeed />} />
                        <Route path="/analytics" element={<AnalyticsDashboard />} />
                        <Route path="/learning" element={<AdaptiveLearningDashboard />} />
                        <Route path="/memory" element={<MemoryExplorer />} />
                    </Routes>
                </Container>
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
