/**
 * Advanced Surveillance Dashboard
 * Modern command center UI with real-time video feeds and alerts
 * Supports up to 100+ cameras with pagination
 * Integrated with Cross-Camera Tracker and Enhanced Learning
 */
import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Typography,
    Grid,
    Paper,
    Card,
    CardContent,
    CardHeader,
    Chip,
    LinearProgress,
    Avatar,
    IconButton,
    Tooltip,
    Badge,
    Tabs,
    Tab,
    Switch,
    FormControlLabel,
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Alert,
    Button,
    Pagination,
} from '@mui/material';
import {
    Videocam,
    VideocamOff,
    Warning,
    CheckCircle,
    Error,
    NotificationsActive,
    Radar,
    Speed,
    People,
    Refresh,
    Fullscreen,
    Pause,
    PlayArrow,
    Event,
    Settings,
    Close,
    Timeline,
    Delete,
} from '@mui/icons-material';
import { cameraAPI, eventAPI, crossCameraAPI, systemAPI } from '../services/api';

const CAMERAS_PER_PAGE = 12; // Show 12 cameras per page (3 rows x 4 columns on large screens)

function SurveillanceDashboard() {
    const [cameras, setCameras] = useState([]);
    const [events, setEvents] = useState([]);
    const [liveMode, setLiveMode] = useState(true);
    const [selectedCamera, setSelectedCamera] = useState(null);
    const [crossCameraTargets, setCrossCameraTargets] = useState([]);
    const [crossCameraStats, setCrossCameraStats] = useState(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [stats, setStats] = useState({
        activeCameras: 0,
        totalEvents: 0,
        highPriority: 0,
        aiStatus: 'online',
    });
    const [drawerOpen, setDrawerOpen] = useState(false);

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, liveMode ? 2000 : 10000);
        return () => clearInterval(interval);
    }, [liveMode]);

    useEffect(() => {
        fetchCrossCameraData();
        const interval = setInterval(fetchCrossCameraData, 10000);
        return () => clearInterval(interval);
    }, []);

    const fetchData = async () => {
        try {
            const camRes = await cameraAPI.getAll();
            setCameras(camRes.data?.cameras || []);
            
            const eventRes = await eventAPI.getAll({ limit: 10 });
            setEvents(eventRes.data?.events || []);
            
            setStats({
                activeCameras: camRes.data?.cameras?.filter(c => c.status === 'online').length || 0,
                totalEvents: eventRes.data?.events?.length || 0,
                highPriority: eventRes.data?.events?.filter(e => e.priority === 'high').length || 0,
                aiStatus: 'online',
            });
        } catch (error) {
            setStats(prev => ({ ...prev, aiStatus: 'offline' }));
        }
    };

    const fetchCrossCameraData = async () => {
        try {
            const targetsRes = await crossCameraAPI.getTargets();
            setCrossCameraTargets(targetsRes.data?.targets || {});
            
            const statsRes = await crossCameraAPI.getTracks();
            setCrossCameraStats(statsRes.data);
        } catch (err) {
            console.warn('Failed to fetch cross-camera data');
        }
    };

    const handleFullscreen = (camera) => {
        setSelectedCamera(camera);
    };

    const handleCloseFullscreen = () => {
        setSelectedCamera(null);
    };

    const handleStopTarget = async (personId) => {
        try {
            await crossCameraAPI.deleteTarget(personId);
            fetchCrossCameraData();
        } catch (err) {
            console.error('Failed to stop target');
        }
    };

    const handlePageChange = (event, value) => {
        setCurrentPage(value);
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'online': return 'success';
            case 'offline': return 'error';
            default: return 'warning';
        }
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'high': return 'error';
            case 'medium': return 'warning';
            default: return 'info';
        }
    };

    const getPriorityBg = (priority) => {
        switch (priority) {
            case 'high': return 'rgba(255, 82, 82, 0.1)';
            case 'medium': return 'rgba(255, 152, 0, 0.1)';
            default: return 'rgba(0, 188, 212, 0.1)';
        }
    };

    // Paginated cameras
    const totalPages = Math.ceil(cameras.length / CAMERAS_PER_PAGE);
    const paginatedCameras = cameras.slice(
        (currentPage - 1) * CAMERAS_PER_PAGE,
        currentPage * CAMERAS_PER_PAGE
    );

    return (
        <Box sx={{ p: 2 }}>
            {/* Header with Status */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="h3" component="h1" sx={{ 
                        fontWeight: 'bold', 
                        background: 'linear-gradient(90deg, #00e5ff, #2196f3)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        Argus Command Center
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary">
                        AI-Powered Video Surveillance & Analytics ({cameras.length} cameras configured)
                    </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <FormControlLabel
                        control={<Switch checked={liveMode} onChange={(e) => setLiveMode(e.target.checked)} />}
                        label="Live Mode"
                        sx={{ color: 'text.secondary' }}
                    />
                    <Tooltip title="Refresh Data">
                        <IconButton onClick={fetchData} sx={{ color: 'primary.main' }}>
                            <Refresh />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Cross-Camera Targets">
                        <IconButton onClick={() => setDrawerOpen(true)} sx={{ color: 'primary.main' }}>
                            <People />
                        </IconButton>
                    </Tooltip>
                    <Chip
                        icon={stats.aiStatus === 'online' ? <CheckCircle /> : <Error />}
                        label={`AI ${stats.aiStatus.toUpperCase()}`}
                        color={getStatusColor(stats.aiStatus)}
                        variant="outlined"
                    />
                </Box>
            </Box>

            {/* Stats Overview */}
            <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={3}>
                    <Paper sx={{ 
                        p: 2, 
                        bgcolor: 'rgba(0, 188, 212, 0.1)',
                        border: '1px solid rgba(0, 188, 212, 0.2)',
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Avatar sx={{ bgcolor: 'primary.main', width: 48, height: 48 }}>
                                <Videocam />
                            </Avatar>
                            <Box>
                                <Typography variant="h4">{stats.activeCameras}</Typography>
                                <Typography variant="body2" color="text.secondary">Active Cameras</Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                    <Paper sx={{ 
                        p: 2, 
                        bgcolor: 'rgba(255, 152, 0, 0.1)',
                        border: '1px solid rgba(255, 152, 0, 0.2)',
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Avatar sx={{ bgcolor: 'warning.main', width: 48, height: 48 }}>
                                <Event />
                            </Avatar>
                            <Box>
                                <Typography variant="h4">{stats.totalEvents}</Typography>
                                <Typography variant="body2" color="text.secondary">Recent Events</Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                    <Paper sx={{ 
                        p: 2, 
                        bgcolor: 'rgba(255, 82, 82, 0.1)',
                        border: '1px solid rgba(255, 82, 82, 0.2)',
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Avatar sx={{ bgcolor: 'error.main', width: 48, height: 48 }}>
                                <Warning />
                            </Avatar>
                            <Box>
                                <Typography variant="h4">{stats.highPriority}</Typography>
                                <Typography variant="body2" color="text.secondary">High Priority Alerts</Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Grid>
                
                <Grid item xs={12} md={3}>
                    <Paper sx={{ 
                        p: 2, 
                        bgcolor: 'rgba(76, 175, 80, 0.1)',
                        border: '1px solid rgba(76, 175, 80, 0.2)',
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Avatar sx={{ bgcolor: 'success.main', width: 48, height: 48 }}>
                                <Radar />
                            </Avatar>
                            <Box>
                                <Typography variant="h4">AI</Typography>
                                <Typography variant="body2" color="text.secondary">Learning Active</Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>

            {/* Cross-Camera Stats */}
            {crossCameraStats && (
                <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, bgcolor: 'rgba(156, 39, 176, 0.1)' }}>
                            <Typography variant="subtitle2" color="secondary.main">
                                Cross-Camera Tracks
                            </Typography>
                            <Typography variant="h5">
                                {crossCameraStats.active_tracks || 0} active
                            </Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, bgcolor: 'rgba(255, 82, 82, 0.1)' }}>
                            <Typography variant="subtitle2" color="error.main">
                                Match Success Rate
                            </Typography>
                            <Typography variant="h5">
                                {crossCameraStats.matching_stats?.match_success_rate || 0}%
                            </Typography>
                        </Paper>
                    </Grid>
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, bgcolor: 'rgba(0, 188, 212, 0.1)' }}>
                            <Typography variant="subtitle2" color="primary.main">
                                Cameras in Network
                            </Typography>
                            <Typography variant="h5">
                                {crossCameraStats.cameras_in_graph || 0} connected
                            </Typography>
                        </Paper>
                    </Grid>
                </Grid>
            )}

            {/* Main Content */}
            <Box sx={{ display: 'flex', gap: 2, height: 'calc(100vh - 250px)' }}>
                {/* Camera Grid */}
                <Box sx={{ flex: 2, overflow: 'auto' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6">Live Camera Feeds</Typography>
                        {cameras.length > CAMERAS_PER_PAGE && (
                            <Pagination 
                                count={totalPages} 
                                page={currentPage} 
                                onChange={handlePageChange}
                                color="primary"
                                size="small"
                            />
                        )}
                    </Box>
                    <Grid container spacing={1.5}>
                        {paginatedCameras.map((camera) => (
                            <Grid item xs={6} sm={4} md={4} lg={3} xl={2} key={camera.id}>
                                <Paper sx={{ 
                                    position: 'relative',
                                    bgcolor: 'black',
                                    aspectRatio: '16/9',
                                    overflow: 'hidden',
                                    cursor: 'pointer',
                                    transition: 'transform 0.2s',
                                    '&:hover': {
                                        transform: 'scale(1.02)',
                                    }
                                }}>
                                    <Box sx={{ 
                                        position: 'absolute',
                                        top: 4,
                                        left: 4,
                                        zIndex: 1,
                                    }}>
                                        <Chip 
                                            size="small"
                                            label={camera.status.toUpperCase()}
                                            color={getStatusColor(camera.status)}
                                        />
                                    </Box>
                                    <Box sx={{ 
                                        position: 'absolute',
                                        top: 4,
                                        right: 4,
                                        zIndex: 1,
                                    }}>
                                        <Tooltip title="Fullscreen">
                                            <IconButton 
                                                size="small" 
                                                onClick={() => handleFullscreen(camera)}
                                                sx={{ color: 'white', bgcolor: 'rgba(0,0,0,0.5)' }}
                                            >
                                                <Fullscreen />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>
                                    <Box sx={{ 
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        height: '100%',
                                        color: 'text.disabled',
                                    }}>
                                        <VideocamOff sx={{ fontSize: 32, opacity: 0.3 }} />
                                    </Box>
                                    <Box sx={{ 
                                        position: 'absolute',
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        p: 0.5,
                                        background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
                                    }}>
                                        <Typography variant="caption" noWrap>{camera.name}</Typography>
                                        {camera.location_tag && (
                                            <Typography variant="caption" display="block" color="text.secondary" noWrap>
                                                {camera.location_tag}
                                            </Typography>
                                        )}
                                    </Box>
                                </Paper>
                            </Grid>
                        ))}
                    </Grid>
                </Box>

                {/* Alerts Panel */}
                <Box sx={{ flex: 1, minWidth: 280 }}>
                    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                        <Typography variant="h6" gutterBottom>
                            Active Alerts
                        </Typography>
                        <Box sx={{ flex: 1, overflow: 'auto' }}>
                            {events.length === 0 ? (
                                <Typography color="text.secondary" textAlign="center" mt={4}>
                                    No active alerts
                                </Typography>
                            ) : (
                                events.map((event, i) => (
                                    <Box key={i} sx={{ 
                                        mb: 1, 
                                        p: 1.5, 
                                        borderRadius: 1,
                                        bgcolor: getPriorityBg(event.priority),
                                        borderLeft: `3px solid ${getPriorityColor(event.priority)}.main`,
                                    }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                            <Box>
                                                <Typography variant="subtitle2">{event.rule_type}</Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    Camera {event.camera_id} • {event.timestamp}
                                                </Typography>
                                            </Box>
                                            <Chip size="small" label={event.priority} color={getPriorityColor(event.priority)} />
                                        </Box>
                                        {event.details && (
                                            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                                                {JSON.stringify(event.details).substring(0, 100)}...
                                            </Typography>
                                        )}
                                    </Box>
                                ))
                            )}
                        </Box>
                    </Paper>
                </Box>
            </Box>

            {/* Fullscreen Camera Dialog */}
            {selectedCamera && (
                <Box sx={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    bgcolor: 'rgba(0,0,0,0.95)',
                    zIndex: 9999,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}>
                    <IconButton
                        onClick={handleCloseFullscreen}
                        sx={{ position: 'absolute', top: 16, right: 16, color: 'white' }}
                    >
                        <Close />
                    </IconButton>
                    <Paper sx={{ width: '90%', height: '80%', bgcolor: 'black' }}>
                        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="h6" color="white">{selectedCamera.name}</Typography>
                            <Chip label={selectedCamera.status.toUpperCase()} color={getStatusColor(selectedCamera.status)} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100% - 60px)' }}>
                            <VideocamOff sx={{ fontSize: 80, opacity: 0.3 }} />
                        </Box>
                    </Paper>
                </Box>
            )}

            {/* Cross-Camera Targets Drawer */}
            <Drawer
                anchor="right"
                open={drawerOpen}
                onClose={() => setDrawerOpen(false)}
                PaperProps={{
                    sx: { width: 350, bgcolor: 'background.paper' }
                }}
            >
                <Box sx={{ p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6">Cross-Camera Targets</Typography>
                        <IconButton onClick={() => setDrawerOpen(false)}>
                            <Close />
                        </IconButton>
                    </Box>
                    
                    {Object.keys(crossCameraTargets).length > 0 ? (
                        <List>
                            {Object.entries(crossCameraTargets).map(([personId, target]) => (
                                <ListItem key={personId} secondaryAction={
                                    <Tooltip title="Stop Tracking">
                                        <IconButton edge="end" onClick={() => handleStopTarget(personId)}>
                                            <Delete />
                                        </IconButton>
                                    </Tooltip>
                                }>
                                    <ListItemIcon>
                                        <Timeline />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={personId}
                                        secondary={`Camera: ${target.last_camera} • Points: ${target.total_points || 0}`}
                                    />
                                </ListItem>
                            ))}
                        </List>
                    ) : (
                        <Typography color="text.secondary" textAlign="center" mt={4}>
                            No active cross-camera targets
                        </Typography>
                    )}
                </Box>
            </Drawer>
        </Box>
    );
}

export default SurveillanceDashboard;