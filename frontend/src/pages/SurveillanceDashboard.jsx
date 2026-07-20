/**
 * Advanced Surveillance Dashboard
 * Modern command center UI with real-time video feeds and alerts
 */
import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Grid,
    Paper,
    Chip,
    Avatar,
    IconButton,
    Tooltip,
    Switch,
    FormControlLabel,
} from '@mui/material';
import {
    Videocam,
    VideocamOff,
    Warning,
    CheckCircle,
    Error,
    Radar,
    Refresh,
    Event,
} from '@mui/icons-material';
import api from '../services/api';

function SurveillanceDashboard() {
    const [cameras, setCameras] = useState([]);
    const [events, setEvents] = useState([]);
    const [liveMode, setLiveMode] = useState(true);
    const [stats, setStats] = useState({
        activeCameras: 0,
        totalEvents: 0,
        highPriority: 0,
        aiStatus: 'online',
    });

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, liveMode ? 2000 : 10000);
        return () => clearInterval(interval);
    }, [liveMode]);

    const fetchData = async () => {
        try {
            const camRes = await api.get('/api/v1/cameras');
            setCameras(camRes.data || []);
            
            const eventRes = await api.get('/api/v1/events?limit=10');
            setEvents(eventRes.data || []);
            
            setStats({
                activeCameras: camRes.data?.filter(c => c.status === 'online').length || 0,
                totalEvents: eventRes.data?.length || 0,
                highPriority: eventRes.data?.filter(e => e.priority === 'high').length || 0,
                aiStatus: 'online',
            });
        } catch (error) {
            setStats(prev => ({ ...prev, aiStatus: 'offline' }));
        }
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
                        AI-Powered Video Surveillance & Analytics
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

            {/* Main Content */}
            <Box sx={{ display: 'flex', gap: 2, height: 'calc(100vh - 250px)' }}>
                {/* Camera Grid */}
                <Box sx={{ flex: 2, overflow: 'auto' }}>
                    <Typography variant="h6" gutterBottom>Live Camera Feeds</Typography>
                    <Grid container spacing={2}>
                        {cameras.map((camera) => (
                            <Grid item xs={12} md={6} lg={4} key={camera.id}>
                                <Paper sx={{ 
                                    position: 'relative',
                                    bgcolor: 'black',
                                    aspectRatio: '16/9',
                                    overflow: 'hidden',
                                }}>
                                    <Box sx={{ 
                                        position: 'absolute',
                                        top: 8,
                                        left: 8,
                                        zIndex: 1,
                                    }}>
                                        <Chip 
                                            size="small"
                                            label={camera.status.toUpperCase()}
                                            color={getStatusColor(camera.status)}
                                        />
                                    </Box>
                                    <Box sx={{ 
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        height: '100%',
                                        color: 'text.disabled',
                                    }}>
                                        <VideocamOff sx={{ fontSize: 48, opacity: 0.3 }} />
                                    </Box>
                                    <Box sx={{ 
                                        position: 'absolute',
                                        bottom: 0,
                                        left: 0,
                                        right: 0,
                                        p: 1,
                                        background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
                                    }}>
                                        <Typography variant="caption">{camera.name}</Typography>
                                        <Typography variant="caption" display="block" color="text.secondary">
                                            {camera.location_tag}
                                        </Typography>
                                    </Box>
                                </Paper>
                            </Grid>
                        ))}
                    </Grid>
                </Box>

                {/* Alerts Panel */}
                <Box sx={{ flex: 1, minWidth: 300 }}>
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
                                        bgcolor: 'rgba(255,255,255,0.03)',
                                        borderLeft: `3px solid ${getPriorityColor(event.priority)}.main`,
                                    }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                            <Typography variant="subtitle2">{event.rule_type}</Typography>
                                            <Chip size="small" label={event.priority} color={getPriorityColor(event.priority)} />
                                        </Box>
                                        <Typography variant="caption" color="text.secondary">
                                            Camera {event.camera_id} • {event.timestamp}
                                        </Typography>
                                    </Box>
                                ))
                            )}
                        </Box>
                    </Paper>
                </Box>
            </Box>
        </Box>
    );
}

export default SurveillanceDashboard;