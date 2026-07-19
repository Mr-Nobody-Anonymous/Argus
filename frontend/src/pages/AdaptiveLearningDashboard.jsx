/**
 * Adaptive Learning Dashboard
 * Shows learned behavior patterns, emotion baselines, and AI predictions
 */
import React, { useState, useEffect } from 'react';
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
    List,
    ListItem,
    ListItemText,
    Divider,
    Alert,
    Button,
    Stack,
} from '@mui/material';
import {
    Psychology,
    Timeline,
    Radar,
    People,
    Speed,
    Warning,
} from '@mui/icons-material';
import api from '../services/api';

function AdaptiveLearningDashboard() {
    const [stats, setStats] = useState(null);
    const [behaviorProfiles, setBehaviorProfiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchLearningStats();
        const interval = setInterval(fetchLearningStats, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchLearningStats = async () => {
        try {
            const response = await api.get('/api/v1/stats/learning');
            setStats(response.data);
            setLoading(false);
        } catch (err) {
            // Try alternative endpoint
            try {
                const health = await api.get('/api/v1/health');
                setStats({
                    total_behavior_profiles: 0,
                    total_emotion_baselines: 0,
                    learning_buffer_size: 0,
                    sklearn_available: true,
                });
            } catch (e) {
                setError('Failed to load learning stats');
            }
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <Box sx={{ p: 3 }}>
                <LinearProgress />
                <Typography>Loading AI Learning Engine...</Typography>
            </Box>
        );
    }

    return (
        <Box>
            <Typography variant="h4" gutterBottom sx={{ color: 'primary.main', mb: 3 }}>
                Adaptive Learning Dashboard
            </Typography>

            {error && <Alert severity="warning">{error}</Alert>}

            {/* Stats Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={3}>
                    <Card sx={{ bgcolor: 'background.paper', borderLeft: '4px solid #00bcd4' }}>
                        <CardContent>
                            <Stack direction="row" alignItems="center" spacing={2}>
                                <Avatar sx={{ bgcolor: 'primary.main' }}>
                                    <People />
                                </Avatar>
                                <Box>
                                    <Typography variant="h4">
                                        {stats?.total_behavior_profiles || 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Behavior Profiles
                                    </Typography>
                                </Box>
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                    <Card sx={{ bgcolor: 'background.paper', borderLeft: '4px solid #ff4081' }}>
                        <CardContent>
                            <Stack direction="row" alignItems="center" spacing={2}>
                                <Avatar sx={{ bgcolor: 'secondary.main' }}>
                                    <Psychology />
                                </Avatar>
                                <Box>
                                    <Typography variant="h4">
                                        {stats?.total_emotion_baselines || 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Emotion Baselines
                                    </Typography>
                                </Box>
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                    <Card sx={{ bgcolor: 'background.paper', borderLeft: '4px solid #ff9800' }}>
                        <CardContent>
                            <Stack direction="row" alignItems="center" spacing={2}>
                                <Avatar sx={{ bgcolor: 'warning.main' }}>
                                    <Timeline />
                                </Avatar>
                                <Box>
                                    <Typography variant="h4">
                                        {stats?.learning_buffer_size || 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Learning Buffer
                                    </Typography>
                                </Box>
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                    <Card sx={{ bgcolor: 'background.paper', borderLeft: '4px solid #4caf50' }}>
                        <CardContent>
                            <Stack direction="row" alignItems="center" spacing={1}>
                                <Avatar sx={{ bgcolor: 'success.main' }}>
                                    <Radar />
                                </Avatar>
                                <Box>
                                    <Chip
                                        label={stats?.sklearn_available ? 'Active' : 'Disabled'}
                                        color={stats?.sklearn_available ? 'success' : 'default'}
                                        variant="filled"
                                    />
                                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                        ML Clustering
                                    </Typography>
                                </Box>
                            </Stack>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* AI Capabilities */}
            <Paper sx={{ p: 3, mb: 3, bgcolor: 'background.paper' }}>
                <Typography variant="h6" gutterBottom>
                    Self-Learning Intelligence Features
                </Typography>
                <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                        <Card variant="outlined" sx={{ bgcolor: 'rgba(0, 188, 212, 0.05)' }}>
                            <CardContent>
                                <Typography variant="subtitle1" color="primary.main" gutterBottom>
                                    🎯 Behavior Pattern Learning
                                </Typography>
                                <Typography variant="body2">
                                    Automatically adapts to each person's typical speed,
                                    movement patterns, and activity times. Enables
                                    personalized anomaly detection.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>

                    <Grid item xs={12} md={4}>
                        <Card variant="outlined" sx={{ bgcolor: 'rgba(255, 64, 129, 0.05)' }}>
                            <CardContent>
                                <Typography variant="subtitle1" color="secondary.main" gutterBottom>
                                    😊 Emotion Recognition
                                </Typography>
                                <Typography variant="body2">
                                    Learns normal emotional expressions for known persons.
                                    Detects unusual emotional states in real-time.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>

                    <Grid item xs={12} md={4}>
                        <Card variant="outlined" sx={{ bgcolor: 'rgba(255, 152, 0, 0.05)' }}>
                            <CardContent>
                                <Typography variant="subtitle1" color="warning.main" gutterBottom>
                                    🔮 Predictive Analytics
                                </Typography>
                                <Typography variant="body2">
                                    Predicts future behavior trajectories.
                                    Identifies suspicious activity patterns before they escalate.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            </Paper>

            {/* Detection Types */}
            <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
                <Typography variant="h6" gutterBottom>
                    Intelligent Detection Types
                </Typography>
                <List>
                    <ListItem>
                        <ListItemText
                            primary="Erratic Movement Detection"
                            secondary="Identifies unpredictable movement patterns indicating potential threats"
                        />
                    </ListItem>
                    <Divider />
                    <ListItem>
                        <ListItemText
                            primary="Emotion Anomaly Detection"
                            secondary="Flags unusual emotional states based on learned baselines"
                        />
                    </ListItem>
                    <Divider />
                    <ListItem>
                        <ListItemText
                            primary="Crowd Density Analysis"
                            secondary="Monitors for overcrowding and social distancing violations"
                        />
                    </ListItem>
                    <Divider />
                    <ListItem>
                        <ListItemText
                            primary="Trajectory Prediction"
                            secondary="Predicts future positions to enable proactive alerts"
                        />
                    </ListItem>
                    <Divider />
                    <ListItem>
                        <ListItemText
                            primary="Time-Based Activity Anomalies"
                            secondary="Detects unusual activity levels for specific hours"
                        />
                    </ListItem>
                </List>
            </Paper>
        </Box>
    );
}

export default AdaptiveLearningDashboard;