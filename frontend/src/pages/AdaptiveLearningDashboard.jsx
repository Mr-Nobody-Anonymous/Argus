/**
 * Adaptive Learning Dashboard
 * Shows learned behavior patterns, emotion baselines, and AI predictions
 * Integrated with Cross-Camera Tracker for enhanced analytics
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
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
    Tooltip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
} from '@mui/material';
import {
    Psychology,
    Timeline,
    Radar,
    People,
    Speed,
    Warning,
    TrackChanges,
    NetworkCheck,
    Delete,
    Add,
} from '@mui/icons-material';
import { learningAPI, crossCameraAPI } from '../services/api';

function AdaptiveLearningDashboard() {
    const [stats, setStats] = useState(null);
    const [behaviorProfiles, setBehaviorProfiles] = useState([]);
    const [crossCameraStats, setCrossCameraStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [targetDialog, setTargetDialog] = useState(false);
    const [newTarget, setNewTarget] = useState({ person_id: '', camera_id: '', reason: '' });

    useEffect(() => {
        fetchLearningStats();
        fetchCrossCameraStats();
        const interval = setInterval(fetchLearningStats, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchLearningStats = async () => {
        try {
            const response = await learningAPI.getStats();
            setStats(response.data);
            setLoading(false);
        } catch (err) {
            setError('Failed to load learning stats');
            setLoading(false);
        }
    };

    const fetchCrossCameraStats = async () => {
        try {
            const response = await crossCameraAPI.getTracks();
            setCrossCameraStats(response.data);
        } catch (err) {
            console.warn('Failed to load cross-camera stats');
        }
    };

    const handleSetTarget = async () => {
        if (!newTarget.person_id || !newTarget.camera_id) {
            setError('Person ID and Camera ID are required');
            return;
        }
        
        try {
            await crossCameraAPI.setTarget(
                newTarget.person_id, 
                parseInt(newTarget.camera_id), 
                newTarget.reason
            );
            setTargetDialog(false);
            setNewTarget({ person_id: '', camera_id: '', reason: '' });
            fetchCrossCameraStats();
        } catch (err) {
            setError('Failed to set target');
        }
    };

    const handleStopTarget = async (personId) => {
        try {
            await crossCameraAPI.deleteTarget(personId);
            fetchCrossCameraStats();
        } catch (err) {
            setError('Failed to stop target');
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

    const getSuccessRateColor = (rate) => {
        if (rate >= 80) return 'success';
        if (rate >= 50) return 'warning';
        return 'error';
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom sx={{ color: 'primary.main', mb: 3 }}>
                Adaptive Learning Dashboard
            </Typography>

            {error && <Alert severity="warning" sx={{ mb: 2 }}>{error}</Alert>}

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
                                        {crossCameraStats?.active_tracks || 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Active Cross-Camera Tracks
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
                                        label={stats?.sklearn_available ? 'Active' : 'Fallback'}
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

            {/* Match Success Rate */}
            {crossCameraStats?.matching_stats && (
                <Grid container spacing={3} sx={{ mb: 4 }}>
                    <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
                            <Typography variant="h6" gutterBottom>
                                Cross-Camera Matching Performance
                            </Typography>
                            <Stack spacing={2}>
                                <Box>
                                    <Typography variant="body2" color="text.secondary">
                                        Match Success Rate: {crossCameraStats.matching_stats.success_rate || 0}%
                                    </Typography>
                                    <LinearProgress 
                                        variant="determinate" 
                                        value={Math.min(100, crossCameraStats.matching_stats.total_matches || 0)} 
                                        color={getSuccessRateColor(crossCameraStats.matching_stats.match_success_rate || 0)}
                                        sx={{ mt: 1 }}
                                    />
                                </Box>
                                <Box sx={{ display: 'flex', gap: 2 }}>
                                    <Chip 
                                        label={`Total Matches: ${crossCameraStats.matching_stats.total_matches || 0}`} 
                                        size="small" 
                                    />
                                    <Chip 
                                        label={`Successful: ${crossCameraStats.matching_stats.successful_matches || 0}`} 
                                        size="small" 
                                        color="success"
                                    />
                                </Box>
                            </Stack>
                        </Paper>
                    </Grid>

                    <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
                            <Typography variant="h6" gutterBottom>
                                Cross-Camera Configuration
                            </Typography>
                            <Stack spacing={1}>
                                <Typography variant="body2">
                                    <strong>Cameras in Graph:</strong> {crossCameraStats.cameras_in_graph || 0}
                                </Typography>
                                <Typography variant="body2">
                                    <strong>Cameras with Active Tracks:</strong> {crossCameraStats.cameras_with_active_tracks || 0}
                                </Typography>
                                <Typography variant="body2">
                                    <strong>Average Path Length:</strong> {crossCameraStats.avg_path_length || 0} points
                                </Typography>
                                <Button 
                                    variant="contained" 
                                    startIcon={<Add />}
                                    onClick={() => setTargetDialog(true)}
                                    sx={{ mt: 2 }}
                                >
                                    Add Target Person
                                </Button>
                            </Stack>
                        </Paper>
                    </Grid>
                </Grid>
            )}

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

                    <Grid item xs={12} md={4}>
                        <Card variant="outlined" sx={{ bgcolor: 'rgba(76, 175, 80, 0.05)' }}>
                            <CardContent>
                                <Typography variant="subtitle1" color="success.main" gutterBottom>
                                    🌐 Cross-Camera Tracking
                                </Typography>
                                <Typography variant="body2">
                                    Maintains global identity across multiple cameras.
                                    Tracks movement paths through the camera network.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>

                    <Grid item xs={12} md={4}>
                        <Card variant="outlined" sx={{ bgcolor: 'rgba(156, 39, 176, 0.05)' }}>
                            <CardContent>
                                <Typography variant="subtitle1" color="secondary.main" gutterBottom>
                                    📊 Trajectory Clustering
                                </Typography>
                                <Typography variant="body2">
                                    Groups persons with similar movement patterns.
                                    Uses DBSCAN clustering for anomaly detection.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>

                    <Grid item xs={12} md={4}>
                        <Card variant="outlined" sx={{ bgcolor: 'rgba(255, 82, 82, 0.05)' }}>
                            <CardContent>
                                <Typography variant="subtitle1" color="error.main" gutterBottom>
                                    ⚡ Real-time Prediction
                                </Typography>
                                <Typography variant="body2">
                                    Predicts next camera and future positions.
                                    Enables proactive security alerts.
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            </Paper>

            {/* Detection Types */}
            <Paper sx={{ p: 3, mb: 3, bgcolor: 'background.paper' }}>
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
                            primary="Cross-Camera Person Matching"
                            secondary="Matches persons across different camera views for continuous tracking"
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

            {/* Cross-Camera Features Status */}
            <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
                <Typography variant="h6" gutterBottom>
                    Enabled AI Features
                </Typography>
                <Grid container spacing={2}>
                    {stats?.features && Object.entries(stats.features).map(([feature, enabled]) => (
                        <Grid item xs={6} md={3} key={feature}>
                            <Chip
                                label={feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                color={enabled ? 'success' : 'default'}
                                variant={enabled ? 'filled' : 'outlined'}
                                sx={{ width: '100%' }}
                            />
                        </Grid>
                    ))}
                </Grid>
            </Paper>

            {/* Add Target Dialog */}
            <Dialog open={targetDialog} onClose={() => setTargetDialog(false)}>
                <DialogTitle>Add Target Person</DialogTitle>
                <DialogContent>
                    <Stack spacing={2} sx={{ mt: 1, minWidth: 300 }}>
                        <TextField
                            label="Person ID"
                            value={newTarget.person_id}
                            onChange={(e) => setNewTarget({ ...newTarget, person_id: e.target.value })}
                            fullWidth
                        />
                        <TextField
                            label="Camera ID"
                            type="number"
                            value={newTarget.camera_id}
                            onChange={(e) => setNewTarget({ ...newTarget, camera_id: e.target.value })}
                            fullWidth
                        />
                        <TextField
                            label="Reason (optional)"
                            value={newTarget.reason}
                            onChange={(e) => setNewTarget({ ...newTarget, reason: e.target.value })}
                            fullWidth
                            multiline
                            rows={2}
                        />
                    </Stack>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setTargetDialog(false)}>Cancel</Button>
                    <Button onClick={handleSetTarget} variant="contained">Add Target</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}

export default AdaptiveLearningDashboard;