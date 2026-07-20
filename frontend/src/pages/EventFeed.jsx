/**
 * Event Feed - Enhanced Alert Interface
 * Modern cybersecurity design with real-time event monitoring
 */
import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    Grid,
    Chip,
    TextField,
    MenuItem,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    alpha,
    useTheme,
    IconButton,
    Tooltip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Badge,
} from '@mui/material';
import {
    Event,
    Warning,
    AccessTime,
    Videocam,
    FilterList,
    Refresh,
    Search,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import { eventAPI, cameraAPI } from '../services/api';

export default function EventFeed() {
    const theme = useTheme();
    const [events, setEvents] = useState([]);
    const [cameras, setCameras] = useState([]);
    const [filters, setFilters] = useState({
        camera_id: '',
        rule: '',
        priority: '',
        limit: 50,
    });
    const [selectedEvent, setSelectedEvent] = useState(null);
    const [openDialog, setOpenDialog] = useState(false);

    useEffect(() => {
        loadCameras();
        loadEvents();
        const interval = setInterval(loadEvents, 3000);
        return () => clearInterval(interval);
    }, [filters]);

    const loadCameras = async () => {
        try {
            const response = await cameraAPI.getAll();
            setCameras(response.data.cameras);
        } catch (err) {
            console.error('Error loading cameras:', err);
        }
    };

    const loadEvents = async () => {
        try {
            const params = {
                camera_id: filters.camera_id || undefined,
                rule: filters.rule || undefined,
                priority: filters.priority || undefined,
                limit: filters.limit,
            };

            const response = await eventAPI.getAll(params);
            setEvents(response.data.events);
        } catch (err) {
            console.error('Error loading events:', err);
        }
    };

    const handleEventClick = async (event) => {
        try {
            const response = await eventAPI.getById(event.id);
            setSelectedEvent(response.data.event);
            setOpenDialog(true);
        } catch (err) {
            console.error('Error loading event details:', err);
        }
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'critical':
                return 'error';
            case 'high':
                return 'error';
            case 'medium':
                return 'warning';
            case 'low':
                return 'info';
            default:
                return 'default';
        }
    };

    const getPriorityGlow = (priority) => {
        switch (priority) {
            case 'critical':
                return '0 0 20px rgba(255, 0, 110, 0.5)';
            case 'high':
                return '0 0 15px rgba(255, 0, 110, 0.3)';
            case 'medium':
                return '0 0 10px rgba(255, 202, 58, 0.3)';
            default:
                return 'none';
        }
    };

    const getPriorityBackground = (priority) => {
        switch (priority) {
            case 'critical':
            case 'high':
                return `linear-gradient(135deg, ${alpha(theme.palette.error.main, 0.15)}, ${alpha(theme.palette.error.main, 0.05)})`;
            case 'medium':
                return `linear-gradient(135deg, ${alpha(theme.palette.warning.main, 0.15)}, ${alpha(theme.palette.warning.main, 0.05)})`;
            default:
                return `linear-gradient(135deg, ${alpha(theme.palette.info.main, 0.15)}, ${alpha(theme.palette.info.main, 0.05)})`;
        }
    };

    return (
        <Box>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="h3" component="h1" sx={{
                        fontWeight: 800,
                        letterSpacing: 2,
                        background: 'linear-gradient(90deg, #00ff88, #00b4d8)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                    }}>
                        Event Feed
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary">
                        Real-time security alerts and notifications
                    </Typography>
                </Box>
                
                <Tooltip title="Refresh Events">
                    <IconButton onClick={loadEvents} sx={{ color: 'primary.main' }}>
                        <Refresh />
                    </IconButton>
                </Tooltip>
            </Box>

            {/* Filters */}
            <Card sx={{ mb: 3, bgcolor: 'background.paper' }}>
                <CardContent>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} sm={4} md={3}>
                            <TextField
                                select
                                fullWidth
                                label="Camera"
                                value={filters.camera_id}
                                onChange={(e) => setFilters({ ...filters, camera_id: e.target.value })}
                                size="small"
                            >
                                <MenuItem value="">All Cameras</MenuItem>
                                {cameras.map((cam) => (
                                    <MenuItem key={cam.id} value={cam.id}>
                                        {cam.name}
                                    </MenuItem>
                                ))}
                            </TextField>
                        </Grid>

                        <Grid item xs={12} sm={4} md={3}>
                            <TextField
                                select
                                fullWidth
                                label="Rule Type"
                                value={filters.rule}
                                onChange={(e) => setFilters({ ...filters, rule: e.target.value })}
                                size="small"
                            >
                                <MenuItem value="">All Rules</MenuItem>
                                <MenuItem value="intrusion">Intrusion</MenuItem>
                                <MenuItem value="loitering">Loitering</MenuItem>
                            </TextField>
                        </Grid>

                        <Grid item xs={12} sm={4} md={3}>
                            <TextField
                                select
                                fullWidth
                                label="Priority"
                                value={filters.priority}
                                onChange={(e) => setFilters({ ...filters, priority: e.target.value })}
                                size="small"
                            >
                                <MenuItem value="">All Priorities</MenuItem>
                                <MenuItem value="critical">Critical</MenuItem>
                                <MenuItem value="high">High</MenuItem>
                                <MenuItem value="medium">Medium</MenuItem>
                                <MenuItem value="low">Low</MenuItem>
                            </TextField>
                        </Grid>

                        <Grid item xs={12} sm={4} md={3}>
                            <TextField
                                select
                                fullWidth
                                label="Limit"
                                value={filters.limit}
                                onChange={(e) => setFilters({ ...filters, limit: e.target.value })}
                                size="small"
                            >
                                <MenuItem value={25}>25</MenuItem>
                                <MenuItem value={50}>50</MenuItem>
                                <MenuItem value={100}>100</MenuItem>
                            </TextField>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            {/* Events Table */}
            <TableContainer component={Card} sx={{ bgcolor: 'background.paper' }}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Timestamp</TableCell>
                            <TableCell>Camera</TableCell>
                            <TableCell>Rule</TableCell>
                            <TableCell>Object</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Priority</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <AnimatePresence>
                            {events.map((event, index) => (
                                <motion.tr
                                    key={event.id}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20 }}
                                    transition={{ delay: index * 0.05 }}
                                    style={{ cursor: 'pointer' }}
                                    onClick={() => handleEventClick(event)}
                                >
                                    <TableCell>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <AccessTime fontSize="small" color="primary" />
                                            {format(new Date(event.timestamp), 'HH:mm:ss')}
                                        </Box>
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            icon={<Videocam />}
                                            label={`Cam ${event.camera_id}`}
                                            size="small"
                                            variant="outlined"
                                        />
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            label={event.rule_type}
                                            color={event.rule_type === 'intrusion' ? 'error' : 'warning'}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>{event.object_type ?? 'N/A'}</TableCell>
                                    <TableCell>
                                        <Box>
                                            <Typography variant="body2">
                                                {event.confidence ? (event.confidence * 100).toFixed(1) + '%' : 'N/A'}
                                            </Typography>
                                        </Box>
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            label={event.priority}
                                            color={getPriorityColor(event.priority)}
                                            size="small"
                                            sx={{ fontWeight: 600 }}
                                        />
                                    </TableCell>
                                </motion.tr>
                            ))}
                        </AnimatePresence>
                    </TableBody>
                </Table>
            </TableContainer>

            {/* Event Details Dialog */}
            <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="md" fullWidth>
                <DialogTitle>Event Details</DialogTitle>
                <DialogContent>
                    {selectedEvent && (
                        <Box>
                            <Grid container spacing={2}>
                                {selectedEvent.snapshot_path && (
                                    <Grid item xs={12}>
                                        <Box
                                            component="img"
                                            src={`/snapshots/${selectedEvent.snapshot_path.split(/[\\/]/)[selectedEvent.snapshot_path.split(/[\\/]/).length - 1]}`}
                                            alt="Event snapshot"
                                            sx={{ width: '100%', borderRadius: 1 }}
                                        />
                                    </Grid>
                                )}

                                {[
                                    { label: 'Event ID', value: selectedEvent.id },
                                    { label: 'Camera ID', value: selectedEvent.camera_id },
                                    { label: 'Rule Type', value: selectedEvent.rule_type },
                                    { label: 'Priority', value: selectedEvent.priority },
                                    { label: 'Object Type', value: selectedEvent.object_type ?? 'N/A' },
                                    { label: 'Confidence', value: selectedEvent.confidence ? `${(selectedEvent.confidence * 100).toFixed(1)}%` : 'N/A' },
                                    { label: 'Timestamp', value: format(new Date(selectedEvent.timestamp), 'PPpp') },
                                ].map((item) => (
                                    <Grid item xs={6} key={item.label}>
                                        <Typography variant="body2">
                                            <strong>{item.label}:</strong> {item.value}
                                        </Typography>
                                    </Grid>
                                ))}

                                {selectedEvent.metadata && (
                                    <Grid item xs={12}>
                                        <Typography variant="body2" sx={{ mt: 2 }}>
                                            <strong>Metadata:</strong>
                                        </Typography>
                                        <Box
                                            component="pre"
                                            sx={{
                                                fontSize: '0.875rem',
                                                p: 2,
                                                bgcolor: 'rgba(0, 0, 0, 0.3)',
                                                borderRadius: 1,
                                                overflow: 'auto',
                                                maxHeight: 200,
                                            }}
                                        >
                                            {JSON.stringify(selectedEvent.metadata, null, 2)}
                                        </Box>
                                    </Grid>
                                )}
                            </Grid>
                        </Box>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenDialog(false)}>Close</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}