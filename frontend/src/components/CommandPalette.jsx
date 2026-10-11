/**
 * Tactical Command Palette (Ctrl+K / Cmd+K)
 *
 * Palantir Gotham & Anduril Lattice-grade rapid tactical navigation & action launcher.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Dialog, DialogContent, TextField, Box, Typography, List,
    ListItemButton, ListItemIcon, ListItemText, InputAdornment, Chip,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Search, Radar, Video, AlertTriangle, TrendingUp, Car, Fingerprint,
    Ruler, Flame, TrafficCone, Cpu, ShieldCheck, Zap, Terminal, CornerDownLeft,
} from 'lucide-react';
import { C, DISPLAY, MONO, glass } from '../theme';

const COMMANDS = [
    { id: 'deck', label: 'Command Deck // Live Video Wall', category: 'OPERATIONS', path: '/', icon: <Radar size={16} /> },
    { id: 'cameras', label: 'Camera & Sensor Fleet Hub', category: 'OPERATIONS', path: '/cameras', icon: <Video size={16} /> },
    { id: 'events', label: 'Incident Log & Forensic Triage', category: 'OPERATIONS', path: '/events', icon: <AlertTriangle size={16} /> },

    { id: 'anpr', label: 'ANPR & Vehicle Pursuit Intel', category: 'INTELLIGENCE', path: '/analytics/anpr', icon: <Car size={16} /> },
    { id: 'biometrics', label: 'Facial Biometrics & Distance Re-ID', category: 'INTELLIGENCE', path: '/analytics/biometrics', icon: <Fingerprint size={16} /> },
    { id: 'movement', label: 'Height, Pose & Movement Dynamics', category: 'INTELLIGENCE', path: '/analytics/movement', icon: <Ruler size={16} /> },
    { id: 'telemetry', label: 'Macro Telemetry & Trends', category: 'INTELLIGENCE', path: '/analytics', icon: <TrendingUp size={16} /> },

    { id: 'thermal', label: 'Thermal Vision & Radiometric Center', category: 'MULTISPECTRAL', path: '/thermal', icon: <Flame size={16} /> },
    { id: 'cityos', label: 'CityOS Intersection Digital Twin', category: 'CIVIC OS', path: '/cityos', icon: <TrafficCone size={16} /> },
    { id: 'memory', label: 'Neural Memory & Vector Search', category: 'ARCHIVES', path: '/memory', icon: <Search size={16} /> },
    { id: 'learning', label: 'Evolutionary AI & Self-Learning', category: 'ARCHIVES', path: '/learning', icon: <Zap size={16} /> },
    { id: 'system', label: 'Edge System Diagnostics & Cryptographic Ledger', category: 'SYSTEM', path: '/system', icon: <Cpu size={16} /> },
];

export default function CommandPalette({ open, onClose }) {
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const navigate = useNavigate();

    // Filter commands by query
    const filtered = COMMANDS.filter(cmd =>
        cmd.label.toLowerCase().includes(query.toLowerCase()) ||
        cmd.category.toLowerCase().includes(query.toLowerCase())
    );

    useEffect(() => {
        setSelectedIndex(0);
    }, [query]);

    // Handle keyboard navigation
    const handleKeyDown = (e) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIndex(i => Math.min(i + 1, filtered.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIndex(i => Math.max(i - 1, 0));
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (filtered[selectedIndex]) {
                navigate(filtered[selectedIndex].path);
                onClose();
            }
        }
    };

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="sm"
            fullWidth
            PaperProps={{
                sx: {
                    bgcolor: C.panel,
                    border: `1px solid ${C.signal}`,
                    boxShadow: `0 0 40px ${alpha(C.signal, 0.25)}`,
                    borderRadius: 2,
                    overflow: 'hidden',
                },
            }}
        >
            <Box sx={{ p: 2, borderBottom: `1px solid ${C.line}`, bgcolor: alpha(C.void, 0.6) }}>
                <TextField
                    autoFocus
                    placeholder="Type a tactical command, plate, suspect or section..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    fullWidth
                    variant="standard"
                    InputProps={{
                        disableUnderline: true,
                        startAdornment: (
                            <InputAdornment position="start">
                                <Search size={18} color={C.signal} />
                            </InputAdornment>
                        ),
                        endAdornment: (
                            <InputAdornment position="end">
                                <Chip
                                    size="small"
                                    label="ESC TO EXIT"
                                    sx={{ fontFamily: MONO, fontSize: 9, bgcolor: alpha(C.line, 0.8), color: C.textDim }}
                                />
                            </InputAdornment>
                        ),
                        sx: { fontFamily: MONO, fontSize: 13, color: C.text },
                    }}
                />
            </Box>

            <DialogContent sx={{ p: 1, maxHeight: 380 }}>
                {filtered.length === 0 ? (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 12, color: C.textFaint }}>
                            NO MATCHING TACTICAL DIRECTIVES FOUND
                        </Typography>
                    </Box>
                ) : (
                    <List disablePadding>
                        {filtered.map((cmd, idx) => {
                            const isSelected = idx === selectedIndex;
                            return (
                                <ListItemButton
                                    key={cmd.id}
                                    selected={isSelected}
                                    onClick={() => {
                                        navigate(cmd.path);
                                        onClose();
                                    }}
                                    sx={{
                                        borderRadius: 1.5,
                                        mb: 0.5,
                                        bgcolor: isSelected ? alpha(C.signal, 0.15) : 'transparent',
                                        border: `1px solid ${isSelected ? alpha(C.signal, 0.4) : 'transparent'}`,
                                        '&:hover': { bgcolor: alpha(C.signal, 0.1) },
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 32, color: isSelected ? C.signal : C.textDim }}>
                                        {cmd.icon}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={cmd.label}
                                        primaryTypographyProps={{
                                            sx: { fontFamily: DISPLAY, fontSize: 13, fontWeight: 700, color: isSelected ? C.signal : C.text },
                                        }}
                                        secondary={cmd.category}
                                        secondaryTypographyProps={{
                                            sx: { fontFamily: MONO, fontSize: 9.5, color: C.textFaint },
                                        }}
                                    />
                                    {isSelected && (
                                        <CornerDownLeft size={14} color={C.signal} />
                                    )}
                                </ListItemButton>
                            );
                        })}
                    </List>
                )}
            </DialogContent>
        </Dialog>
    );
}
