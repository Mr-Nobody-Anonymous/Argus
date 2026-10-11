/**
 * Mobile Tactical Bottom Navigation Bar
 *
 * Ensures flawless responsiveness and ergonomics on mobile phones & field handhelds:
 * - Fixed frosted glass bottom dock
 * - Instant access to Command Deck, Cameras, Alerts, Thermal & Intelligence
 * - Quick Drawer launcher for sub-directories
 */
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography } from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Radar, Video, AlertTriangle, Flame, Layers, Search,
} from 'lucide-react';
import { C, DISPLAY, MONO } from '../theme';

const ITEMS = [
    { to: '/', label: 'DECK', icon: <Radar size={19} /> },
    { to: '/cameras', label: 'CAMS', icon: <Video size={19} /> },
    { to: '/thermal', label: 'THERMAL', icon: <Flame size={19} /> },
    { to: '/events', label: 'ALERTS', icon: <AlertTriangle size={19} /> },
];

export default function MobileBottomNav({ onOpenMenu, onOpenCommand }) {
    const { pathname } = useLocation();
    const navigate = useNavigate();

    return (
        <Box
            sx={{
                display: { xs: 'flex', md: 'none' },
                position: 'fixed',
                bottom: 0,
                left: 0,
                right: 0,
                height: 56,
                bgcolor: alpha(C.void, 0.92),
                backdropFilter: 'blur(16px)',
                borderTop: `1px solid ${C.line}`,
                zIndex: 1300,
                alignItems: 'center',
                justifyContent: 'space-around',
                px: 1,
                boxShadow: '0 -4px 20px rgba(0,0,0,0.6)',
            }}
        >
            {ITEMS.map((item) => {
                const active = pathname === item.to;
                return (
                    <Box
                        key={item.to}
                        onClick={() => navigate(item.to)}
                        sx={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: 'pointer',
                            color: active ? C.signal : C.textDim,
                            py: 0.5,
                            px: 1.5,
                            transition: 'all 0.2s ease',
                            borderBottom: active ? `2px solid ${C.signal}` : '2px solid transparent',
                        }}
                    >
                        {item.icon}
                        <Typography sx={{ fontFamily: MONO, fontSize: 8.5, fontWeight: active ? 700 : 500, mt: 0.2, letterSpacing: '0.06em' }}>
                            {item.label}
                        </Typography>
                    </Box>
                );
            })}

            {/* Quick Command Launcher */}
            <Box
                onClick={onOpenCommand}
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    color: C.accent,
                    py: 0.5,
                    px: 1.5,
                }}
            >
                <Search size={19} />
                <Typography sx={{ fontFamily: MONO, fontSize: 8.5, fontWeight: 700, mt: 0.2, letterSpacing: '0.06em' }}>
                    SEARCH
                </Typography>
            </Box>

            {/* All Modules Drawer Opener */}
            <Box
                onClick={onOpenMenu}
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    color: C.textDim,
                    py: 0.5,
                    px: 1.5,
                }}
            >
                <Layers size={19} />
                <Typography sx={{ fontFamily: MONO, fontSize: 8.5, fontWeight: 700, mt: 0.2, letterSpacing: '0.06em' }}>
                    MENU
                </Typography>
            </Box>
        </Box>
    );
}
