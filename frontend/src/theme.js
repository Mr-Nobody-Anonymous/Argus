/**
 * Argus design system.
 *
 * The look is a security operations centre, not a SaaS dashboard: near-black
 * backgrounds so bright video is the only thing that draws the eye, a cyan
 * signal colour reserved for live/system state, and amber/red kept exclusively
 * for alert severity. Nothing decorative is allowed to use an alert colour -
 * in a control room, red must always mean red.
 *
 * Numbers that an operator compares (counts, confidences, timestamps, IDs) are
 * set in a monospace face so digits line up column-to-column and a changing
 * value does not reflow the row next to it.
 */
import { createTheme, alpha } from '@mui/material/styles';

export const DISPLAY = "'Rajdhani', 'Chakra Petch', 'Inter', sans-serif";
export const MONO = "'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', monospace";

// Tactical Cybernetic Operations Palette
export const C = {
    void: '#020617',        // Pitch void background
    panel: '#090d16',       // Deep slate surface
    panelHi: '#0f172a',     // Elevated tactical surface
    line: '#1e293b',        // Subtle boundary
    lineHi: '#38bdf8',      // Cyber blue highlight
    signal: '#00f2fe',      // Laser Cyan (live telemetry)
    signalDim: '#0284c7',
    signalGlow: 'rgba(0, 242, 254, 0.35)',
    accent: '#a855f7',      // AI / Neural embeddings
    accentDim: '#7c3aed',
    accentGlow: 'rgba(168, 85, 247, 0.35)',
    ok: '#10b981',          // Target Lock / Nominal
    warn: '#f59e0b',        // Warning / Caution
    danger: '#f43f5e',      // Alert / Threat
    critical: '#ff0055',    // Emergency Beacon
    text: '#f8fafc',        // Ultra-crisp primary text
    textDim: '#94a3b8',     // Secondary telemetry
    textFaint: '#475569',    // De-emphasized labels
};

/** Severity -> colour. One source of truth across all HUD elements. */
export const priorityColor = (priority) => ({
    critical: C.critical,
    high: C.danger,
    medium: C.warn,
    low: C.signal,
}[String(priority || '').toLowerCase()] || C.textDim);

/** Event lifecycle status -> colour. Mirrors the backend state machine. */
export const statusColor = (status) => ({
    detected: C.signal,
    open: C.warn,
    acknowledged: C.accent,
    resolved: C.ok,
    false_positive: C.textFaint,
}[String(status || '').toLowerCase()] || C.textDim);

/** High-tech tactical glass surface with subtle glow refraction. */
export const glass = (opacity = 0.78, glow = null) => ({
    background: `linear-gradient(145deg, ${alpha(C.panelHi, opacity)} 0%, ${alpha(C.panel, opacity)} 100%)`,
    border: `1px solid ${glow ? alpha(glow, 0.35) : C.line}`,
    backdropFilter: 'blur(16px)',
    boxShadow: glow
        ? `0 0 24px ${alpha(glow, 0.16)}, inset 0 1px 0 rgba(255, 255, 255, 0.08)`
        : `0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.04)`,
});

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: C.signal, dark: C.signalDim },
        secondary: { main: C.accent, dark: C.accentDim },
        success: { main: C.ok },
        warning: { main: C.warn },
        error: { main: C.danger },
        info: { main: C.signal },
        background: { default: C.void, paper: C.panel },
        text: { primary: C.text, secondary: C.textDim },
        divider: C.line,
    },
    shape: { borderRadius: 8 },
    typography: {
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        h1: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase' },
        h2: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: '0.03em', textTransform: 'uppercase' },
        h3: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: '0.03em' },
        h4: { fontFamily: DISPLAY, fontWeight: 700, letterSpacing: '0.02em' },
        h5: { fontFamily: DISPLAY, fontWeight: 650, letterSpacing: '0.02em' },
        h6: { fontFamily: DISPLAY, fontWeight: 650, letterSpacing: '0.02em' },
        subtitle1: { fontFamily: DISPLAY, fontWeight: 600, letterSpacing: '0.04em' },
        subtitle2: { fontFamily: MONO, fontSize: 11, letterSpacing: '0.05em' },
        overline: {
            fontFamily: MONO,
            fontSize: 10,
            letterSpacing: '0.18em',
            fontWeight: 700,
            color: C.textFaint,
            textTransform: 'uppercase',
        },
        button: { fontFamily: DISPLAY, textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.06em' },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    backgroundColor: C.void,
                    backgroundImage: `
                        radial-gradient(1000px 500px at 50% -10%, ${alpha(C.signal, 0.07)} 0%, transparent 60%),
                        radial-gradient(800px 500px at 90% 90%, ${alpha(C.accent, 0.05)} 0%, transparent 50%),
                        linear-gradient(${alpha(C.line, 0.4)} 1px, transparent 1px),
                        linear-gradient(90deg, ${alpha(C.line, 0.4)} 1px, transparent 1px)`,
                    backgroundSize: '100% 100%, 100% 100%, 40px 40px, 40px 40px',
                    backgroundAttachment: 'fixed',
                },
                '*::-webkit-scrollbar': { width: 6, height: 6 },
                '*::-webkit-scrollbar-track': { background: alpha(C.void, 0.8) },
                '*::-webkit-scrollbar-thumb': {
                    background: alpha(C.lineHi, 0.4),
                    borderRadius: 3,
                    border: `1px solid ${alpha(C.signal, 0.1)}`,
                },
                '*::-webkit-scrollbar-thumb:hover': { background: C.signal },
                '@keyframes argus-pulse': {
                    '0%, 100%': { opacity: 1, transform: 'scale(1)', filter: `drop-shadow(0 0 6px ${C.signal})` },
                    '50%': { opacity: 0.4, transform: 'scale(0.85)', filter: 'none' },
                },
                '@keyframes argus-radar-sweep': {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' },
                },
                '@keyframes argus-sweep': {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(300%)' },
                },
                '@keyframes argus-scanlines': {
                    '0%': { backgroundPosition: '0 0' },
                    '100%': { backgroundPosition: '0 100%' },
                },
                '@keyframes argus-glitch': {
                    '0%, 100%': { transform: 'translate(0)' },
                    '20%': { transform: 'translate(-1px, 1px)' },
                    '40%': { transform: 'translate(1px, -1px)' },
                    '60%': { transform: 'translate(-1px, -1px)' },
                    '80%': { transform: 'translate(1px, 1px)' },
                },
                // Optional CRT Scanline overlay effect
                '.argus-crt-overlay': {
                    position: 'relative',
                    '&::after': {
                        content: '""',
                        position: 'fixed',
                        top: 0, left: 0, right: 0, bottom: 0,
                        background: `linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.22) 50%)`,
                        backgroundSize: '100% 3px',
                        pointerEvents: 'none',
                        zIndex: 99999,
                        opacity: 0.6,
                    },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                    backgroundColor: C.panel,
                    border: `1px solid ${C.line}`,
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    fontFamily: MONO,
                    fontSize: 11,
                    fontWeight: 700,
                    borderRadius: 4,
                },
                sizeSmall: { height: 22 },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: '#020617',
                    border: `1px solid ${C.lineHi}`,
                    boxShadow: `0 0 12px ${alpha(C.signal, 0.2)}`,
                    fontSize: 11,
                    fontFamily: MONO,
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 6,
                    letterSpacing: '0.05em',
                    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
                    '&:hover': {
                        boxShadow: `0 0 16px ${alpha(C.signal, 0.35)}`,
                    },
                },
            },
        },
        MuiTableCell: {
            styleOverrides: {
                root: { borderColor: C.line },
                head: {
                    fontFamily: MONO,
                    fontSize: 10,
                    letterSpacing: '0.14em',
                    textTransform: 'uppercase',
                    color: C.textFaint,
                    backgroundColor: C.panel,
                },
            },
        },
    },
});

export default theme;
