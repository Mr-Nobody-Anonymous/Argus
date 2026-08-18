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

export const MONO = "'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', monospace";

// Palette. Kept as plain exports so non-MUI surfaces (canvas overlays, raw
// SVG) can use the identical values instead of eyeballing a near-match.
export const C = {
    void: '#05070a',        // page background
    panel: '#0b0f16',       // card background
    panelHi: '#111823',     // raised / hover
    line: '#1c2735',        // hairline borders
    lineHi: '#2b3a4d',
    signal: '#22d3ee',      // live, connected, system-nominal
    signalDim: '#0e7490',
    accent: '#a78bfa',      // AI / inference provenance
    ok: '#34d399',
    warn: '#fbbf24',
    danger: '#f43f5e',
    critical: '#ff2d55',
    text: '#e6edf5',
    textDim: '#8b9bb0',
    textFaint: '#556275',
};

/** Severity -> colour. One source of truth; every surface must use it. */
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

export const glass = (opacity = 0.72) => ({
    background: `linear-gradient(180deg, ${alpha(C.panelHi, opacity)} 0%, ${alpha(C.panel, opacity)} 100%)`,
    border: `1px solid ${C.line}`,
    backdropFilter: 'blur(10px)',
});

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: { main: C.signal, dark: C.signalDim },
        secondary: { main: C.accent },
        success: { main: C.ok },
        warning: { main: C.warn },
        error: { main: C.danger },
        info: { main: C.signal },
        background: { default: C.void, paper: C.panel },
        text: { primary: C.text, secondary: C.textDim },
        divider: C.line,
    },
    shape: { borderRadius: 10 },
    typography: {
        fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
        h4: { fontWeight: 700, letterSpacing: '-0.02em' },
        h5: { fontWeight: 700, letterSpacing: '-0.01em' },
        h6: { fontWeight: 650, letterSpacing: '-0.01em' },
        overline: {
            fontFamily: MONO,
            fontSize: 10,
            letterSpacing: '0.18em',
            fontWeight: 600,
            color: C.textFaint,
        },
        button: { textTransform: 'none', fontWeight: 600 },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    backgroundColor: C.void,
                    // A faint vignette + grid. Subtle enough not to compete
                    // with video, present enough to read as "instrument".
                    backgroundImage: `
                        radial-gradient(1200px 600px at 70% -10%, ${alpha(C.signal, 0.06)} 0%, transparent 60%),
                        linear-gradient(${alpha(C.line, 0.35)} 1px, transparent 1px),
                        linear-gradient(90deg, ${alpha(C.line, 0.35)} 1px, transparent 1px)`,
                    backgroundSize: '100% 100%, 48px 48px, 48px 48px',
                    backgroundAttachment: 'fixed',
                },
                '*::-webkit-scrollbar': { width: 8, height: 8 },
                '*::-webkit-scrollbar-track': { background: 'transparent' },
                '*::-webkit-scrollbar-thumb': {
                    background: C.lineHi,
                    borderRadius: 8,
                },
                '*::-webkit-scrollbar-thumb:hover': { background: C.signalDim },
                '@keyframes argus-pulse': {
                    '0%, 100%': { opacity: 1, transform: 'scale(1)' },
                    '50%': { opacity: 0.35, transform: 'scale(0.85)' },
                },
                '@keyframes argus-sweep': {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(300%)' },
                },
                '@keyframes argus-flash': {
                    '0%, 100%': { backgroundColor: 'transparent' },
                    '30%': { backgroundColor: alpha(C.danger, 0.16) },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                    backgroundColor: C.panel,
                    border: `1px solid ${C.line}`,
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: { fontFamily: MONO, fontSize: 11, fontWeight: 600 },
                sizeSmall: { height: 22 },
            },
        },
        MuiTooltip: {
            styleOverrides: {
                tooltip: {
                    backgroundColor: '#000',
                    border: `1px solid ${C.lineHi}`,
                    fontSize: 12,
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: { borderRadius: 8 },
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
