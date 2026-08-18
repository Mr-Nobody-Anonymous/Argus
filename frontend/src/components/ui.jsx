/**
 * Shared control-room primitives.
 *
 * Two rules these components exist to enforce:
 *
 * 1. **Never render a fabricated value.** Where a number is unknown the
 *    components show an em-dash, not a zero. A surveillance dashboard that
 *    prints "0 events" when the API call failed is actively dangerous - the
 *    operator reads it as "nothing happened".
 * 2. **Unavailable is a first-class state.** Panels distinguish loading,
 *    empty, and failed, because "no data yet" and "the query broke" demand
 *    different reactions from the person watching.
 */
import React from 'react';
import { Box, Typography, Tooltip, CircularProgress, Skeleton } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { C, MONO, glass } from '../theme';

/** Blinking dot used for live/stale/offline state. */
export function StatusDot({ color = C.ok, pulse = true, size = 8, title }) {
    const dot = (
        <Box
            component="span"
            sx={{
                width: size,
                height: size,
                borderRadius: '50%',
                background: color,
                boxShadow: `0 0 ${size}px ${alpha(color, 0.9)}`,
                display: 'inline-block',
                flexShrink: 0,
                animation: pulse ? 'argus-pulse 1.8s ease-in-out infinite' : 'none',
            }}
        />
    );
    return title ? <Tooltip title={title}>{dot}</Tooltip> : dot;
}

/** Monospace numeric readout. Renders an em-dash for null/undefined/NaN. */
export function Metric({ value, unit, color = C.text, size = 26, dim }) {
    const missing =
        value === null || value === undefined || value === '' ||
        (typeof value === 'number' && !Number.isFinite(value));
    return (
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
            <Typography
                sx={{
                    fontFamily: MONO,
                    fontSize: size,
                    fontWeight: 700,
                    lineHeight: 1.1,
                    color: missing ? C.textFaint : color,
                    fontVariantNumeric: 'tabular-nums',
                }}
            >
                {missing ? '—' : value}
            </Typography>
            {unit && !missing && (
                <Typography sx={{ fontFamily: MONO, fontSize: 11, color: dim || C.textFaint }}>
                    {unit}
                </Typography>
            )}
        </Box>
    );
}

/** Section label in the control-room stencil style. */
export function Label({ children, sx }) {
    return (
        <Typography variant="overline" sx={{ display: 'block', lineHeight: 1.6, ...sx }}>
            {children}
        </Typography>
    );
}

/**
 * Framed panel. `accent` tints the top hairline so a panel can carry severity
 * without repainting its whole surface.
 */
export function Panel({ title, subtitle, right, accent = C.signal, children, sx, dense, ...rest }) {
    return (
        <Box
            sx={{
                ...glass(),
                borderRadius: 2.5,
                position: 'relative',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0, left: 0, right: 0, height: '1px',
                    background: `linear-gradient(90deg, transparent, ${alpha(accent, 0.85)}, transparent)`,
                },
                ...sx,
            }}
            {...rest}
        >
            {(title || right) && (
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: 1,
                        px: dense ? 1.5 : 2,
                        py: dense ? 1 : 1.5,
                        borderBottom: `1px solid ${C.line}`,
                        flexShrink: 0,
                    }}
                >
                    <Box sx={{ minWidth: 0 }}>
                        {title && <Label>{title}</Label>}
                        {subtitle && (
                            <Typography sx={{ fontSize: 12, color: C.textDim, mt: 0.25 }} noWrap>
                                {subtitle}
                            </Typography>
                        )}
                    </Box>
                    {right}
                </Box>
            )}
            <Box sx={{ p: dense ? 1.5 : 2, flex: 1, minHeight: 0 }}>{children}</Box>
        </Box>
    );
}

/**
 * The three non-success states, rendered distinctly.
 *
 * `error` outranks `empty`: a failed query must never be dressed up as "no
 * results", which is the single most misleading thing a monitoring UI can do.
 */
export function PanelState({ loading, error, empty, emptyText = 'No data', children, minHeight = 120 }) {
    if (loading) {
        return (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight, gap: 1.5 }}>
                <CircularProgress size={16} thickness={5} />
                <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint }}>
                    LOADING
                </Typography>
            </Box>
        );
    }
    if (error) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight, gap: 0.75, px: 2 }}>
                <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.danger, letterSpacing: '0.1em' }}>
                    UNAVAILABLE
                </Typography>
                <Typography sx={{ fontSize: 11.5, color: C.textDim, textAlign: 'center' }}>
                    {String(error)}
                </Typography>
            </Box>
        );
    }
    if (empty) {
        return (
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight }}>
                <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint, letterSpacing: '0.1em' }}>
                    {emptyText}
                </Typography>
            </Box>
        );
    }
    return children;
}

/** Small pill. Colour is always caller-supplied so severity stays centralised. */
export function Tag({ label, color = C.textDim, filled, title, sx }) {
    const pill = (
        <Box
            component="span"
            sx={{
                display: 'inline-flex',
                alignItems: 'center',
                px: 0.85,
                height: 20,
                borderRadius: 1,
                fontFamily: MONO,
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                whiteSpace: 'nowrap',
                color: filled ? '#04070c' : color,
                background: filled ? color : alpha(color, 0.12),
                border: `1px solid ${alpha(color, filled ? 0 : 0.4)}`,
                ...sx,
            }}
        >
            {label}
        </Box>
    );
    return title ? <Tooltip title={title}>{pill}</Tooltip> : pill;
}

/** Horizontal proportion bar for distribution readouts. */
export function Bar({ value, max, color = C.signal, height = 6 }) {
    const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
    return (
        <Box sx={{ width: '100%', height, borderRadius: 4, background: alpha(C.line, 0.8), overflow: 'hidden' }}>
            <Box
                sx={{
                    width: `${pct}%`,
                    height: '100%',
                    borderRadius: 4,
                    background: `linear-gradient(90deg, ${alpha(color, 0.55)}, ${color})`,
                    transition: 'width .5s cubic-bezier(.4,0,.2,1)',
                }}
            />
        </Box>
    );
}

export function LoadingRows({ rows = 4, height = 34 }) {
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {Array.from({ length: rows }).map((_, i) => (
                <Skeleton key={i} variant="rounded" height={height} sx={{ bgcolor: alpha(C.line, 0.5) }} />
            ))}
        </Box>
    );
}

/** Relative time. Absolute timestamp stays available on hover. */
export function TimeAgo({ ts, sx }) {
    const date = ts ? new Date(String(ts).replace(' ', 'T')) : null;
    const valid = date && !Number.isNaN(date.getTime());
    const label = React.useMemo(() => {
        if (!valid) return '—';
        const s = Math.max(0, (Date.now() - date.getTime()) / 1000);
        if (s < 10) return 'now';
        if (s < 60) return `${Math.floor(s)}s`;
        if (s < 3600) return `${Math.floor(s / 60)}m`;
        if (s < 86400) return `${Math.floor(s / 3600)}h`;
        return `${Math.floor(s / 86400)}d`;
    }, [ts, valid, date]);

    return (
        <Tooltip title={valid ? date.toLocaleString() : 'no timestamp'}>
            <Typography
                component="span"
                sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint, fontVariantNumeric: 'tabular-nums', ...sx }}
            >
                {label}
            </Typography>
        </Tooltip>
    );
}
