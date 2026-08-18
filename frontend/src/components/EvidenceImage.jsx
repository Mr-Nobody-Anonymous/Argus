/**
 * Authenticated snapshot viewer.
 *
 * Snapshots are event evidence containing identifiable people, so the bare
 * /snapshots static mount only exists when auth is disabled. The real route is
 *
 *     GET /api/snapshots/{camera_id}/{filename}   (viewer role required)
 *
 * A plain <img src="/snapshots/foo.jpg"> therefore 404s on every authenticated
 * deployment - which is exactly what the event dialog was doing, rendering a
 * broken-image icon next to real evidence. The browser cannot attach a bearer
 * token to an <img> request, so the bytes are fetched through axios and shown
 * from an object URL.
 */
import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { ImageNotSupported } from '@mui/icons-material';
import { snapshotAPI } from '../services/api';
import { C, MONO } from '../theme';

/** Snapshots are stored as absolute paths; the API wants camera id + filename. */
export function snapshotParts(snapshotPath, cameraId) {
    if (!snapshotPath) return null;
    const filename = String(snapshotPath).split(/[\\/]/).pop();
    if (!filename) return null;
    // Filenames are written as cam{ID}_{rule}_{timestamp}.jpg. Prefer the id
    // encoded in the name; fall back to the event's camera_id.
    const match = /^cam(\d+)_/.exec(filename);
    const id = match ? match[1] : cameraId;
    if (id == null) return null;
    return { cameraId: id, filename };
}

export default function EvidenceImage({ snapshotPath, cameraId, height = 260 }) {
    const [state, setState] = useState({ url: null, error: null, loading: true });
    const urlRef = useRef(null);

    useEffect(() => {
        let alive = true;
        const parts = snapshotParts(snapshotPath, cameraId);
        if (!parts) {
            setState({ url: null, error: 'No snapshot recorded', loading: false });
            return undefined;
        }
        setState({ url: null, error: null, loading: true });
        (async () => {
            try {
                const url = await snapshotAPI.fetch(parts.cameraId, parts.filename);
                if (!alive) { URL.revokeObjectURL(url); return; }
                if (urlRef.current) URL.revokeObjectURL(urlRef.current);
                urlRef.current = url;
                setState({ url, error: null, loading: false });
            } catch (err) {
                if (!alive) return;
                const code = err?.response?.status;
                setState({
                    url: null,
                    loading: false,
                    error: code === 404
                        ? 'Snapshot file is no longer on disk (retention)'
                        : 'Snapshot could not be loaded',
                });
            }
        })();
        return () => { alive = false; };
    }, [snapshotPath, cameraId]);

    useEffect(() => () => {
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    }, []);

    if (state.loading) {
        return (
            <Box sx={{
                height, display: 'grid', placeItems: 'center',
                borderRadius: 1.5, border: `1px solid ${C.line}`,
                background: alpha(C.panelHi, 0.5),
            }}>
                <CircularProgress size={16} thickness={5} />
            </Box>
        );
    }

    if (state.error) {
        return (
            <Box sx={{
                height, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: 0.75,
                borderRadius: 1.5, border: `1px dashed ${C.line}`,
                background: alpha(C.panelHi, 0.4),
            }}>
                <ImageNotSupported sx={{ fontSize: 22, color: C.textFaint }} />
                <Typography sx={{ fontFamily: MONO, fontSize: 10.5, color: C.textFaint }}>
                    {state.error}
                </Typography>
            </Box>
        );
    }

    return (
        <Box
            component="img"
            src={state.url}
            alt="Event snapshot"
            sx={{
                width: '100%', maxHeight: height * 1.6, objectFit: 'contain',
                borderRadius: 1.5, border: `1px solid ${C.line}`,
                background: '#000', display: 'block',
            }}
        />
    );
}
