/**
 * Pre-event clip playback for an event.
 *
 * The clip route is role-protected, and a <video src="..."> attribute cannot
 * carry an Authorization header, so the file is fetched through the
 * authenticated axios instance and played from an object URL.
 *
 * The three failure modes are reported distinctly, because they mean different
 * things to the operator:
 *   404 - no clip was written for this rule (expected for most event kinds)
 *   410 - a clip existed and retention has since removed it
 *   else - something is actually broken
 * A player that renders an empty black box for all three teaches the operator
 * to ignore it.
 */
import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Button, CircularProgress } from '@mui/material';
import { alpha } from '@mui/material/styles';
import { Movie, FileDownload, PlayArrow } from '@mui/icons-material';
import { evidenceAPI } from '../services/api';
import { C, MONO } from '../theme';

export default function ClipPlayer({ eventId, ruleType }) {
    const [state, setState] = useState({ url: null, loading: false, error: null, code: null });
    const urlRef = useRef(null);

    useEffect(() => () => {
        // The object URL is owned by this component; free it on unmount.
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    }, []);

    const load = async () => {
        setState({ url: null, loading: true, error: null, code: null });
        try {
            const url = await evidenceAPI.fetchClip(eventId);
            if (urlRef.current) URL.revokeObjectURL(urlRef.current);
            urlRef.current = url;
            setState({ url, loading: false, error: null, code: null });
        } catch (err) {
            const code = err?.response?.status;
            let detail = err?.response?.data?.detail;
            // A blob-typed error body arrives as a Blob, not JSON.
            if (detail instanceof Blob) {
                try { detail = JSON.parse(await detail.text()).detail; } catch { detail = null; }
            }
            const message =
                code === 404 ? (detail || 'No clip was recorded for this event.')
                    : code === 410 ? 'The clip has been removed by the retention policy.'
                        : detail || err?.message || 'Clip could not be loaded.';
            setState({ url: null, loading: false, error: message, code });
        }
    };

    if (state.url) {
        return (
            <Box>
                <Box
                    component="video"
                    src={state.url}
                    controls
                    autoPlay
                    loop
                    sx={{
                        width: '100%', borderRadius: 1.5, display: 'block',
                        border: `1px solid ${C.line}`, background: '#000',
                    }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, flex: 1 }}>
                        PRE-EVENT FOOTAGE · the seconds leading up to the alert
                    </Typography>
                    <Button
                        size="small" startIcon={<FileDownload sx={{ fontSize: 14 }} />}
                        href={state.url} download={`argus_event_${eventId}.mp4`}
                        sx={{ fontSize: 11, color: C.signal }}
                    >
                        Download
                    </Button>
                </Box>
            </Box>
        );
    }

    return (
        <Box sx={{
            p: 2, borderRadius: 1.5, textAlign: 'center',
            border: `1px dashed ${state.error && state.code !== 404 ? alpha(C.warn, 0.4) : C.line}`,
            background: alpha(C.panelHi, 0.4),
        }}>
            {state.loading ? (
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1.5, py: 1 }}>
                    <CircularProgress size={15} thickness={5} />
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint }}>
                        FETCHING CLIP
                    </Typography>
                </Box>
            ) : state.error ? (
                <Box>
                    <Movie sx={{ fontSize: 22, color: state.code === 404 ? C.textFaint : C.warn, mb: 0.5 }} />
                    <Typography sx={{ fontSize: 12, color: C.textDim, mb: 0.5 }}>
                        {state.error}
                    </Typography>
                    {state.code === 404 && (
                        <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint }}>
                            CLIPS ARE EXPORTED ONLY FOR HIGH-VALUE RULES
                        </Typography>
                    )}
                </Box>
            ) : (
                <Button
                    size="small" onClick={load} startIcon={<PlayArrow sx={{ fontSize: 16 }} />}
                    sx={{
                        fontSize: 11.5, color: C.signal,
                        border: `1px solid ${alpha(C.signal, 0.35)}`,
                        '&:hover': { background: alpha(C.signal, 0.1) },
                    }}
                >
                    Load pre-event clip
                </Button>
            )}
        </Box>
    );
}
