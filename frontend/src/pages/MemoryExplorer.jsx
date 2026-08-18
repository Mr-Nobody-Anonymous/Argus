/**
 * Memory Explorer - search what Argus has actually observed.
 *
 * The perception layer stores every actionable observation with the evidence
 * behind it, and an appearance descriptor per tracked person. Ten endpoints
 * served that data and nothing in the UI ever called them, so the analysis was
 * invisible to the people it was built for.
 *
 * Two rules this page is careful about:
 *   1. Every observation is shown WITH its evidence. A conclusion without its
 *      measurements is not reviewable, and this system's whole claim is that
 *      an operator can disagree with it.
 *   2. An appearance match is labelled a candidate, never an identification.
 *      The descriptor compares clothing colour layout - two people in dark
 *      coats match strongly. The API says so and this UI must not quietly
 *      upgrade that into a claim about who someone is.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    Grid,
    Chip,
    TextField,
    MenuItem,
    Button,
    alpha,
    useTheme,
    Alert,
    CircularProgress,
    Divider,
    Tooltip,
    LinearProgress,
} from '@mui/material';
import {
    Search,
    Psychology,
    Warning,
    Insights,
    FactCheck,
    Straighten,
} from '@mui/icons-material';
import { memoryAPI, capabilityAPI } from '../services/api';

const WHEN_OPTIONS = [
    { value: '', label: 'Any time' },
    { value: 'last_hour', label: 'Last hour' },
    { value: 'today', label: 'Today' },
    { value: 'yesterday', label: 'Yesterday' },
    { value: '24h', label: 'Last 24 hours' },
    { value: 'last_week', label: 'Last week' },
];

const KIND_OPTIONS = [
    { value: '', label: 'All kinds' },
    { value: 'dwell', label: 'Dwell' },
    { value: 'pacing', label: 'Pacing' },
    { value: 'abandoned_object', label: 'Abandoned object' },
    { value: 'occupancy_anomaly', label: 'Occupancy anomaly' },
    { value: 'disappeared', label: 'Disappeared' },
    { value: 'scene_change', label: 'Scene change' },
];

function ObservationCard({ observation }) {
    const theme = useTheme();
    const confidence = Number(observation.confidence ?? 0);
    const evidence = observation.evidence || [];

    return (
        <Card
            sx={{
                mb: 2,
                border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                background: alpha(theme.palette.background.paper, 0.6),
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                    <Chip
                        size="small"
                        label={observation.kind}
                        color="primary"
                        variant="outlined"
                    />
                    <Chip
                        size="small"
                        label={`camera ${observation.camera_id ?? '?'}`}
                        variant="outlined"
                    />
                    <Chip
                        size="small"
                        label={`${(confidence * 100).toFixed(0)}% confidence`}
                        color={confidence >= 0.7 ? 'success' : 'warning'}
                        variant="outlined"
                    />
                    {observation.source && (
                        <Tooltip title="Which subsystem made this claim">
                            <Chip size="small" label={observation.source} variant="outlined" />
                        </Tooltip>
                    )}
                </Box>

                <Typography variant="body1" sx={{ mb: 1.5 }}>
                    {observation.summary}
                </Typography>

                {/* Evidence is mandatory, not decorative: the API refuses to
                    store an observation without it. */}
                {evidence.length > 0 ? (
                    <Box>
                        <Typography
                            variant="caption"
                            sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}
                            color="text.secondary"
                        >
                            <FactCheck fontSize="inherit" /> Evidence
                        </Typography>
                        <Box component="ul" sx={{ m: 0, pl: 2.5 }}>
                            {evidence.map((item, i) => (
                                <Typography
                                    key={i}
                                    component="li"
                                    variant="caption"
                                    color="text.secondary"
                                >
                                    {item}
                                </Typography>
                            ))}
                        </Box>
                    </Box>
                ) : (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                        Stored without evidence - this should not happen and is worth reporting.
                    </Alert>
                )}
            </CardContent>
        </Card>
    );
}

export default function MemoryExplorer() {
    const theme = useTheme();
    const [text, setText] = useState('');
    const [when, setWhen] = useState('');
    const [kind, setKind] = useState('');
    const [cameraId, setCameraId] = useState('');
    const [results, setResults] = useState([]);
    const [byKind, setByKind] = useState({});
    const [stats, setStats] = useState(null);
    const [rules, setRules] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [searched, setSearched] = useState(false);

    const loadContext = useCallback(async () => {
        try {
            const [s, r] = await Promise.all([
                memoryAPI.stats(),
                capabilityAPI.rules(),
            ]);
            setStats(s.data);
            setRules(r.data);
        } catch (e) {
            // Non-fatal: the search itself still works.
        }
    }, []);

    useEffect(() => {
        loadContext();
    }, [loadContext]);

    const search = async () => {
        setLoading(true);
        setError(null);
        try {
            const params = { limit: 100 };
            if (text) params.text = text;
            if (when) params.when = when;
            if (kind) params.kind = kind;
            if (cameraId) params.camera_id = cameraId;
            const res = await memoryAPI.recall(params);
            setResults(res.data.results || []);
            setByKind(res.data.by_kind || {});
            setSearched(true);
        } catch (e) {
            setError(e.response?.data?.detail || e.message || 'Search failed');
        } finally {
            setLoading(false);
        }
    };

    const blocked = rules?.summary?.blocked || [];

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" sx={{ mb: 0.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Psychology /> Memory Explorer
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Search everything Argus has observed, with the evidence behind each claim.
            </Typography>

            {/* Honesty panel: what is NOT being analysed right now. Hiding this
                would let an operator assume coverage they do not have. */}
            {blocked.length > 0 && (
                <Alert severity="info" icon={<Warning />} sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {blocked.length} rule{blocked.length > 1 ? 's are' : ' is'} configured but cannot fire here
                    </Typography>
                    {blocked.map((name) => (
                        <Typography key={name} variant="caption" component="div">
                            <strong>{name}</strong>: {(rules.rules[name].blockers || []).join('; ')}
                        </Typography>
                    ))}
                </Alert>
            )}

            {stats && (
                <Grid container spacing={2} sx={{ mb: 3 }}>
                    {[
                        { label: 'Observations stored', value: stats.observations ?? 0, icon: <Insights /> },
                        { label: 'Appearances indexed', value: stats.appearances ?? 0, icon: <Straighten /> },
                        { label: 'Tracks known', value: stats.tracks ?? 0, icon: <Psychology /> },
                        { label: 'Vector backend', value: stats.backend ?? 'sqlite', icon: <Search /> },
                    ].map((card) => (
                        <Grid item xs={6} md={3} key={card.label}>
                            <Card sx={{ background: alpha(theme.palette.primary.main, 0.06) }}>
                                <CardContent>
                                    <Typography variant="caption" color="text.secondary">
                                        {card.label}
                                    </Typography>
                                    <Typography variant="h5">{card.value}</Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            )}

            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                size="small"
                                label="Search text"
                                placeholder="e.g. pacing, loading bay"
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && search()}
                            />
                        </Grid>
                        <Grid item xs={6} md={2}>
                            <TextField
                                select fullWidth size="small" label="When"
                                value={when} onChange={(e) => setWhen(e.target.value)}
                            >
                                {WHEN_OPTIONS.map((o) => (
                                    <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                                ))}
                            </TextField>
                        </Grid>
                        <Grid item xs={6} md={2}>
                            <TextField
                                select fullWidth size="small" label="Kind"
                                value={kind} onChange={(e) => setKind(e.target.value)}
                            >
                                {KIND_OPTIONS.map((o) => (
                                    <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                                ))}
                            </TextField>
                        </Grid>
                        <Grid item xs={6} md={2}>
                            <TextField
                                fullWidth size="small" label="Camera ID" type="number"
                                value={cameraId} onChange={(e) => setCameraId(e.target.value)}
                            />
                        </Grid>
                        <Grid item xs={6} md={2}>
                            <Button
                                fullWidth variant="contained" startIcon={<Search />}
                                onClick={search} disabled={loading}
                            >
                                Search
                            </Button>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            {loading && <LinearProgress sx={{ mb: 2 }} />}
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            {Object.keys(byKind).length > 0 && (
                <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {Object.entries(byKind).map(([k, n]) => (
                        <Chip key={k} size="small" label={`${k}: ${n}`} />
                    ))}
                </Box>
            )}

            {searched && !loading && results.length === 0 && (
                <Alert severity="info">
                    No observations matched. An empty result is a real answer here -
                    it means nothing worth remembering happened in that window.
                </Alert>
            )}

            {results.map((observation, i) => (
                <ObservationCard key={observation.observation_id || i} observation={observation} />
            ))}

            {results.length > 0 && (
                <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="caption" color="text.secondary">
                        Appearance matching compares clothing colour layout, not identity.
                        A strong match is a candidate for human review, never an identification.
                    </Typography>
                </>
            )}
        </Box>
    );
}
