/**
 * Command Center - the operator's default screen.
 *
 * Layout follows how a control room is actually watched: video occupies the
 * majority of the canvas, the alert feed sits on the right where new items
 * enter peripheral vision, and system state is a thin strip that only demands
 * attention when something is wrong.
 *
 * Two deliberate honesty rules, both learned from bugs in this codebase:
 *
 * - **A failed poll never renders as zero.** Panels keep the last good value
 *   and mark the header stale; they do not silently print 0, which an operator
 *   reads as "nothing is happening".
 * - **Delivery state is shown, not assumed.** Argus spent its whole history
 *   recording events and sending them nowhere. If no alert channel can
 *   deliver, this screen says so at the top rather than looking healthy.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    Box, Grid, Typography, Tooltip, IconButton, Dialog, DialogContent,
    ToggleButtonGroup, ToggleButton, Divider, LinearProgress,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Videocam, NotificationsActive, Close, GridView, ViewAgenda,
    Layers, LayersClear, Bolt, Storage, Memory, CloudOff,
} from '@mui/icons-material';

import CameraTile from '../components/CameraTile';
import { Panel, PanelState, Metric, Label, Tag, StatusDot, Bar, TimeAgo } from '../components/ui';
import { C, MONO, priorityColor, statusColor } from '../theme';
import {
    cameraAPI, eventAPI, zoneAPI, systemAPI, capabilityAPI, alertAPI, evidenceAPI,
} from '../services/api';

const POLL_MS = 4000;

/** Poll a request, retaining the last good value and surfacing failure. */
function usePoll(fn, deps = [], interval = POLL_MS) {
    const [state, setState] = useState({ data: null, error: null, loading: true, stale: false });
    const mounted = useRef(true);
    const fnRef = useRef(fn);
    fnRef.current = fn;

    useEffect(() => {
        mounted.current = true;
        let timer;
        const run = async () => {
            try {
                const data = await fnRef.current();
                if (!mounted.current) return;
                setState({ data, error: null, loading: false, stale: false });
            } catch (err) {
                if (!mounted.current) return;
                const detail = err?.response?.data?.detail || err?.message || 'request failed';
                // Keep the previous data and flag it stale rather than
                // blanking the panel to zeros.
                setState((prev) => ({
                    data: prev.data,
                    error: prev.data ? null : detail,
                    loading: false,
                    stale: Boolean(prev.data),
                }));
            } finally {
                if (mounted.current) timer = setTimeout(run, interval);
            }
        };
        run();
        return () => { mounted.current = false; clearTimeout(timer); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, deps);

    return state;
}

function StatCard({ icon, label, value, unit, color = C.signal, hint, sub }) {
    return (
        <Panel dense accent={color} sx={{ height: '100%' }}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.25 }}>
                <Box sx={{
                    width: 34, height: 34, borderRadius: 1.5, flexShrink: 0,
                    display: 'grid', placeItems: 'center',
                    background: alpha(color, 0.1), border: `1px solid ${alpha(color, 0.25)}`,
                }}>
                    {React.cloneElement(icon, { sx: { fontSize: 17, color } })}
                </Box>
                <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Tooltip title={hint || ''}>
                        <Box>
                            <Label sx={{ mb: 0.25 }}>{label}</Label>
                            <Metric value={value} unit={unit} color={color} size={22} />
                            {sub && (
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, mt: 0.25 }} noWrap>
                                    {sub}
                                </Typography>
                            )}
                        </Box>
                    </Tooltip>
                </Box>
            </Box>
        </Panel>
    );
}

function AlertRow({ event, isNew }) {
    const color = priorityColor(event.priority);
    const meta = typeof event.metadata === 'string'
        ? (() => { try { return JSON.parse(event.metadata); } catch { return {}; } })()
        : (event.metadata || {});
    const evidence = Array.isArray(meta.evidence) ? meta.evidence : [];

    return (
        <Box sx={{
            px: 1.25, py: 1, borderRadius: 1.5,
            border: `1px solid ${alpha(color, 0.22)}`,
            background: alpha(color, 0.05),
            animation: isNew ? 'argus-flash 1.4s ease-out' : 'none',
            transition: 'background .2s',
            '&:hover': { background: alpha(color, 0.11) },
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 0.4 }}>
                <StatusDot color={color} pulse={false} size={6} />
                <Typography sx={{
                    fontFamily: MONO, fontSize: 11, fontWeight: 700,
                    color, textTransform: 'uppercase', letterSpacing: '0.04em',
                }} noWrap>
                    {String(event.rule_type || 'event').replace(/_/g, ' ')}
                </Typography>
                <Box sx={{ flex: 1 }} />
                <Tag label={event.status || 'detected'} color={statusColor(event.status)} />
                <TimeAgo ts={event.timestamp} />
            </Box>
            <Typography sx={{ fontSize: 11.5, color: C.textDim, lineHeight: 1.45 }} noWrap>
                {meta.summary || `${event.object_type || 'object'} on camera ${event.camera_id}`}
            </Typography>
            {evidence.length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.4, mt: 0.6 }}>
                    {evidence.slice(0, 3).map((e, i) => (
                        <Typography key={i} sx={{
                            fontFamily: MONO, fontSize: 9.5, color: C.textFaint,
                            px: 0.5, py: 0.15, borderRadius: 0.5,
                            border: `1px solid ${C.line}`,
                        }}>
                            {String(e)}
                        </Typography>
                    ))}
                </Box>
            )}
        </Box>
    );
}

export default function CommandCenter() {
    const [expanded, setExpanded] = useState(null);
    const [layout, setLayout] = useState('grid');
    const [overlays, setOverlays] = useState(true);
    const [zonesByCam, setZonesByCam] = useState({});
    const seenIds = useRef(new Set());
    const [freshIds, setFreshIds] = useState(new Set());

    const cameras = usePoll(async () => (await cameraAPI.getAll()).data, [], 8000);
    const events = usePoll(async () => (await eventAPI.getAll({ limit: 14 })).data, []);
    const stats = usePoll(async () => (await eventAPI.getStats()).data, []);
    const health = usePoll(async () => (await systemAPI.health()).data, [], 10000);
    const rules = usePoll(async () => (await capabilityAPI.rules()).data, [], 15000);
    const delivery = usePoll(async () => (await alertAPI.status()).data, [], 10000);
    const evidence = usePoll(async () => (await evidenceAPI.status()).data, [], 8000);
    const promotion = usePoll(async () => (await capabilityAPI.promotion()).data, [], 15000);

    const cameraList = cameras.data?.cameras || [];
    const eventList = useMemo(() => events.data?.events || [], [events.data]);

    // Flag genuinely new events so they can flash once, without re-flashing
    // every poll. First load seeds the set silently.
    useEffect(() => {
        if (!eventList.length) return;
        if (seenIds.current.size === 0) {
            eventList.forEach((e) => seenIds.current.add(e.id));
            return;
        }
        const fresh = new Set();
        eventList.forEach((e) => {
            if (!seenIds.current.has(e.id)) { fresh.add(e.id); seenIds.current.add(e.id); }
        });
        if (fresh.size) {
            setFreshIds(fresh);
            const t = setTimeout(() => setFreshIds(new Set()), 1600);
            return () => clearTimeout(t);
        }
        return undefined;
    }, [eventList]);

    // Zones are static relative to the poll loop; fetch once per camera set.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            const out = {};
            await Promise.all(cameraList.map(async (cam) => {
                try {
                    const res = await zoneAPI.getAll(cam.id);
                    out[cam.id] = res.data.zones || [];
                } catch { out[cam.id] = []; }
            }));
            if (!cancelled) setZonesByCam(out);
        })();
        return () => { cancelled = true; };
    }, [cameraList.map((c) => c.id).join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

    const online = cameraList.filter((c) => c.status === 'online').length;
    const byPriority = stats.data?.by_priority || {};
    const highCount = (byPriority.high || 0) + (byPriority.critical || 0);
    const ruleMap = rules.data?.rules || {};
    const ruleNames = Object.keys(ruleMap);
    const canFire = ruleNames.filter((r) => ruleMap[r]?.can_fire).length;

    const channels = delivery.data?.channels || [];
    const canDeliver = delivery.data?.any_channel_available;
    const deliveryWarning = delivery.data?.warning;

    const byRule = stats.data?.by_rule || {};
    const ruleMax = Math.max(1, ...Object.values(byRule));

    const handleExpand = useCallback((cam) => setExpanded(cam), []);

    return (
        <Box>
            {/* Delivery banner. Argus recorded events and sent them nowhere for
                its entire history; that failure mode is now impossible to miss. */}
            {delivery.data && !canDeliver && (
                <Box sx={{
                    mb: 2, px: 2, py: 1.25, borderRadius: 2,
                    display: 'flex', alignItems: 'center', gap: 1.5,
                    background: alpha(C.warn, 0.08),
                    border: `1px solid ${alpha(C.warn, 0.35)}`,
                }}>
                    <CloudOff sx={{ fontSize: 18, color: C.warn }} />
                    <Box sx={{ minWidth: 0 }}>
                        <Typography sx={{ fontFamily: MONO, fontSize: 11, fontWeight: 700, color: C.warn, letterSpacing: '0.05em' }}>
                            NO ALERT CHANNEL CAN DELIVER
                        </Typography>
                        <Typography sx={{ fontSize: 11.5, color: C.textDim }}>
                            {deliveryWarning || 'Events are still recorded and queryable, but nothing is being sent anywhere.'}
                        </Typography>
                    </Box>
                    <Box sx={{ flex: 1 }} />
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        {channels.map((ch) => (
                            <Tag
                                key={ch.transport}
                                label={ch.transport}
                                color={ch.available ? C.ok : C.danger}
                                title={ch.reason || (ch.available ? 'available' : 'unavailable')}
                            />
                        ))}
                    </Box>
                </Box>
            )}

            {/* Top metrics */}
            <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={6} md={3}>
                    <StatCard
                        icon={<Videocam />} label="Cameras online" color={online ? C.ok : C.danger}
                        value={cameras.data ? `${online}/${cameraList.length}` : null}
                        hint="Cameras reporting a recent frame"
                        sub={cameras.stale ? 'last known — poll failing' : 'polled live'}
                    />
                </Grid>
                <Grid item xs={6} md={3}>
                    <StatCard
                        icon={<NotificationsActive />} label="Events stored" color={C.signal}
                        value={stats.data?.total ?? null}
                        hint="Total events in the database"
                        sub={`${Object.keys(byRule).length} rule types firing`}
                    />
                </Grid>
                <Grid item xs={6} md={3}>
                    <StatCard
                        icon={<Bolt />} label="High priority" color={highCount ? C.danger : C.textDim}
                        value={stats.data ? highCount : null}
                        hint="High and critical severity events"
                        sub={`${byPriority.medium || 0} medium · ${byPriority.low || 0} low`}
                    />
                </Grid>
                <Grid item xs={6} md={3}>
                    <StatCard
                        icon={<Layers />} label="Rules armed" color={canFire === ruleNames.length ? C.ok : C.warn}
                        value={rules.data ? `${canFire}/${ruleNames.length}` : null}
                        hint="Rules that can actually fire here, not merely enabled in config"
                        sub={canFire < ruleNames.length ? 'some rules are blocked' : 'all rules operational'}
                    />
                </Grid>
            </Grid>

            <Grid container spacing={2}>
                {/* Video wall */}
                <Grid item xs={12} lg={8}>
                    <Panel
                        title="Live video wall"
                        subtitle={`${cameraList.length} camera${cameraList.length === 1 ? '' : 's'} · detections rendered client-side from the inference stream`}
                        right={
                            <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'center' }}>
                                <Tooltip title={overlays ? 'Hide detection overlays' : 'Show detection overlays'}>
                                    <IconButton size="small" onClick={() => setOverlays((v) => !v)}
                                        sx={{ color: overlays ? C.signal : C.textFaint }}>
                                        {overlays ? <Layers sx={{ fontSize: 17 }} /> : <LayersClear sx={{ fontSize: 17 }} />}
                                    </IconButton>
                                </Tooltip>
                                <ToggleButtonGroup
                                    size="small" exclusive value={layout}
                                    onChange={(_, v) => v && setLayout(v)}
                                    sx={{
                                        '& .MuiToggleButton-root': {
                                            py: 0.25, px: 0.75, border: `1px solid ${C.line}`,
                                            color: C.textFaint,
                                            '&.Mui-selected': { color: C.signal, background: alpha(C.signal, 0.12) },
                                        },
                                    }}
                                >
                                    <ToggleButton value="grid"><GridView sx={{ fontSize: 15 }} /></ToggleButton>
                                    <ToggleButton value="single"><ViewAgenda sx={{ fontSize: 15 }} /></ToggleButton>
                                </ToggleButtonGroup>
                            </Box>
                        }
                    >
                        <PanelState
                            loading={cameras.loading}
                            error={cameras.error}
                            empty={!cameraList.length}
                            emptyText="NO CAMERAS REGISTERED"
                            minHeight={280}
                        >
                            <Grid container spacing={1.5}>
                                {cameraList.map((cam) => {
                                    // One camera should fill the wall rather than
                                    // sit in a half-width box beside dead space.
                                    const cols = layout === 'single' || cameraList.length === 1
                                        ? 12 : cameraList.length <= 4 ? 6 : 4;
                                    const tileH = cols === 12 ? 460 : cols === 6 ? 250 : 190;
                                    return (
                                        <Grid item xs={12} md={cols} key={cam.id}>
                                            <CameraTile
                                                camera={cam}
                                                zones={zonesByCam[cam.id] || []}
                                                height={tileH}
                                                onExpand={handleExpand}
                                                showOverlays={overlays}
                                                compact={cols === 4}
                                            />
                                        </Grid>
                                    );
                                })}
                            </Grid>
                        </PanelState>
                    </Panel>

                    {/* Rule status strip */}
                    <Panel
                        title="Detection rules"
                        subtitle="Configured is not the same as operational — a blocked rule names its blocker"
                        sx={{ mt: 2 }}
                        accent={canFire === ruleNames.length ? C.ok : C.warn}
                    >
                        <PanelState loading={rules.loading} error={rules.error} empty={!ruleNames.length} minHeight={80}>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {ruleNames.map((name) => {
                                    const r = ruleMap[name];
                                    const ok = r.can_fire;
                                    const blockers = r.blockers || [];
                                    return (
                                        <Tooltip
                                            key={name}
                                            title={ok
                                                ? `${r.description || name} — operational`
                                                : `Blocked: ${blockers.join('; ') || 'unknown reason'}`}
                                        >
                                            <Box sx={{
                                                display: 'flex', alignItems: 'center', gap: 0.75,
                                                px: 1, py: 0.6, borderRadius: 1.5,
                                                border: `1px solid ${alpha(ok ? C.ok : C.warn, 0.3)}`,
                                                background: alpha(ok ? C.ok : C.warn, 0.06),
                                            }}>
                                                <StatusDot color={ok ? C.ok : C.warn} pulse={false} size={6} />
                                                <Typography sx={{
                                                    fontFamily: MONO, fontSize: 10.5,
                                                    color: ok ? C.text : C.warn, textTransform: 'uppercase',
                                                }}>
                                                    {name.replace(/_/g, ' ')}
                                                </Typography>
                                            </Box>
                                        </Tooltip>
                                    );
                                })}
                            </Box>
                        </PanelState>
                    </Panel>

                    <Grid container spacing={2} sx={{ mt: 0 }}>
                        <Grid item xs={12} md={6}>
                            <Panel
                                title="Pre-event evidence"
                                subtitle="Rolling buffer held per camera, ready to export on alert"
                                accent={C.accent}
                                sx={{ height: '100%' }}
                            >
                                <PanelState loading={evidence.loading} error={evidence.error} minHeight={110}>
                                    <Box sx={{ display: 'flex', gap: 2, mb: 1.5 }}>
                                        <Box>
                                            <Label>Buffered</Label>
                                            <Metric
                                                value={evidence.data?.total_buffered_mb}
                                                unit="MB" size={20} color={C.accent}
                                            />
                                        </Box>
                                        <Box>
                                            <Label>Window</Label>
                                            <Metric
                                                value={evidence.data?.pre_event_seconds}
                                                unit="s" size={20} color={C.textDim}
                                            />
                                        </Box>
                                        <Box>
                                            <Label>Clips written</Label>
                                            <Metric
                                                value={evidence.data?.counters?.clips_written}
                                                size={20}
                                                color={evidence.data?.counters?.clips_written ? C.ok : C.textDim}
                                            />
                                        </Box>
                                    </Box>
                                    {Object.entries(evidence.data?.buffers || {}).map(([cid, b]) => (
                                        <Box key={cid} sx={{ mb: 1 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.4 }}>
                                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim }}>
                                                    CAM {cid} · {b.seconds_buffered}s
                                                </Typography>
                                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }}>
                                                    {b.mb} / {b.max_mb} MB
                                                </Typography>
                                            </Box>
                                            <Bar value={b.mb} max={b.max_mb} color={C.accent} />
                                        </Box>
                                    ))}
                                    <Typography sx={{ fontSize: 10.5, color: C.textFaint, mt: 1, lineHeight: 1.5 }}>
                                        Clips are exported only for{' '}
                                        {(evidence.data?.clip_rules || []).length} high-value rules —
                                        frames are JPEG-encoded so the buffer costs megabytes, not gigabytes.
                                    </Typography>
                                </PanelState>
                            </Panel>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Panel
                                title="Observation promotion"
                                subtitle="What perception saw vs. what became an alert"
                                accent={C.signal}
                                sx={{ height: '100%' }}
                            >
                                <PanelState loading={promotion.loading} error={promotion.error} minHeight={110}>
                                    {(() => {
                                        const st = promotion.data?.stats || {};
                                        const rows = [
                                            ['Promoted to events', st.promoted, C.ok],
                                            ['Suppressed as duplicate', st.suppressed_duplicate, C.textDim],
                                            ['Suppressed by policy', st.suppressed_by_policy, C.textDim],
                                            ['Not actionable', st.suppressed_not_actionable, C.textDim],
                                        ];
                                        const max = Math.max(1, ...rows.map((r) => r[1] || 0));
                                        return (
                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.1 }}>
                                                {rows.map(([label, val, col]) => (
                                                    <Box key={label}>
                                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.4 }}>
                                                            <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim, textTransform: 'uppercase' }}>
                                                                {label}
                                                            </Typography>
                                                            <Typography sx={{ fontFamily: MONO, fontSize: 10.5, color: C.text }}>
                                                                {val ?? '—'}
                                                            </Typography>
                                                        </Box>
                                                        <Bar value={val || 0} max={max} color={col} />
                                                    </Box>
                                                ))}
                                                <Typography sx={{ fontSize: 10.5, color: C.textFaint, mt: 0.5, lineHeight: 1.5 }}>
                                                    Suppression is deliberate: bookkeeping observations stay
                                                    queryable in memory without paging anyone.
                                                </Typography>
                                            </Box>
                                        );
                                    })()}
                                </PanelState>
                            </Panel>
                        </Grid>
                    </Grid>
                </Grid>

                {/* Right rail */}
                <Grid item xs={12} lg={4}>
                    <Panel
                        title="Alert feed"
                        subtitle="Newest first"
                        accent={C.danger}
                        right={<StatusDot color={events.stale ? C.warn : C.ok} title={events.stale ? 'poll failing — showing last known' : 'live'} />}
                        sx={{ mb: 2 }}
                    >
                        <PanelState
                            loading={events.loading}
                            error={events.error}
                            empty={!eventList.length}
                            emptyText="NO EVENTS RECORDED"
                            minHeight={200}
                        >
                            <Box sx={{
                                display: 'flex', flexDirection: 'column', gap: 0.85,
                                maxHeight: 520, overflowY: 'auto', pr: 0.5,
                            }}>
                                {eventList.map((e) => (
                                    <AlertRow key={e.id} event={e} isNew={freshIds.has(e.id)} />
                                ))}
                            </Box>
                        </PanelState>
                    </Panel>

                    <Panel title="Event distribution" subtitle="By rule type" sx={{ mb: 2 }}>
                        <PanelState
                            loading={stats.loading} error={stats.error}
                            empty={!Object.keys(byRule).length} minHeight={100}
                        >
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.1 }}>
                                {Object.entries(byRule)
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([rule, count]) => (
                                        <Box key={rule}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.4 }}>
                                                <Typography sx={{
                                                    fontFamily: MONO, fontSize: 10.5, color: C.textDim,
                                                    textTransform: 'uppercase',
                                                }}>
                                                    {rule.replace(/_/g, ' ')}
                                                </Typography>
                                                <Typography sx={{
                                                    fontFamily: MONO, fontSize: 10.5, color: C.text,
                                                    fontVariantNumeric: 'tabular-nums',
                                                }}>
                                                    {count}
                                                </Typography>
                                            </Box>
                                            <Bar value={count} max={ruleMax} color={C.signal} />
                                        </Box>
                                    ))}
                            </Box>
                        </PanelState>
                    </Panel>

                    <Panel title="System" subtitle="Backend subsystems" accent={C.accent}>
                        <PanelState loading={health.loading} error={health.error} minHeight={90}>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.9 }}>
                                <SubsystemRow
                                    icon={<Storage />} label="Database"
                                    ok={health.data?.subsystems?.database === 'ok'}
                                    detail={health.data?.subsystems?.database || '—'}
                                />
                                <SubsystemRow
                                    icon={<Memory />} label="Retention"
                                    ok={health.data?.subsystems?.retention?.running}
                                    detail={health.data?.subsystems?.retention?.running ? 'running' : 'stopped'}
                                />
                                <SubsystemRow
                                    icon={<NotificationsActive />} label="Alert delivery"
                                    ok={canDeliver}
                                    detail={canDeliver ? 'channel available' : 'nothing can deliver'}
                                />
                                <Divider sx={{ my: 0.5 }} />
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }}>
                                        VERSION
                                    </Typography>
                                    <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim }}>
                                        {health.data?.version || '—'}
                                    </Typography>
                                </Box>
                            </Box>
                        </PanelState>
                    </Panel>
                </Grid>
            </Grid>

            {/* Expanded feed */}
            <Dialog
                open={Boolean(expanded)} onClose={() => setExpanded(null)}
                maxWidth="lg" fullWidth
                PaperProps={{ sx: { background: C.void, border: `1px solid ${C.lineHi}` } }}
            >
                <DialogContent sx={{ p: 1.5, position: 'relative' }}>
                    <IconButton
                        size="small" onClick={() => setExpanded(null)}
                        sx={{
                            position: 'absolute', top: 12, right: 12, zIndex: 3,
                            bgcolor: alpha('#000', 0.65), color: C.text,
                            '&:hover': { bgcolor: alpha(C.danger, 0.3) },
                        }}
                    >
                        <Close sx={{ fontSize: 18 }} />
                    </IconButton>
                    {expanded && (
                        <CameraTile
                            camera={expanded}
                            zones={zonesByCam[expanded.id] || []}
                            height={620}
                            showOverlays={overlays}
                        />
                    )}
                </DialogContent>
            </Dialog>
        </Box>
    );
}

function SubsystemRow({ icon, label, ok, detail }) {
    const color = ok ? C.ok : C.danger;
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {React.cloneElement(icon, { sx: { fontSize: 15, color } })}
            <Typography sx={{ fontFamily: MONO, fontSize: 10.5, color: C.textDim, flex: 1 }}>
                {label}
            </Typography>
            <Tag label={detail} color={color} />
        </Box>
    );
}
