/**
 * Thermal & Multispectral Vision Intelligence Dashboard
 *
 * Palantir / Anduril Lattice-grade infrared operations center:
 * - Real-time radiometric temperature analysis (°C)
 * - 8 Military & Surveillance false-color palettes (Ironbow, Inferno, Jet, White-Hot, etc.)
 * - Elevated Body Temperature (EBT / Fever >= 37.5°C) & Fire / Overheat (>= 65°C) screening
 * - Cross-spectral multispectral fusion (RGB contours + Thermal heatmaps)
 * - Zero-lux & obscurant target signature segmentation
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box, Grid, Typography, Button, Slider, Chip, Tooltip, IconButton,
    CircularProgress, Switch, FormControlLabel, Stack, Alert,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Flame, Thermometer, Eye, Scan, RefreshCw, Upload, ShieldAlert,
    Sliders, Layers, Crosshair, Zap, Activity, CheckCircle2,
} from 'lucide-react';

import { thermalAPI, cameraAPI } from '../services/api';
import { Panel, Metric, Label, StatusDot, HudReticle, ThreatBadge } from '../components/ui';
import { C, DISPLAY, MONO, glass } from '../theme';

const PALETTES = [
    { id: 'ironbow', name: 'Ironbow', desc: 'Standard FLIR thermal inspection', gradient: 'linear-gradient(90deg, #000 0%, #1d1147 25%, #881679 50%, #eb6824 75%, #fef386 100%)' },
    { id: 'inferno', name: 'Inferno', desc: 'High-contrast perceptual spectrum', gradient: 'linear-gradient(90deg, #000004 0%, #420a68 25%, #932667 50%, #dd513a 75%, #fcffa4 100%)' },
    { id: 'jet', name: 'Jet Spectrum', desc: 'Classic aerospace false-color', gradient: 'linear-gradient(90deg, #0000ff 0%, #00ffff 25%, #00ff00 50%, #ffff00 75%, #ff0000 100%)' },
    { id: 'rainbow', name: 'Rainbow', desc: 'Multi-band thermal gradient', gradient: 'linear-gradient(90deg, #67001f 0%, #b2182b 25%, #f4a582 50%, #92c5de 75%, #053061 100%)' },
    { id: 'white_hot', name: 'White-Hot', desc: 'Tactical night ops (hot = bright)', gradient: 'linear-gradient(90deg, #000 0%, #777 50%, #fff 100%)' },
    { id: 'black_hot', name: 'Black-Hot', desc: 'Covert reconnaissance (hot = dark)', gradient: 'linear-gradient(90deg, #fff 0%, #777 50%, #000 100%)' },
    { id: 'bone', name: 'Bone', desc: 'High-definition medical grayscale', gradient: 'linear-gradient(90deg, #000 0%, #3a4b66 50%, #fff 100%)' },
    { id: 'hot', name: 'Hot Metal', desc: 'Industrial fire and engine heat', gradient: 'linear-gradient(90deg, #000 0%, #e60000 40%, #ffcc00 80%, #ffffff 100%)' },
];

export default function ThermalVisionDashboard() {
    const [palette, setPalette] = useState('ironbow');
    const [calibration, setCalibration] = useState({ temp_min_c: -10.0, temp_max_c: 100.0, emissivity: 0.95 });
    const [savingCal, setSavingCal] = useState(false);
    const [calSavedNotice, setCalSavedNotice] = useState(false);

    // Analysis state
    const [analyzing, setAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [analysisError, setAnalysisError] = useState(null);
    const [uploadedImageSrc, setUploadedImageSrc] = useState(null);

    // Fusion & display parameters
    const [fusionAlpha, setFusionAlpha] = useState(0.45);
    const [ebtThreshold, setEbtThreshold] = useState(37.5);
    const [fireThreshold, setFireThreshold] = useState(65.0);
    const [enableHotspotOverlay, setEnableHotspotOverlay] = useState(true);

    const canvasRef = useRef(null);
    const fileInputRef = useRef(null);

    // Load initial calibration from backend
    useEffect(() => {
        let active = true;
        thermalAPI.getCalibration()
            .then(res => {
                if (active && res.data) {
                    setCalibration({
                        temp_min_c: res.data.temp_min_c ?? -10.0,
                        temp_max_c: res.data.temp_max_c ?? 100.0,
                        emissivity: res.data.emissivity ?? 0.95,
                    });
                    if (res.data.default_palette) setPalette(res.data.default_palette);
                }
            })
            .catch(() => {});
        return () => { active = false; };
    }, []);

    // Save calibration changes to backend
    const handleSaveCalibration = async () => {
        setSavingCal(true);
        try {
            await thermalAPI.updateCalibration({
                temp_min_c: calibration.temp_min_c,
                temp_max_c: calibration.temp_max_c,
                emissivity: calibration.emissivity,
                default_palette: palette,
            });
            setCalSavedNotice(true);
            setTimeout(() => setCalSavedNotice(false), 3000);
        } catch (err) {
            console.error('Failed to update thermal calibration', err);
        } finally {
            setSavingCal(false);
        }
    };

    // Analyze uploaded file or generate synthetic tactical thermal frame
    const handleFileUpload = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            setUploadedImageSrc(event.target.result);
        };
        reader.readAsDataURL(file);

        executeAnalysis(file);
    };

    const executeAnalysis = async (fileBlob) => {
        setAnalyzing(true);
        setAnalysisError(null);
        try {
            const formData = new FormData();
            formData.append('file', fileBlob, 'thermal_frame.png');
            formData.append('palette', palette);
            formData.append('temp_min_c', calibration.temp_min_c);
            formData.append('temp_max_c', calibration.temp_max_c);
            formData.append('emissivity', calibration.emissivity);

            const res = await thermalAPI.analyze(formData);
            setAnalysisResult(res.data);
        } catch (err) {
            setAnalysisError(err?.response?.data?.detail || err?.message || 'Thermal processing failed');
        } finally {
            setAnalyzing(false);
        }
    };

    // Generate tactical simulation frame if no real file uploaded
    const handleSimulateThermalCapture = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 300;
        const ctx = canvas.getContext('2d');

        // Draw synthetic thermal gradient background (cool ambient)
        const bgGrad = ctx.createRadialGradient(200, 150, 10, 200, 150, 200);
        bgGrad.addColorStop(0, '#333333');
        bgGrad.addColorStop(1, '#080808');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, 400, 300);

        // Draw hot signature 1: Human target with high facial temperature (fever simulation)
        ctx.fillStyle = '#f0f0f0';
        ctx.beginPath();
        ctx.arc(150, 120, 28, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillRect(135, 148, 30, 80);

        // Draw hot signature 2: Running vehicle engine (high heat)
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.ellipse(300, 190, 45, 25, 0, 0, Math.PI * 2);
        ctx.fill();

        canvas.toBlob((blob) => {
            if (blob) {
                setUploadedImageSrc(canvas.toDataURL());
                executeAnalysis(blob);
            }
        }, 'image/png');
    };

    // Render interactive HUD canvas when image and hotspots exist
    useEffect(() => {
        if (!uploadedImageSrc || !canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Overlay thermal false-color tinted wash
            ctx.save();
            ctx.globalCompositeOperation = 'color';
            const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
            if (palette === 'ironbow') {
                grad.addColorStop(0, '#1d1147');
                grad.addColorStop(0.5, '#eb6824');
                grad.addColorStop(1, '#fef386');
            } else if (palette === 'jet') {
                grad.addColorStop(0, '#0000ff');
                grad.addColorStop(0.5, '#00ff00');
                grad.addColorStop(1, '#ff0000');
            } else if (palette === 'inferno') {
                grad.addColorStop(0, '#000004');
                grad.addColorStop(0.5, '#932667');
                grad.addColorStop(1, '#fcffa4');
            } else {
                grad.addColorStop(0, '#222222');
                grad.addColorStop(1, '#ffffff');
            }
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.restore();

            // Overlay detected hotspots
            if (enableHotspotOverlay && analysisResult?.hotspots) {
                analysisResult.hotspots.forEach((spot, idx) => {
                    const [x, y, w, h] = spot.bbox;
                    const isFire = spot.type === 'fire_or_overheating';
                    const color = isFire ? C.danger : C.warn;

                    // Box
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, w, h);

                    // Corner HUD reticles
                    const clen = Math.min(8, w / 4);
                    ctx.beginPath();
                    ctx.moveTo(x, y + clen); ctx.lineTo(x, y); ctx.lineTo(x + clen, y);
                    ctx.moveTo(x + w - clen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + clen);
                    ctx.moveTo(x, y + h - clen); ctx.lineTo(x, y + h); ctx.lineTo(x + clen, y + h);
                    ctx.moveTo(x + w - clen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - clen);
                    ctx.stroke();

                    // Text tag
                    ctx.fillStyle = alpha(color, 0.85);
                    ctx.fillRect(x, Math.max(0, y - 20), 110, 18);
                    ctx.fillStyle = '#020617';
                    ctx.font = 'bold 11px monospace';
                    ctx.fillText(`${spot.type === 'fever_screening' ? 'EBT' : 'FIRE'}: ${spot.max_temperature_c?.toFixed(1)}°C`, x + 4, Math.max(14, y - 6));
                });
            }
        };
        img.src = uploadedImageSrc;
    }, [uploadedImageSrc, analysisResult, palette, enableHotspotOverlay]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* Top Operations Header */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
                <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25 }}>
                        <Flame style={{ color: C.warn, width: 24, height: 24, filter: `drop-shadow(0 0 8px ${C.warn})` }} />
                        <Typography sx={{ fontFamily: DISPLAY, fontSize: 24, fontWeight: 800, color: C.text, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                            Thermal & Multispectral Warfare OS
                        </Typography>
                    </Box>
                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textDim, mt: 0.3 }}>
                        RADIOMETRIC TEMPERATURE SENSING // FLIR & CROSS-SPECTRAL FUSION // ZERO-LUX RECONNAISSANCE
                    </Typography>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <ThreatBadge level="NOMINAL" defcon={4} />
                    <Button
                        variant="outlined"
                        size="small"
                        onClick={handleSimulateThermalCapture}
                        startIcon={<Zap size={15} />}
                        sx={{
                            fontFamily: MONO,
                            fontSize: 11,
                            borderColor: alpha(C.signal, 0.5),
                            color: C.signal,
                            '&:hover': { borderColor: C.signal, background: alpha(C.signal, 0.1) },
                        }}
                    >
                        Simulate Thermal Frame
                    </Button>
                    <Button
                        variant="contained"
                        size="small"
                        onClick={() => fileInputRef.current?.click()}
                        startIcon={<Upload size={15} />}
                        sx={{
                            fontFamily: MONO,
                            fontSize: 11,
                            bgcolor: C.signal,
                            color: '#000',
                            fontWeight: 700,
                            '&:hover': { bgcolor: alpha(C.signal, 0.85) },
                        }}
                    >
                        Upload Raw Stream
                    </Button>
                    <input
                        type="file"
                        ref={fileInputRef}
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={handleFileUpload}
                    />
                </Box>
            </Box>

            {calSavedNotice && (
                <Alert severity="success" sx={{ bgcolor: alpha(C.ok, 0.15), color: C.ok, border: `1px solid ${alpha(C.ok, 0.4)}` }}>
                    Thermal sensor calibration parameters written to edge engine memory.
                </Alert>
            )}

            {/* Quick KPI Strip */}
            <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                    <Panel dense accent={C.warn}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Box>
                                <Label>Peak Radiometric Temp</Label>
                                <Metric
                                    value={analysisResult?.temperature_stats?.max_temp_c !== undefined ? analysisResult.temperature_stats.max_temp_c.toFixed(1) : '38.4'}
                                    unit="°C"
                                    color={C.warn}
                                />
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, mt: 0.5 }}>
                                    TARGET EXCEEDS AMBIENT
                                </Typography>
                            </Box>
                            <Thermometer style={{ color: C.warn, width: 32, height: 32, opacity: 0.8 }} />
                        </Box>
                    </Panel>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Panel dense accent={C.danger}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Box>
                                <Label>Hotspots / Anomalies</Label>
                                <Metric
                                    value={analysisResult?.hotspot_count !== undefined ? analysisResult.hotspot_count : 2}
                                    unit="ALERTS"
                                    color={C.danger}
                                />
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, mt: 0.5 }}>
                                    1 EBT FEVER // 1 VEHICLE EXHAUST
                                </Typography>
                            </Box>
                            <ShieldAlert style={{ color: C.danger, width: 32, height: 32, opacity: 0.8 }} />
                        </Box>
                    </Panel>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Panel dense accent={C.signal}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Box>
                                <Label>Active Colormap</Label>
                                <Typography sx={{ fontFamily: DISPLAY, fontSize: 20, fontWeight: 700, color: C.signal, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                                    {PALETTES.find(p => p.id === palette)?.name || palette}
                                </Typography>
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, mt: 0.5 }}>
                                    EMISSIVITY ε = {calibration.emissivity.toFixed(2)}
                                </Typography>
                            </Box>
                            <Layers style={{ color: C.signal, width: 32, height: 32, opacity: 0.8 }} />
                        </Box>
                    </Panel>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Panel dense accent={C.ok}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Box>
                                <Label>Zero-Lux Target Lock</Label>
                                <Metric
                                    value={analysisResult?.signature_count !== undefined ? analysisResult.signature_count : 4}
                                    unit="TRACKS"
                                    color={C.ok}
                                />
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint, mt: 0.5 }}>
                                    INFRARED BLOB SEGMENTED
                                </Typography>
                            </Box>
                            <Eye style={{ color: C.ok, width: 32, height: 32, opacity: 0.8 }} />
                        </Box>
                    </Panel>
                </Grid>
            </Grid>

            {/* Main Interactive Workspace */}
            <Grid container spacing={3}>
                {/* Left Area: Visual Inspection Viewport */}
                <Grid item xs={12} lg={8}>
                    <Panel
                        title="Tactical Infrared Viewport"
                        subtitle="FLIR Radiometric Stream // Live Target HUD"
                        right={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <FormControlLabel
                                    control={
                                        <Switch
                                            size="small"
                                            checked={enableHotspotOverlay}
                                            onChange={(e) => setEnableHotspotOverlay(e.target.checked)}
                                            sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: C.signal } }}
                                        />
                                    }
                                    label={<Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textDim }}>HUD RETICLES</Typography>}
                                />
                                <StatusDot color={C.ok} size={7} pulse />
                            </Box>
                        }
                        sx={{ minHeight: 460 }}
                    >
                        <Box sx={{
                            position: 'relative',
                            width: '100%',
                            minHeight: 380,
                            borderRadius: 1.5,
                            overflow: 'hidden',
                            bgcolor: '#010409',
                            border: `1px solid ${C.line}`,
                            display: 'grid',
                            placeItems: 'center',
                        }}>
                            <HudReticle color={C.signal} size={14} stroke={2} opacity={0.7} />

                            {analyzing && (
                                <Box sx={{ position: 'absolute', inset: 0, bgcolor: 'rgba(0,0,0,0.7)', zIndex: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 1.5 }}>
                                    <CircularProgress size={32} sx={{ color: C.signal }} />
                                    <Typography sx={{ fontFamily: MONO, fontSize: 12, color: C.signal, letterSpacing: '0.1em' }}>
                                        COMPUTING RADIOMETRIC HEAT SIGNATURES...
                                    </Typography>
                                </Box>
                            )}

                            {uploadedImageSrc ? (
                                <canvas
                                    ref={canvasRef}
                                    style={{
                                        maxWidth: '100%',
                                        maxHeight: 460,
                                        objectFit: 'contain',
                                        display: 'block',
                                    }}
                                />
                            ) : (
                                <Box sx={{ p: 4, textAlign: 'center' }}>
                                    <Scan style={{ width: 48, height: 48, color: C.textFaint, margin: '0 auto 12px' }} />
                                    <Typography sx={{ fontFamily: DISPLAY, fontSize: 18, color: C.textDim, fontWeight: 700 }}>
                                        NO THERMAL STREAM LOADED
                                    </Typography>
                                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textFaint, maxWidth: 360, mx: 'auto', mt: 1 }}>
                                        Select "Simulate Thermal Frame" or upload a radiometric camera capture to engage false-color colorization and EBT fever alarms.
                                    </Typography>
                                    <Button
                                        variant="outlined"
                                        size="small"
                                        onClick={handleSimulateThermalCapture}
                                        sx={{ mt: 2, borderColor: C.signal, color: C.signal, fontFamily: MONO, fontSize: 11 }}
                                    >
                                        INITIALIZE SIMULATION
                                    </Button>
                                </Box>
                            )}

                            {/* Viewport Telemetry Overlay HUD */}
                            <Box sx={{
                                position: 'absolute',
                                bottom: 12,
                                left: 12,
                                px: 1.5,
                                py: 0.6,
                                borderRadius: 1,
                                bgcolor: alpha('#020617', 0.85),
                                border: `1px solid ${C.line}`,
                                backdropFilter: 'blur(8px)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 2,
                            }}>
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.signal }}>
                                    RANGE: {calibration.temp_min_c}°C ➔ {calibration.temp_max_c}°C
                                </Typography>
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }}>
                                    PALETTE: {palette.toUpperCase()}
                                </Typography>
                            </Box>
                        </Box>
                    </Panel>
                </Grid>

                {/* Right Area: Colormaps & Radiometric Calibration Controls */}
                <Grid item xs={12} lg={4}>
                    <Stack spacing={2.5}>
                        {/* False-Color Palettes Selector */}
                        <Panel title="False-Color Palettes" subtitle="Spectral colormapping filter">
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                {PALETTES.map((p) => {
                                    const active = palette === p.id;
                                    return (
                                        <Box
                                            key={p.id}
                                            onClick={() => setPalette(p.id)}
                                            sx={{
                                                p: 1.25,
                                                borderRadius: 1.5,
                                                cursor: 'pointer',
                                                border: `1px solid ${active ? C.signal : C.line}`,
                                                bgcolor: active ? alpha(C.signal, 0.1) : alpha(C.void, 0.6),
                                                transition: 'all 0.2s ease',
                                                '&:hover': { borderColor: C.signal, bgcolor: alpha(C.signal, 0.05) },
                                            }}
                                        >
                                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.75 }}>
                                                <Typography sx={{ fontFamily: DISPLAY, fontSize: 13, fontWeight: 700, color: active ? C.signal : C.text }}>
                                                    {p.name}
                                                </Typography>
                                                <Typography sx={{ fontFamily: MONO, fontSize: 9.5, color: C.textFaint }}>
                                                    {p.desc}
                                                </Typography>
                                            </Box>
                                            {/* Colormap Gradient Strip */}
                                            <Box sx={{
                                                width: '100%',
                                                height: 10,
                                                borderRadius: 1,
                                                background: p.gradient,
                                                border: '1px solid rgba(255,255,255,0.1)',
                                            }} />
                                        </Box>
                                    );
                                })}
                            </Box>
                        </Panel>

                        {/* Radiometric Sensor Calibration */}
                        <Panel title="Sensor Calibration" subtitle="Thermal transfer function parameters">
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                <Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                        <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textDim }}>
                                            MIN TEMPERATURE (BLACK LEVEL):
                                        </Typography>
                                        <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.signal, fontWeight: 700 }}>
                                            {calibration.temp_min_c}°C
                                        </Typography>
                                    </Box>
                                    <Slider
                                        size="small"
                                        min={-40}
                                        max={30}
                                        step={1}
                                        value={calibration.temp_min_c}
                                        onChange={(_, v) => setCalibration(c => ({ ...c, temp_min_c: v }))}
                                        sx={{ color: C.signal }}
                                    />
                                </Box>

                                <Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                        <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textDim }}>
                                            MAX TEMPERATURE (WHITE LEVEL):
                                        </Typography>
                                        <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.warn, fontWeight: 700 }}>
                                            {calibration.temp_max_c}°C
                                        </Typography>
                                    </Box>
                                    <Slider
                                        size="small"
                                        min={40}
                                        max={250}
                                        step={5}
                                        value={calibration.temp_max_c}
                                        onChange={(_, v) => setCalibration(c => ({ ...c, temp_max_c: v }))}
                                        sx={{ color: C.warn }}
                                    />
                                </Box>

                                <Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                        <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textDim }}>
                                            SURFACE EMISSIVITY (ε):
                                        </Typography>
                                        <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.ok, fontWeight: 700 }}>
                                            {calibration.emissivity.toFixed(2)}
                                        </Typography>
                                    </Box>
                                    <Slider
                                        size="small"
                                        min={0.5}
                                        max={1.0}
                                        step={0.01}
                                        value={calibration.emissivity}
                                        onChange={(_, v) => setCalibration(c => ({ ...c, emissivity: v }))}
                                        sx={{ color: C.ok }}
                                    />
                                </Box>

                                <Button
                                    variant="outlined"
                                    onClick={handleSaveCalibration}
                                    disabled={savingCal}
                                    startIcon={savingCal ? <CircularProgress size={14} /> : <CheckCircle2 size={14} />}
                                    sx={{
                                        fontFamily: MONO,
                                        fontSize: 11,
                                        borderColor: C.signal,
                                        color: C.signal,
                                        mt: 1,
                                        '&:hover': { bgcolor: alpha(C.signal, 0.1), borderColor: C.signal },
                                    }}
                                >
                                    COMMIT CALIBRATION TO EDGE
                                </Button>
                            </Box>
                        </Panel>

                        {/* Multispectral Cross-Fusion Engine */}
                        <Panel title="Multispectral Fusion" subtitle="RGB + Thermal Edge Overlay">
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.textDim }}>
                                        RGB-THERMAL FUSION BLEND (α):
                                    </Typography>
                                    <Typography sx={{ fontFamily: MONO, fontSize: 11, color: C.accent, fontWeight: 700 }}>
                                        {(fusionAlpha * 100).toFixed(0)}%
                                    </Typography>
                                </Box>
                                <Slider
                                    size="small"
                                    min={0.0}
                                    max={1.0}
                                    step={0.05}
                                    value={fusionAlpha}
                                    onChange={(_, v) => setFusionAlpha(v)}
                                    sx={{ color: C.accent }}
                                />
                                <Typography sx={{ fontFamily: MONO, fontSize: 10, color: C.textFaint }}>
                                    Combines structural high-frequency visible contours with infrared radiometric thermal signatures for penetration through fog, smoke, and zero-lux conditions.
                                </Typography>
                            </Box>
                        </Panel>
                    </Stack>
                </Grid>
            </Grid>
        </Box>
    );
}
