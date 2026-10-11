/**
 * Tactical Mock Telemetry & Autonomous Demo Simulation Engine
 *
 * Provides high-fidelity operational data when deployed statically on GitHub Pages
 * or when the Python API server is offline. Designed to mirror Palantir Gotham &
 * Anduril Lattice telemetry formats.
 */

// Core seed cameras fleet
const MOCK_CAMERAS = [
    {
        id: 'cam-1',
        name: 'North Gate Perimeter (Thermal/RGB)',
        rtsp_url: 'rtsp://edge-01.argus.internal/stream1',
        status: 'online',
        fps: 29.8,
        resolution: '1920x1080',
        zones_count: 2,
        location_tag: 'SECTOR-01 // NORTH VAULT',
    },
    {
        id: 'cam-2',
        name: 'Highway Corridor Alpha (ANPR High-Speed)',
        rtsp_url: 'rtsp://edge-02.argus.internal/stream1',
        status: 'online',
        fps: 30.0,
        resolution: '1920x1080',
        zones_count: 3,
        location_tag: 'CORRIDOR-04 // HIGHWAY SPEEDWAY',
    },
    {
        id: 'cam-3',
        name: 'Downtown Intersection #4 (CityOS Twin)',
        rtsp_url: 'rtsp://edge-03.argus.internal/stream1',
        status: 'online',
        fps: 28.5,
        resolution: '1920x1080',
        zones_count: 4,
        location_tag: 'GRID-09 // BROADWAY CROSSING',
    },
    {
        id: 'cam-4',
        name: 'South Depot Gate 2 (Biometric Vault)',
        rtsp_url: 'rtsp://edge-04.argus.internal/stream1',
        status: 'online',
        fps: 29.4,
        resolution: '1920x1080',
        zones_count: 1,
        location_tag: 'VAULT-02 // SECURE ACCESS',
    },
];

// Dual-access wrapper: allows both `Array.isArray(x)` and `x.cameras` or `x.events`
function wrapList(arr, key) {
    const copy = [...arr];
    copy[key] = arr;
    copy.total = arr.length;
    return copy;
}

export const getDemoFallbackData = (url = '') => {
    const u = String(url);

    // Cameras fleet
    if (u.includes('/cameras')) {
        return wrapList(MOCK_CAMERAS, 'cameras');
    }

    // Zones
    if (u.includes('/zones')) {
        const mockZones = [
            { id: 'z-1', name: 'Perimeter Exclusion Zone Alpha', camera_id: 'cam-1', polygon: [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]], color: '#f43f5e' },
            { id: 'z-2', name: 'Speed Vector Gate A', camera_id: 'cam-2', polygon: [[0.2, 0.3], [0.7, 0.3], [0.7, 0.8], [0.2, 0.8]], color: '#00f2fe' },
            { id: 'z-3', name: 'Pedestrian Conflict Box', camera_id: 'cam-3', polygon: [[0.3, 0.3], [0.6, 0.3], [0.6, 0.7], [0.3, 0.7]], color: '#fbbf24' },
        ];
        return wrapList(mockZones, 'zones');
    }

    // Events & Incident stats
    if (u.includes('/events/stats')) {
        return {
            total: 342,
            open: 12,
            acknowledged: 4,
            resolved: 326,
            critical: 3,
            by_priority: { low: 120, medium: 145, high: 58, critical: 19 },
            by_rule: {
                loitering: 48,
                line_crossing: 85,
                zone_intrusion: 112,
                speed_violation: 64,
                thermal_anomaly: 33,
            },
        };
    }

    if (u.includes('/events')) {
        const mockEvents = [
            {
                id: 101,
                camera_id: 'cam-1',
                rule_type: 'zone_intrusion',
                priority: 'critical',
                status: 'open',
                confidence: 0.96,
                timestamp: new Date(Date.now() - 35000).toISOString(),
                object_type: 'person',
                metadata: {
                    summary: 'Restricted perimeter breach detected via YOLOv8 pose detector',
                    evidence: ['Distance: 14m', 'Confidence: 96%', 'Dwell: 38s'],
                },
            },
            {
                id: 102,
                camera_id: 'cam-2',
                rule_type: 'speed_violation',
                priority: 'high',
                status: 'acknowledged',
                confidence: 0.94,
                timestamp: new Date(Date.now() - 95000).toISOString(),
                object_type: 'car',
                metadata: {
                    summary: 'Vehicle exceeding 80 km/h in urban corridor (Speed: 82 km/h)',
                    evidence: ['Plate: MH-12-DE-1944', 'Vector: +18 km/h over limit'],
                },
            },
            {
                id: 103,
                camera_id: 'cam-3',
                rule_type: 'red_light_running',
                priority: 'medium',
                status: 'resolved',
                confidence: 0.98,
                timestamp: new Date(Date.now() - 210000).toISOString(),
                object_type: 'car',
                metadata: {
                    summary: 'Vehicle crossed stop bar during red phase (PET: 0.8s)',
                    evidence: ['Conflict PET: 0.8s', 'Approach: Northbound'],
                },
            },
            {
                id: 104,
                camera_id: 'cam-4',
                rule_type: 'watchlist_face_match',
                priority: 'critical',
                status: 'open',
                confidence: 0.95,
                timestamp: new Date(Date.now() - 400000).toISOString(),
                object_type: 'person',
                metadata: {
                    summary: 'Subject Alexander Vance identified at depot gate (Cosine: 0.95)',
                    evidence: ['512D ArcFace match', 'Classification: CRITICAL'],
                },
            },
        ];
        const res = wrapList(mockEvents, 'events');
        res.event = mockEvents[0];
        return res;
    }

    // Thermal Vision & Radiometrics
    if (u.includes('/thermal/palettes')) {
        return { palettes: ['ironbow', 'inferno', 'jet', 'rainbow', 'white_hot', 'black_hot', 'bone', 'hot'] };
    }
    if (u.includes('/thermal/calibration')) {
        return { temp_min_c: -10.0, temp_max_c: 100.0, emissivity: 0.95, default_palette: 'ironbow' };
    }
    if (u.includes('/thermal/analyze')) {
        return {
            temperature_stats: { min_temp_c: 18.2, max_temp_c: 41.6, mean_temp_c: 24.5 },
            hotspot_count: 2,
            signature_count: 4,
            hotspots: [
                { bbox: [120, 80, 60, 60], max_temperature_c: 38.6, type: 'fever_screening', severity: 'warning' },
                { bbox: [260, 160, 90, 50], max_temperature_c: 72.4, type: 'fire_or_overheating', severity: 'critical' },
            ],
        };
    }

    // ANPR & Vehicles
    if (u.includes('/lpr')) {
        const mockPlates = [
            { id: 1, license_plate: 'KA-03-HA-8821', plate: 'KA-03-HA-8821', vehicle_type: 'SUV (White)', vehicle: 'SUV (White)', camera_id: 'cam-1', camera: 'North Gate Perimeter', speed_kmh: 48, speed: 48, confidence: 0.98, is_flagged: false, status: 'NOMINAL', timestamp: new Date(Date.now() - 45000).toISOString() },
            { id: 2, license_plate: 'MH-12-DE-1944', plate: 'MH-12-DE-1944', vehicle_type: 'Sedan (Black)', vehicle: 'Sedan (Black)', camera_id: 'cam-2', camera: 'Highway Corridor Alpha', speed_kmh: 82, speed: 82, confidence: 0.95, is_flagged: true, status: 'HOTLIST_BOLO', timestamp: new Date(Date.now() - 120000).toISOString() },
            { id: 3, license_plate: 'DL-01-AB-3310', plate: 'DL-01-AB-3310', vehicle_type: 'Truck (Heavy)', vehicle: 'Truck (Heavy)', camera_id: 'cam-4', camera: 'South Depot Gate 2', speed_kmh: 28, speed: 28, confidence: 0.93, is_flagged: true, status: 'EXPIRED_REG', timestamp: new Date(Date.now() - 240000).toISOString() },
            { id: 4, license_plate: 'KA-05-MQ-9002', plate: 'KA-05-MQ-9002', vehicle_type: 'Motorcycle', vehicle: 'Motorcycle', camera_id: 'cam-3', camera: 'Downtown Intersection #4', speed_kmh: 55, speed: 55, confidence: 0.89, is_flagged: false, status: 'NOMINAL', timestamp: new Date(Date.now() - 360000).toISOString() },
        ];
        return wrapList(mockPlates, 'plates');
    }

    // Faces & Biometrics
    if (u.includes('/faces/status')) {
        return { total_faces: 3, database_encrypted: true, cipher: 'AES-256-GCM' };
    }
    if (u.includes('/faces')) {
        const mockFaces = [
            { id: 'f-1', name: 'Alexander Vance', role: 'PERSON OF INTEREST', classification: 'CRITICAL', confidence: 0.96, distance: '14.2m', camera: 'Perimeter Gate East', match_vector_id: 'vec_7701a', status: 'CRITICAL', created_at: '2026-09-14', timestamp: new Date(Date.now() - 35000).toISOString() },
            { id: 'f-2', name: 'Dr. Elena Rostova', role: 'AUTHORIZED VIP', classification: 'VIP', confidence: 0.99, distance: '6.5m', camera: 'Terminal Main Lobby', match_vector_id: 'vec_1098b', status: 'AUTHORIZED', created_at: '2026-08-20', timestamp: new Date(Date.now() - 95000).toISOString() },
            { id: 'f-3', name: 'Marcus Sterling', role: 'WATCHLIST ESCORT', classification: 'BOLO', confidence: 0.91, distance: '22.8m', camera: 'Transit Hub Camera 3', match_vector_id: 'vec_3319c', status: 'MONITOR', created_at: '2026-10-01', timestamp: new Date(Date.now() - 210000).toISOString() },
        ];
        return wrapList(mockFaces, 'faces');
    }

    // Poses & Movement
    if (u.includes('/poses')) {
        const mockPoses = [
            { id: 1, track_id: 'TRK-4401', height_cm: 182, posture: 'WALKING', velocity_mps: 1.4, dwell_time_seconds: 45, is_fallen: false, camera_id: 'cam-1', timestamp: new Date().toISOString() },
            { id: 2, track_id: 'TRK-4402', height_cm: 168, posture: 'RUNNING', velocity_mps: 4.8, dwell_time_seconds: 12, is_fallen: false, camera_id: 'cam-2', timestamp: new Date().toISOString() },
            { id: 3, track_id: 'TRK-4403', height_cm: 175, posture: 'LOITERING', velocity_mps: 0.2, dwell_time_seconds: 290, is_fallen: false, camera_id: 'cam-3', timestamp: new Date().toISOString() },
            { id: 4, track_id: 'TRK-4404', height_cm: 172, posture: 'PRONE', velocity_mps: 0.0, dwell_time_seconds: 80, is_fallen: true, camera_id: 'cam-4', timestamp: new Date().toISOString() },
        ];
        return wrapList(mockPoses, 'poses');
    }

    // System Health & Hardware Telemetry
    if (u.includes('/health')) {
        return {
            status: 'healthy',
            version: '2.4.0',
            engine: 'ONNX-YOLOv8 & ArcFace 512D',
            mode: 'Autonomous Standalone Deck (Zero-Backend Mode)',
            crypto_verified: true,
            uptime_seconds: 86420,
        };
    }
    if (u.includes('/metrics')) {
        return {
            system: { cpu_percent: 24.8, memory_percent: 38.2, disk_percent: 45.1 },
            cameras: MOCK_CAMERAS.map((c) => ({
                id: c.id,
                name: c.name,
                status: c.status,
                fps: c.fps,
                queue_depth: 2,
            })),
            cpu_percent: 24.8,
            memory_percent: 38.2,
            fps: 54.0,
            latency_ms: 18.4,
        };
    }
    if (u.includes('/audit/integrity') || u.includes('/audit')) {
        return {
            intact: true,
            block_count: 1420,
            latest_hash: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
            merkle_root: '9f833776d648679f2204eb8b9a2dc1e149afbf4c8996fb92427ae41e4649b934',
        };
    }

    // Rules & Capabilities
    if (u.includes('/rules/status') || u.includes('/rules')) {
        return {
            rules: {
                zone_intrusion: { can_fire: true },
                line_crossing: { can_fire: true },
                loitering: { can_fire: true },
                speed_violation: { can_fire: true },
                thermal_anomaly: { can_fire: true },
            },
            summary: { total: 5, active: 5, blocked: [] },
        };
    }

    // Alert / Notification Delivery
    if (u.includes('/notifications/status') || u.includes('/notifications')) {
        return {
            channels: [
                { id: 'webhook', name: 'Tactical Edge Webhook (Lattice)', enabled: true },
                { id: 'smtp', name: 'Encrypted Security SMTP', enabled: true },
            ],
            any_channel_available: true,
        };
    }

    // Evidence status
    if (u.includes('/evidence/status') || u.includes('/evidence')) {
        return {
            total_clips: 142,
            ledger_intact: true,
            merkle_root: '7f91a20c3d5e8b41e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934',
        };
    }

    // Observation Promotion
    if (u.includes('/observations/promotion')) {
        return { ready: true, status: 'operational', promoted_models: 3 };
    }

    // CityOS Intersection Digital Twin
    if (u.includes('/cityos/status')) {
        return { mode: 'adaptive', active_intersections: 8, efficiency: 0.94, green_wave_active: true };
    }
    if (u.includes('/cityos/twin')) {
        return {
            intersections: [
                { id: 'int-1', name: 'Downtown 4th & Main', phase: 'NORTH_SOUTH_GREEN', cycle_length: 90, elapsed: 42, queue_length: 4 },
                { id: 'int-2', name: 'Broadway & 7th Ave', phase: 'EAST_WEST_GREEN', cycle_length: 75, elapsed: 18, queue_length: 2 },
            ],
            objects: [
                { track_id: 101, category: 'vehicle', x: 280, y: 320, speed_mps: 12.5, trajectory: [[280, 360], [280, 320]] },
                { track_id: 102, category: 'pedestrian', x: 190, y: 180, speed_mps: 1.2, trajectory: [[180, 180], [190, 180]] },
                { track_id: 103, category: 'cyclist', x: 310, y: 240, speed_mps: 4.5, trajectory: [[310, 260], [310, 240]] },
                { track_id: 104, category: 'truck', x: 420, y: 450, speed_mps: 8.0, trajectory: [[420, 500], [420, 450]] },
            ],
            signal: { mode: 'adaptive', state: 'green', phase: 'NS', current_green_s: 28, max_green_s: 45, countdown_s: 17 },
            signal_recommendation: { action: 'extend_green', suggested_green_s: 35, reason: 'High North-South queue volume detected' },
            counts_by_category: { vehicle: 14, pedestrian: 6, cyclist: 2, bus: 1, truck: 2 },
            safety: { near_miss_count: 1, pet_histogram: [0, 1, 4, 12, 8] },
            edge_node: { cpu_percent: 28.4, mem_percent: 42.1, status: 'operational' },
            flow: {
                volume_series: [
                    { minute: Math.floor(Date.now() / 1000) - 240, volume: 45 },
                    { minute: Math.floor(Date.now() / 1000) - 180, volume: 52 },
                    { minute: Math.floor(Date.now() / 1000) - 120, volume: 48 },
                    { minute: Math.floor(Date.now() / 1000) - 60, volume: 61 },
                ],
            },
        };
    }
    if (u.includes('/cityos/alerts')) {
        const mockAlerts = [
            { id: 1, type: 'PET_CONFLICT', kind: 'near_miss', severity: 'HIGH', pet_seconds: 1.2, location: 'Downtown 4th & Main', timestamp: new Date(Date.now() - 40000).toISOString() },
            { id: 2, type: 'RED_LIGHT_VIOLATION', kind: 'wrong_way', severity: 'CRITICAL', speed_kmh: 68, location: 'Broadway & 7th Ave', timestamp: new Date(Date.now() - 180000).toISOString() },
        ];
        return wrapList(mockAlerts, 'alerts');
    }

    // Cross-Camera Tracker
    if (u.includes('/cross-camera/tracks') || u.includes('/cross-camera')) {
        return {
            active_tracks: 14,
            matching_stats: { match_success_rate: 94.2, total_matches: 128, successful_matches: 121 },
            tracks: [
                { person_id: 'P-902', current_camera: 'cam-1', last_seen: new Date().toISOString(), confidence: 0.96 },
                { person_id: 'P-903', current_camera: 'cam-2', last_seen: new Date().toISOString(), confidence: 0.91 },
            ],
        };
    }

    // Adaptive Learning
    if (u.includes('/stats/learning')) {
        return {
            total_behavior_profiles: 142,
            total_emotion_baselines: 89,
            sklearn_available: true,
            generation: 42,
            best_fitness: 0.982,
            mutations_evaluated: 1240,
            status: 'converged',
        };
    }

    // Neural Memory
    if (u.includes('/memory/stats')) {
        return {
            observations: 12450,
            people_tracked: 420,
            by_kind: { dwell: 45, pacing: 18, abandoned_object: 12, occupancy_anomaly: 8, disappeared: 4, scene_change: 6 },
        };
    }
    if (u.includes('/memory/recall')) {
        return {
            results: [
                { id: 1, camera_id: 'cam-1', kind: 'dwell', summary: 'Subject loitered near perimeter fence for 290s', confidence: 0.94, evidence: ['Dwell time: 290s', 'Threshold: 120s', 'Bounding box stationary: True'] },
                { id: 2, camera_id: 'cam-2', kind: 'speed_violation', summary: 'Vehicle exceeded corridor limit by 22 km/h', confidence: 0.96, evidence: ['Recorded speed: 82 km/h', 'Corridor speed limit: 60 km/h'] },
            ],
            by_kind: { dwell: 1, speed_violation: 1 },
        };
    }

    // Clusters
    if (u.includes('/clusters')) {
        return { clusters: [{ id: 1, label: 'Vehicle Clusters', size: 120 }, { id: 2, label: 'Pedestrian Clusters', size: 450 }] };
    }

    return {};
};
