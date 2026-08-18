import axios from 'axios';

const API_BASE_URL = '/api/v1';

const ACCESS_KEY = 'argus_access_token';
const REFRESH_KEY = 'argus_refresh_token';
const USER_KEY = 'argus_user';

// ── Token storage ────────────────────────────────────────────────────────────
// sessionStorage, not localStorage: tokens die with the tab rather than lingering
// on a shared workstation. Control-room machines are rarely single-user.
export const tokenStore = {
    getAccess: () => sessionStorage.getItem(ACCESS_KEY),
    getRefresh: () => sessionStorage.getItem(REFRESH_KEY),
    getUser: () => {
        const raw = sessionStorage.getItem(USER_KEY);
        try {
            return raw ? JSON.parse(raw) : null;
        } catch {
            return null;
        }
    },
    set: (access, refresh, user) => {
        if (access) sessionStorage.setItem(ACCESS_KEY, access);
        if (refresh) sessionStorage.setItem(REFRESH_KEY, refresh);
        if (user) sessionStorage.setItem(USER_KEY, JSON.stringify(user));
    },
    clear: () => {
        sessionStorage.removeItem(ACCESS_KEY);
        sessionStorage.removeItem(REFRESH_KEY);
        sessionStorage.removeItem(USER_KEY);
    },
};

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Attach the bearer token to every outgoing request.
api.interceptors.request.use((config) => {
    const token = tokenStore.getAccess();
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// ── Transparent refresh on 401 ───────────────────────────────────────────────
// Access tokens live 30 minutes. Without this an operator gets silently logged
// out mid-shift; with it the refresh token (7 days) renews in the background and
// the failed request is replayed. Concurrent 401s share one refresh call so a
// dashboard polling several endpoints does not stampede the auth endpoint.
let refreshInFlight = null;

const notifyAuthFailure = () => {
    tokenStore.clear();
    window.dispatchEvent(new Event('argus:unauthenticated'));
};

api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const original = error.config;
        const status = error.response?.status;

        if (status !== 401 || !original || original._retried) {
            return Promise.reject(error);
        }
        // Never try to refresh the refresh call itself.
        if (original.url && original.url.includes('/auth/')) {
            notifyAuthFailure();
            return Promise.reject(error);
        }

        const refreshToken = tokenStore.getRefresh();
        if (!refreshToken) {
            notifyAuthFailure();
            return Promise.reject(error);
        }

        original._retried = true;
        try {
            if (!refreshInFlight) {
                refreshInFlight = axios
                    .post(`${API_BASE_URL}/auth/refresh`, { refresh_token: refreshToken })
                    .finally(() => { refreshInFlight = null; });
            }
            const { data } = await refreshInFlight;
            tokenStore.set(data.access_token, data.refresh_token || refreshToken, data.user);
            original.headers = original.headers || {};
            original.headers.Authorization = `Bearer ${data.access_token}`;
            return api(original);
        } catch (refreshError) {
            notifyAuthFailure();
            return Promise.reject(refreshError);
        }
    }
);

// ── Auth API ─────────────────────────────────────────────────────────────────
export const authAPI = {
    login: async (username, password) => {
        const { data } = await axios.post(`${API_BASE_URL}/auth/login`, { username, password });
        tokenStore.set(data.access_token, data.refresh_token, data.user);
        return data;
    },
    logout: () => {
        tokenStore.clear();
        window.dispatchEvent(new Event('argus:unauthenticated'));
    },
    me: () => api.get('/auth/me'),
    currentUser: () => tokenStore.getUser(),
    isAuthenticated: () => Boolean(tokenStore.getAccess()),
};

// Build a WebSocket URL carrying the access token.
// Browsers cannot set headers on a WebSocket handshake, so the token travels as
// a query parameter - which is why the backend authenticates before accept().
export const buildStreamUrl = (cameraId) => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = tokenStore.getAccess();
    const query = token ? `?token=${encodeURIComponent(token)}` : '';
    return `${proto}//${window.location.host}/api/ws/stream/${cameraId}${query}`;
};

// Camera API
export const cameraAPI = {
    getAll: () => api.get('/cameras'),
    getById: (id) => api.get(`/cameras/${id}`),
    create: (data) => api.post('/cameras', data),
    update: (id, data) => api.put(`/cameras/${id}`, data),
    delete: (id) => api.delete(`/cameras/${id}`),
};

// Zone API
export const zoneAPI = {
    getAll: (cameraId) => api.get('/zones', { params: { camera_id: cameraId } }),
    create: (data) => api.post('/zones', data),
    update: (id, data) => api.put(`/zones/${id}`, data),
    delete: (id) => api.delete(`/zones/${id}`),
};

// Event API
export const eventAPI = {
    getAll: (params) => api.get('/events', { params }),
    getById: (id) => api.get(`/events/${id}`),
    getStats: (params) => api.get('/events/stats', { params }),
    // Lifecycle: detected -> open -> acknowledged -> resolved.
    updateStatus: (id, status) => api.patch(`/events/${id}/status`, { status }),
    lifecycle: () => api.get('/events/lifecycle'),
};

// Perception memory: durable observations and appearance search.
export const memoryAPI = {
    recall: (params) => api.get('/memory/recall', { params }),
    summary: (params) => api.get('/memory/summary', { params }),
    similar: (cameraId, trackId, limit) =>
        api.get(`/memory/appearances/${cameraId}/${trackId}/similar`, { params: { limit } }),
    stats: () => api.get('/memory/stats'),
};

// What Argus can actually do here, as opposed to what config claims.
export const capabilityAPI = {
    rules: () => api.get('/rules/status'),
    calibration: () => api.get('/rules/calibration'),
    promotion: () => api.get('/observations/promotion'),
};

// Alert delivery. Argus recorded events and delivered them nowhere for its
// entire history, so the UI must be able to show whether any channel can
// actually send right now - availability is probed, not read from config.
export const alertAPI = {
    status: () => api.get('/notifications/status'),
    test: () => api.post('/notifications/test'),
};

// Snapshots. These are event evidence containing identifiable people, so the
// route is role-protected: GET /api/snapshots/{camera_id}/{filename}. Note it
// sits under /api, NOT /api/v1, so it cannot use the shared axios baseURL.
export const snapshotAPI = {
    fetch: async (cameraId, filename) => {
        const res = await api.get(
            `/snapshots/${cameraId}/${encodeURIComponent(filename)}`,
            { baseURL: '/api', responseType: 'blob' },
        );
        return URL.createObjectURL(res.data);
    },
};

// Pre-event video evidence.
export const evidenceAPI = {
    status: () => api.get('/evidence/status'),
    // The clip route requires a bearer token, and a <video src> attribute
    // cannot carry an Authorization header. Fetch it through axios (which the
    // interceptor authenticates and refreshes) and hand back an object URL.
    // The caller owns the URL and must revoke it.
    fetchClip: async (eventId) => {
        const res = await api.get(`/events/${eventId}/clip`, { responseType: 'blob' });
        return URL.createObjectURL(res.data);
    },
};

// System API
export const systemAPI = {
    health: () => api.get('/health'),
    metrics: () => api.get('/metrics'),
};

// Cross-Camera Tracker API
export const crossCameraAPI = {
    getTracks: () => api.get('/cross-camera/tracks'),
    getTargets: () => api.get('/cross-camera/targets'),
    setTarget: (person_id, camera_id, reason) => {
        const formData = new FormData();
        formData.append('person_id', person_id);
        formData.append('camera_id', camera_id);
        formData.append('reason', reason || '');
        return api.post('/cross-camera/target', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    },
    deleteTarget: (person_id) => api.delete(`/cross-camera/target/${person_id}`),
    getPath: (person_id) => api.get(`/cross-camera/path/${person_id}`),
    predictTrajectory: (person_id, horizon_seconds) => api.get(`/cross-camera/predict/${person_id}`, { params: { horizon_seconds } }),
    getGraph: () => api.get('/cross-camera/graph'),
    setGraph: (graph) => api.post('/cross-camera/graph', graph),
    clearOldTracks: (max_age_hours) => api.post('/cross-camera/clear-old', { max_age_hours }),
};

// Learning stats API
export const learningAPI = {
    getStats: () => api.get('/stats/learning'),
};

// Video Testing API
export const videoAPI = {
    processVideo: (video_path, camera_ids, duration_seconds) => {
        const formData = new FormData();
        formData.append('video_path', video_path);
        formData.append('camera_ids', camera_ids);
        formData.append('duration_seconds', duration_seconds);
        return api.post('/video/process', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    },
};

// Clusters API
export const clusterAPI = {
    getClusters: () => api.get('/clusters'),
};

export default api;
