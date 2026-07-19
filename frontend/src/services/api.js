import axios from 'axios';

const API_BASE_URL = '/api/v1';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

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
