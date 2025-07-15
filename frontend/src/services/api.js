import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health and system endpoints
  getHealth: () => api.get('/health'),
  getMetrics: () => api.get('/metrics'),
  getModels: () => api.get('/models'),

  // Transaction endpoints
  predictTransaction: (transaction) => api.post('/predict', transaction),
  predictBatch: (transactions) => api.post('/predict/batch', transactions),
  getRecentTransactions: (limit = 100) => api.get(`/transactions/recent?limit=${limit}`),

  // Analytics endpoints
  getHourlyAnalytics: () => api.get('/analytics/hourly'),
  getDailyAnalytics: () => api.get('/analytics/daily'),

  // Alerts and monitoring
  getAlerts: () => api.get('/alerts'),

  // Model management
  retrainModels: () => api.post('/models/retrain'),
};

export { api }; 