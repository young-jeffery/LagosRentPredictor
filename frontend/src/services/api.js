import axios from 'axios';

// Initialize the Axios client pointing to the local FastAPI server
const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
    headers: {
        'Content-Type': 'application/json',
    },
});

// Function to send the prediction payload to the backend
export const estimateRentPrice = async (payload) => {
    try {
        const response = await apiClient.post('/predict', payload);
        // Return the successfully resolved data payload
        return response.data;
    } catch (error) {
        // Extract the error message from FastAPI if it exists, otherwise provide a generic fallback
        const errorMessage = error.response?.data?.detail || 'Failed to connect to the prediction server. Please try again.';
        throw new Error(errorMessage, { cause: error });
    }
};

// [ADDED] Budget-based recommender
export const getRecommendations = async (payload) => {
    try {
        const response = await apiClient.post('/recommend', payload);
        return response.data;
    } catch (error) {
        const errorMessage = error.response?.data?.detail || 'Failed to connect to the recommendation server. Please try again.';
        throw new Error(errorMessage, { cause: error });
    }
};