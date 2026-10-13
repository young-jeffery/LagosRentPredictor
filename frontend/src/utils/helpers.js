// src/utils/helpers.js

export const formatNaira = (amount) => {
    if (amount === undefined || amount === null) return '₦0';
    return new Intl.NumberFormat('en-NG', {
        style: 'currency',
        currency: 'NGN',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(amount);
};

const HISTORY_KEY = 'lagos_rent_history';

export const getHistory = () => {
    const data = localStorage.getItem(HISTORY_KEY);
    return data ? JSON.parse(data) : [];
};

export const saveToHistory = (newEntry) => {
    const currentHistory = getHistory();
    const updatedHistory = [newEntry, ...currentHistory].slice(0, 30);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updatedHistory));
    return updatedHistory;
};

// --- NEW MATH LOGIC ---

// Determine the rounding step dynamically based on the value size
const getRoundingStep = (value) => {
    if (value >= 5000000) return 100000; // Nearest 100k for 5M+
    if (value >= 1000000) return 50000;  // Nearest 50k for 1M to 5M
    return 10000;                        // Nearest 10k for under 1M
};

// Round negotiation limits (Lower to floor, Upper to ceiling)
export const getApproximatedRanges = (lower, upper) => {
    const lowerStep = getRoundingStep(lower);
    const upperStep = getRoundingStep(upper);
    
    return {
        approxLower: Math.floor(lower / lowerStep) * lowerStep,
        approxUpper: Math.ceil(upper / upperStep) * upperStep
    };
};

// Round the single point estimate using the same dynamic step logic
export const getApproximatedTct = (value) => {
    const step = getRoundingStep(value);
    return Math.round(value / step) * step;
};

// Round a single breakdown line item for display (same step logic as TCT/ranges)
export const getApproximatedBreakdownValue = (value) => {
    const step = getRoundingStep(value);
    return Math.round(value / step) * step;
};

// Calculate proportional sub-ranges for the breakdown items
export const getBreakdownRange = (itemValue, originalTct, approxLower, approxUpper) => {
    const weight = itemValue / originalTct;
    
    // We round to the nearest 1,000 so the sub-breakdown looks clean
    return {
        itemLower: Math.round((approxLower * weight) / 1000) * 1000,
        itemUpper: Math.round((approxUpper * weight) / 1000) * 1000
    };
};