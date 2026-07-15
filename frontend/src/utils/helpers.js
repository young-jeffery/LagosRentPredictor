// Format numbers securely into Nigerian Naira (₦)
export const formatNaira = (amount) => {
    if (amount === undefined || amount === null) return '₦0';
    
    return new Intl.NumberFormat('en-NG', {
        style: 'currency',
        currency: 'NGN',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(amount);
};

// --- LocalStorage History Management ---
const HISTORY_KEY = 'lagos_rent_history';

// Retrieve the saved estimation history
export const getHistory = () => {
    const data = localStorage.getItem(HISTORY_KEY);
    return data ? JSON.parse(data) : [];
};

// Save a new estimation to history, enforcing a maximum of 30 records
export const saveToHistory = (newEntry) => {
    const currentHistory = getHistory();
    
    // Unshift the new entry to the top of the array, and slice to keep only the latest 30
    const updatedHistory = [newEntry, ...currentHistory].slice(0, 30);
    
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updatedHistory));
    return updatedHistory;
};