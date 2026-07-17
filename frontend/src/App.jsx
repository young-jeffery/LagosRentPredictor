import { useState, useEffect } from 'react';
import Navbar from './components/layout/Navbar';
import EstimatorForm from './components/EstimatorForm';
import ResultsDisplay from './components/ResultsDisplay';
import HistoryTable from './components/HistoryTable';
import Footer from './components/layout/Footer';
import { estimateRentPrice } from './services/api';
import { getHistory, saveToHistory } from './utils/helpers';
import { FEATURES_LIST } from './constants/appConstants';

const INITIAL_FORM_STATE = {
  Consolidated_Location: '', Bedrooms: 1, Bathrooms: 1, Is_Serviced: 0, 
  Is_Furnished: 0, has_prepaid_meter: 0, has_screeded_walls: 0, 
  has_fitted_kitchen: 0, has_clean_water: 0, has_steady_power: 0, has_ac: 0, 
  has_water_heater: 0, has_internet: 0, is_ensuite: 0, has_bq: 0, has_pop_ceiling: 0,
  has_wardrobe: 0, has_balcony: 0, is_secured: 0, in_gated_estate: 0, has_cctv: 0, 
  has_pool: 0, has_gym: 0, has_smart_home: 0, has_elevator: 0, has_parking: 0, 
  is_tarred: 0, has_easy_access: 0, prefers_working_class: 0
};

export default function App() {
  const [formData, setFormData] = useState(INITIAL_FORM_STATE);
  const [history, setHistory] = useState((() => getHistory())());
  const [activeResult, setActiveResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  const handleEstimateRequest = async (e) => {
    e.preventDefault();
    if (!formData.Consolidated_Location) { setApiError("Please select a location before estimating."); return; }
    
    setIsLoading(true); setApiError(null); setActiveResult(null);

    const isDuplicate = history.some(item => {
      if (!item.fullFormData) return false;
      return Object.keys(formData).every(key => item.fullFormData[key] === formData[key]);
    });

    try {
      const [result] = await Promise.all([
        estimateRentPrice(formData),
        new Promise(resolve => setTimeout(resolve, 2000))
      ]);
      
      setActiveResult(result);
      
      if (isDuplicate) {
        setTimeout(() => alert("This exact property combination has already been estimated and is in your history."), 100); 
      } else {
        const historyEntry = {
          id: crypto.randomUUID(), 
          location: formData.Consolidated_Location, 
          beds: formData.Bedrooms, baths: formData.Bathrooms, 
          estimated_tct: result.estimated_tct, 
          timestamp: new Date().toISOString(),
          fullFormData: formData, fullResult: result
        };
        setHistory(saveToHistory(historyEntry));
      }
    } catch (error) { setApiError(error.message); } finally { setIsLoading(false); }
  };

  const handleReset = () => {
    if (window.confirm("Are you sure you want to clear all inputs and history?")) {
      localStorage.removeItem('lagos_rent_history');
      setHistory([]); setActiveResult(null); setFormData(INITIAL_FORM_STATE);
    }
  };

  const handleDeleteHistory = (idToDelete) => {
    if(window.confirm("Delete this entry?")) {
      const updatedHistory = history.filter(item => item.id !== idToDelete);
      setHistory(updatedHistory);
      localStorage.setItem('lagos_rent_history', JSON.stringify(updatedHistory));
      if(activeResult && updatedHistory.length === 0) { setActiveResult(null); setFormData(INITIAL_FORM_STATE); }
    }
  };

  const handleHistorySelect = (item) => {
    if (!item.fullFormData) { alert("Old record cannot be restored."); return; }
    setFormData(item.fullFormData); setActiveResult(item.fullResult);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleClearSelections = () => {
    const clearedFeatures = {};
    FEATURES_LIST.forEach(feat => { clearedFeatures[feat.id] = 0; });
    setFormData(prev => ({ ...prev, ...clearedFeatures }));
    setActiveResult(null);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar isDarkMode={isDarkMode} toggleTheme={() => setIsDarkMode(!isDarkMode)} />

      <main className="grow w-full px-4 sm:px-6 md:px-8 py-8 mx-auto flex flex-col">
        <header className="mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 animate-fade-up">
          <div>
            {/* Replaced font-rye with font-spicy */}
            <h1 className="text-4xl font-spicy mb-1 text-brand-dark dark:text-white">Property Rent Estimator</h1>
            <p className="opacity-80 text-lg text-brand-dark dark:text-gray-300">Select your location and property features to get an AI-powered estimate.</p>
          </div>
          <button onClick={handleReset} className="px-4 py-2 border border-brand-accent text-brand-accent hover:bg-brand-accent hover:text-white rounded-md transition-colors font-semibold text-sm shadow-sm">
            Reset Session
          </button>
        </header>

        <div className="rounded-2xl shadow-xl overflow-hidden mb-12 bg-brand-bg/20 border border-brand-gray/20 dark:border-white/10 animate-fade-up delay-100">
          <EstimatorForm formData={formData} setFormData={setFormData} onSubmit={handleEstimateRequest} isLoading={isLoading} apiError={apiError} onClear={handleClearSelections} />
          <ResultsDisplay result={activeResult} formData={formData} />
          <HistoryTable history={history} onSelect={handleHistorySelect} onDelete={handleDeleteHistory} />
        </div>
      </main>

      <Footer />
    </div>
  );
}