// src/App.jsx
import { useState, useEffect } from 'react';
import Navbar from './components/layout/Navbar';
import EstimatorForm from './components/EstimatorForm';
import ResultsDisplay from './components/ResultsDisplay';
import RecommenderForm from './components/RecommenderForm';
import RecommenderResults from './components/RecommenderResults';
import HistoryTable from './components/HistoryTable';
import Footer from './components/layout/Footer';
import ConfirmModal from './components/ui/ConfirmModal';
import { estimateRentPrice, getRecommendations } from './services/api';
import { getHistory, saveToHistory } from './utils/helpers';
import { FEATURES_LIST } from './constants/appConstants';

const INITIAL_FORM_STATE = {
  Official_Sub_Location: '', Main_District: '', Property_Type: 'Flat/Apartment',
  Bedrooms: 1, Bathrooms: 1, Is_Serviced: 0,
  Is_Furnished: 0, has_prepaid_meter: 0, has_screeded_walls: 0,
  has_fitted_kitchen: 0, has_clean_water: 0, has_steady_power: 0, has_ac: 0,
  has_water_heater: 0, has_internet: 0, is_ensuite: 0, has_bq: 0, has_pop_ceiling: 0,
  has_wardrobe: 0, has_balcony: 0, is_secured: 0, in_gated_estate: 0, has_cctv: 0,
  has_pool: 0, has_gym: 0, has_smart_home: 0, has_elevator: 0, has_parking: 0,
  is_tarred: 0, has_easy_access: 0, prefers_working_class: 0,
  has_jacuzzi: 0, is_premium_marketed: 0, is_urgent_listing: 0, is_standalone_fenced: 0
};

// [ADDED] Recommender's own form state -- everything except budget is
// optional/nullable by design.
const INITIAL_RECOMMENDER_STATE = {
  budget: '', Main_District: null, Official_Sub_Location: null, Property_Type: null,
  Bedrooms: null, Bathrooms: null, preferred_features: [], sort_by: 'value',
};

export default function App() {
  // [ADDED] Page-switch state -- "estimator" or "recommender"
  const [activeView, setActiveView] = useState('estimator');

  const [formData, setFormData] = useState(INITIAL_FORM_STATE);
  const [history, setHistory] = useState((() => getHistory())());
  const [activeResult, setActiveResult] = useState(null);
  const [resultFormData, setResultFormData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState(null);

  // [ADDED] Recommender state, separate from the estimator's
  const [recommenderFormData, setRecommenderFormData] = useState(INITIAL_RECOMMENDER_STATE);
  const [recommenderResult, setRecommenderResult] = useState(null);
  const [isRecommending, setIsRecommending] = useState(false);
  const [recommenderError, setRecommenderError] = useState(null);

  const [modal, setModal] = useState({ isOpen: false, type: 'alert', title: '', message: '', onConfirm: null });
  const closeModal = () => setModal(prev => ({ ...prev, isOpen: false }));

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
    if (!formData.Official_Sub_Location) { setApiError("Please select a location before estimating."); return; }

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
      setResultFormData(formData);

      if (isDuplicate) {
        setModal({
          isOpen: true,
          type: 'alert',
          title: 'Duplicate Estimate',
          message: 'This exact property combination has already been estimated and is in your search history. We have loaded the result for you.',
          onConfirm: closeModal
        });
      } else {
        const historyEntry = {
          id: crypto.randomUUID(),
          location: `${formData.Official_Sub_Location}, ${formData.Main_District}`,
          propertyType: formData.Property_Type,
          beds: formData.Bedrooms, baths: formData.Bathrooms,
          estimated_tct: result.estimated_tct,
          timestamp: new Date().toISOString(),
          fullFormData: formData, fullResult: result
        };
        setHistory(saveToHistory(historyEntry));
      }
    } catch (error) { setApiError(error.message); } finally { setIsLoading(false); }
  };

  // [MODIFIED] Extracted the actual API call into its own function so both
  // a fresh search and a sort-change can trigger it without duplicating logic.
  const runRecommendation = async (payload) => {
    setIsRecommending(true); setRecommenderError(null);
    try {
      const [result] = await Promise.all([
        getRecommendations(payload),
        new Promise(resolve => setTimeout(resolve, 1500))
      ]);
      setRecommenderResult(result);
    } catch (error) {
      setRecommenderError(error.message);
    } finally {
      setIsRecommending(false);
    }
  };

  const buildRecommendationPayload = (sortOverride = null) => ({
    budget: recommenderFormData.budget,
    Official_Sub_Location: recommenderFormData.Official_Sub_Location || null,
    Main_District: recommenderFormData.Main_District || null,
    Property_Type: recommenderFormData.Property_Type || null,
    Bedrooms: recommenderFormData.Bedrooms || null,
    Bathrooms: recommenderFormData.Bathrooms || null,
    preferred_features: recommenderFormData.preferred_features,
    sort_by: sortOverride || recommenderFormData.sort_by,
  });

  const handleRecommendRequest = async (e) => {
    e.preventDefault();
    if (!recommenderFormData.budget) { setRecommenderError("Please enter a budget."); return; }
    await runRecommendation(buildRecommendationPayload());
  };

  // [ADDED] Changing sort re-queries automatically, but ONLY if a search has
  // already run (recommenderResult exists). Diversity capping depends on
  // sort order, so this has to be a real re-query, not a client-side
  // re-sort of the already-limited/capped results -- otherwise "Cheapest
  // First" could miss cheaper properties that got excluded from the
  // original top-50 under a different sort.
  const handleSortChange = (newSortValue) => {
    setRecommenderFormData(prev => ({ ...prev, sort_by: newSortValue }));
    if (recommenderResult) {
      runRecommendation(buildRecommendationPayload(newSortValue));
    }
  };

  // [ADDED] Clears the recommender form/results only -- separate from
  // "Reset Session", which wipes the estimator's history too.
  const handleClearRecommender = () => {
    setRecommenderFormData(INITIAL_RECOMMENDER_STATE);
    setRecommenderResult(null);
    setRecommenderError(null);
  };

  const handleReset = () => {
    setModal({
      isOpen: true,
      type: 'confirm',
      title: 'Reset Session',
      message: 'Are you sure you want to completely clear all your inputs and wipe your search history? This action cannot be undone.',
      onConfirm: () => {
        localStorage.removeItem('lagos_rent_history');
        setHistory([]); setActiveResult(null); setResultFormData(null); setFormData(INITIAL_FORM_STATE);
        setRecommenderFormData(INITIAL_RECOMMENDER_STATE); setRecommenderResult(null);
        closeModal();
      }
    });
  };

  const handleDeleteHistory = (idToDelete) => {
    setModal({
      isOpen: true,
      type: 'confirm',
      title: 'Delete Entry',
      message: 'Are you sure you want to remove this estimate from your search history?',
      onConfirm: () => {
        const updatedHistory = history.filter(item => item.id !== idToDelete);
        setHistory(updatedHistory);
        localStorage.setItem('lagos_rent_history', JSON.stringify(updatedHistory));
        if (activeResult && updatedHistory.length === 0) {
          setActiveResult(null); setResultFormData(null); setFormData(INITIAL_FORM_STATE);
        }
        closeModal();
      }
    });
  };

  const handleHistorySelect = (item) => {
    if (!item.fullFormData) {
      setModal({
        isOpen: true,
        type: 'alert',
        title: 'Incomplete Record',
        message: 'This is an older history record that cannot be fully restored. Please generate a new estimate.',
        onConfirm: closeModal
      });
      return;
    }
    setActiveView('estimator');
    setFormData(item.fullFormData); setActiveResult(item.fullResult);
    setResultFormData(item.fullFormData);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleClearSelections = () => {
    const clearedFeatures = {};
    FEATURES_LIST.forEach(feat => { clearedFeatures[feat.id] = 0; });
    setFormData(prev => ({ ...prev, ...clearedFeatures }));
    setActiveResult(null);
    setResultFormData(null);
  };

  return (
    <div className="min-h-screen flex flex-col relative">
      <Navbar isDarkMode={isDarkMode} toggleTheme={() => setIsDarkMode(!isDarkMode)} />

      <main className="grow w-full px-4 sm:px-6 md:px-8 py-8 mx-auto flex flex-col">
        <header className="mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 animate-fade-up">
          <div>
            <h1 className="text-4xl font-spicy mb-1 text-brand-dark dark:text-white">
              {activeView === 'estimator' ? 'Residential Property Rent Estimator' : 'Budget-Based Recommender'}
            </h1>
            <p className="opacity-80 text-lg text-brand-dark dark:text-gray-300">
              {activeView === 'estimator'
                ? <>Select your location and property features to get an AI-powered estimate. <span className="italic text-brand-accent font-medium">See footer for disclaimer.</span></>
                : <>Set a budget and your must-haves, and see the best-value properties that fit. <span className="italic text-brand-accent font-medium">See footer for disclaimer.</span></>
              }
            </p>
          </div>
          <button onClick={handleReset} className="px-4 py-2 border border-brand-accent text-brand-accent hover:bg-brand-accent hover:text-white rounded-md transition-colors font-semibold text-sm shadow-sm">
            Reset Session
          </button>
        </header>

        {/* [ADDED] Page switch */}
        <div className="inline-flex mb-6 rounded-lg border border-brand-gray/20 dark:border-white/10 overflow-hidden self-start animate-fade-up delay-100">
          <button
            onClick={() => setActiveView('estimator')}
            className={`px-5 py-2 text-sm font-bold transition-colors ${activeView === 'estimator' ? 'bg-brand-accent text-white' : 'bg-transparent text-brand-dark dark:text-white hover:bg-white/20 dark:hover:bg-white/10'}`}
          >
            Estimator
          </button>
          <button
            onClick={() => setActiveView('recommender')}
            className={`px-5 py-2 text-sm font-bold transition-colors ${activeView === 'recommender' ? 'bg-brand-accent text-white' : 'bg-transparent text-brand-dark dark:text-white hover:bg-white/20 dark:hover:bg-white/10'}`}
          >
            Recommender
          </button>
        </div>

        {activeView === 'estimator' ? (
          <div className="rounded-2xl shadow-xl overflow-hidden mb-12 bg-brand-bg/20 border border-brand-gray/20 dark:border-white/10 animate-fade-up delay-100">
            <EstimatorForm formData={formData} setFormData={setFormData} onSubmit={handleEstimateRequest} isLoading={isLoading} apiError={apiError} onClear={handleClearSelections} />
            <ResultsDisplay result={activeResult} formData={resultFormData} />
            <HistoryTable history={history} onSelect={handleHistorySelect} onDelete={handleDeleteHistory} />
          </div>
        ) : (
          <div className="rounded-2xl shadow-xl overflow-hidden mb-12 bg-brand-bg/20 border border-brand-gray/20 dark:border-white/10 animate-fade-up delay-100">
            <RecommenderForm
              formData={recommenderFormData}
              setFormData={setRecommenderFormData}
              onSubmit={handleRecommendRequest}
              onSortChange={handleSortChange}
              isLoading={isRecommending}
              apiError={recommenderError}
              onClear={handleClearRecommender}
            />
            <RecommenderResults result={recommenderResult} isLoading={isRecommending} sortBy={recommenderFormData.sort_by} />
          </div>
        )}
      </main>

      <Footer />

      <ConfirmModal
        isOpen={modal.isOpen}
        type={modal.type}
        title={modal.title}
        message={modal.message}
        onConfirm={modal.onConfirm}
        onCancel={closeModal}
      />
    </div>
  );
}