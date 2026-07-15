// src/components/EstimatorForm.jsx
import SearchableSelect from './ui/SearchableSelect';
import { LOCATIONS, FEATURES_LIST } from '../constants/appConstants';

export default function EstimatorForm({ formData, setFormData, onSubmit, isLoading, apiError }) {

    const handleNumberChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: Number(value) }));
    };

    const toggleFeature = (featureId) => {
        setFormData(prev => ({ ...prev, [featureId]: prev[featureId] === 1 ? 0 : 1 }));
    };

    const clearAllFeatures = () => {
        const clearedFeatures = {};
        FEATURES_LIST.forEach(feat => { clearedFeatures[feat.id] = 0; });
        setFormData(prev => ({ ...prev, ...clearedFeatures }));
    };

    return (
        <form onSubmit={onSubmit} className="p-6 sm:p-8">

            <div className="max-w-4xl grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div className="md:col-span-2">
                    <label className="block text-sm font-semibold uppercase text-brand-dark mb-1.5 opacity-80">Location</label>
                    <SearchableSelect
                        options={LOCATIONS}
                        value={formData.Consolidated_Location}
                        onChange={(val) => setFormData(p => ({ ...p, Consolidated_Location: val }))}
                        placeholder="Select district..."
                    />
                </div>

                <div className="md:col-span-1">
                    <label className="block text-sm font-semibold uppercase text-brand-dark mb-1.5 opacity-80">Bedrooms</label>
                    <input
                        type="number" name="Bedrooms" min="1" max="10"
                        value={formData.Bedrooms} onChange={handleNumberChange}
                        disabled={isLoading}
                        className="w-full font-rye text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2 focus:ring-2 focus:ring-brand-accent outline-none bg-white dark:bg-[#0b1116] text-brand-dark disabled:opacity-50 transition-colors"
                    />
                </div>

                <div className="md:col-span-1">
                    <label className="block text-sm font-semibold uppercase text-brand-dark mb-1.5 opacity-80">Bathrooms</label>
                    <input
                        type="number" name="Bathrooms" min="1" max="10"
                        value={formData.Bathrooms} onChange={handleNumberChange}
                        disabled={isLoading}
                        className="w-full font-rye text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2 focus:ring-2 focus:ring-brand-accent outline-none bg-white dark:bg-[#0b1116] text-brand-dark disabled:opacity-50 transition-colors"
                    />
                </div>
            </div>

            <div className="mb-6">
                <h2 className="text-lg font-bold text-brand-dark mb-3 font-rye">Property Features</h2>

                <div className="flex flex-wrap gap-2.5">
                    {FEATURES_LIST.map((feature) => {
                        const isActive = formData[feature.id] === 1;
                        return (
                            <button
                                key={feature.id}
                                type="button"
                                disabled={isLoading}
                                onClick={() => toggleFeature(feature.id)}
                                className={`px-3 py-1.5 rounded-md text-sm font-bold transition-all duration-200 border 
                  ${isActive
                                        ? 'bg-brand-gray dark:bg-white text-white dark:text-brand-gray border-brand-gray dark:border-white shadow-sm scale-[1.02]'
                                        : 'bg-brand-gray/10 dark:bg-[#1a2530] text-brand-dark dark:text-gray-300 border-brand-gray/30 dark:border-white/10 hover:border-brand-gray/50 dark:hover:border-white/30 hover:bg-brand-gray/20 dark:hover:bg-[#233140]'}
                  disabled:opacity-50 disabled:cursor-not-allowed
                `}
                            >
                                {feature.label}
                            </button>
                        );
                    })}
                </div>
            </div>

            {apiError && (
                <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/30 border-l-4 border-brand-accent text-brand-dark rounded animate-scale-in">
                    <p className="font-medium text-sm">{apiError}</p>
                </div>
            )}

            <div className="flex flex-col sm:flex-row justify-end items-center gap-4 pt-6 border-t border-brand-gray/10 dark:border-white/10">
                <button
                    type="button"
                    onClick={clearAllFeatures}
                    disabled={isLoading}
                    className="w-full sm:w-auto px-5 py-2.5 rounded-lg border border-brand-gray/40 dark:border-white/30 text-brand-gray dark:text-gray-300 bg-transparent hover:bg-brand-gray/10 dark:hover:bg-white/10 font-bold text-sm transition-colors disabled:opacity-50"
                >
                    Clear Selections
                </button>

                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full sm:w-auto px-8 py-2.5 rounded-lg bg-brand-accent hover:bg-brand-accent-hover text-white font-bold text-lg font-rye shadow-lg transition-all active:scale-[0.98] disabled:opacity-80 disabled:cursor-not-allowed flex justify-center items-center h-13 min-w-50"
                >
                    {isLoading ? (
                        <div className="sound-wave">
                            <div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div>
                        </div>
                    ) : (
                        "Estimate Rent"
                    )}
                </button>
            </div>
        </form>
    );
}