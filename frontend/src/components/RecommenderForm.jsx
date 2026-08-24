// src/components/RecommenderForm.jsx
import SearchableSelect from './ui/SearchableSelect';
import Tooltip from './ui/Tooltip';
import { LAGOS_DISTRICTS, LAGOS_LOCATIONS, FEATURES_LIST, PROPERTY_TYPES } from '../constants/appConstants';
// [ADDED] Property-type-aware bed/bath bounds
import { getBedBathConfig, clamp } from '../constants/propertyTypeConstraints';

const ANY_DISTRICT_OPTIONS = [{ label: 'Any District', district: null }, ...LAGOS_DISTRICTS];
const ANY_PROPERTY_TYPE_OPTIONS = [
    { label: 'Any Property Type', value: null },
    ...PROPERTY_TYPES.map(type => ({ label: type, value: type })),
];
// Generic bounds used when Property_Type is "Any" -- the backend clamps
// per-candidate-type regardless, so this is just a sensible open range.
const ANY_TYPE_BOUNDS = { bedMin: 1, bedMax: 6, bathMin: 1, bathMax: 6 };

// [ADDED] Formats a raw number as a comma-grouped string for display,
// e.g. 5000000 -> "5,000,000".
const formatWithCommas = (value) => {
    if (value === '' || value === null || value === undefined) return '';
    return Number(value).toLocaleString('en-US');
};

export default function RecommenderForm({ formData, setFormData, onSubmit, onSortChange, isLoading, apiError, onClear }) {
    // [ADDED] Only apply type-specific bounds when a specific type is chosen.
    // "Any Property Type" keeps the generic 1-6 range since the backend
    // clamps per-row against each candidate's own type anyway.
    const activeConfig = formData.Property_Type ? getBedBathConfig(formData.Property_Type) : null;
    const bedBounds = activeConfig || ANY_TYPE_BOUNDS;
    const bathBounds = activeConfig || ANY_TYPE_BOUNDS;
    const bedsFixed = activeConfig && activeConfig.bedMin === activeConfig.bedMax;
    const bathsFixed = activeConfig && activeConfig.bathMin === activeConfig.bathMax;

    // [MODIFIED] Strips non-digits so the comma-formatted display can be
    // typed into naturally, then stores the raw number in state.
    const handleBudgetChange = (e) => {
        const raw = e.target.value.replace(/[^0-9]/g, '');
        setFormData(prev => ({ ...prev, budget: raw === '' ? '' : Number(raw) }));
    };

    const handleOptionalNumberChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value === '' ? null : Number(value) }));
    };

    // [ADDED] Clamps an already-set explicit preference into the new type's
    // range. Leaves it alone if it was already "Any" (null).
    const handlePropertyTypeChange = (selected) => {
        const newType = selected.value;
        const config = newType ? getBedBathConfig(newType) : null;
        setFormData(p => ({
            ...p,
            Property_Type: newType,
            Bedrooms: (config && p.Bedrooms !== null) ? clamp(p.Bedrooms, config.bedMin, config.bedMax) : p.Bedrooms,
            Bathrooms: (config && p.Bathrooms !== null) ? clamp(p.Bathrooms, config.bathMin, config.bathMax) : p.Bathrooms,
        }));
    };

    const toggleFeature = (featureId) => {
        setFormData(prev => {
            const isOn = prev.preferred_features.includes(featureId);
            return {
                ...prev,
                preferred_features: isOn
                    ? prev.preferred_features.filter(id => id !== featureId)
                    : [...prev.preferred_features, featureId],
            };
        });
    };

    const subLocationOptions = formData.Main_District
        ? [
            { label: 'Any Sub-location', subLocation: null },
            ...LAGOS_LOCATIONS
                .filter(loc => loc.district === formData.Main_District)
                .map(loc => ({ label: loc.subLocation, subLocation: loc.subLocation })),
        ]
        : [];

    return (
        <form onSubmit={onSubmit} className="p-6 sm:p-8">
            <div className="mb-8">
                <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">
                    Budget (Max Total Cost to Tenant)
                </label>
                {/* [MODIFIED] type="text" + comma formatting instead of a raw number input */}
                <input
                    type="text" inputMode="numeric" required
                    value={formatWithCommas(formData.budget)}
                    onChange={handleBudgetChange}
                    disabled={isLoading}
                    placeholder="e.g. 5,000,000"
                    className="w-full max-w-xs font-spicy text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2.5 focus:ring-2 focus:ring-brand-accent outline-none bg-white/20 dark:bg-black/20 text-brand-dark dark:text-white disabled:opacity-50 transition-colors"
                />
                <p className="text-xs opacity-60 mt-1 text-brand-dark dark:text-gray-300">
                    The most you're willing to pay in total, including agency/legal/caution fees.
                </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-2">
                <div>
                    <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">District (optional)</label>
                    <SearchableSelect
                        options={ANY_DISTRICT_OPTIONS}
                        value={formData.Main_District ? { label: formData.Main_District } : { label: 'Any District' }}
                        onChange={(selected) => setFormData(p => ({ ...p, Main_District: selected.district, Official_Sub_Location: null }))}
                        placeholder="Any District"
                    />
                </div>

                <div>
                    <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Sub-location (optional)</label>
                    <SearchableSelect
                        options={subLocationOptions}
                        value={formData.Official_Sub_Location ? { label: formData.Official_Sub_Location } : { label: formData.Main_District ? 'Any Sub-location' : 'Select a district first' }}
                        onChange={(selected) => setFormData(p => ({ ...p, Official_Sub_Location: selected.subLocation }))}
                        placeholder={formData.Main_District ? 'Any Sub-location' : 'Select a district first'}
                    />
                </div>

                <div>
                    <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Property Type (optional)</label>
                    <SearchableSelect
                        options={ANY_PROPERTY_TYPE_OPTIONS}
                        value={formData.Property_Type ? { label: formData.Property_Type } : { label: 'Any Property Type' }}
                        onChange={handlePropertyTypeChange}
                        placeholder="Any Property Type"
                    />
                </div>

                <div className="grid grid-cols-2 gap-2">
                    <div>
                        <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Beds</label>
                        <input
                            type="number" name="Bedrooms"
                            min={bedBounds.bedMin} max={bedBounds.bedMax}
                            value={bedsFixed ? activeConfig.bedDefault : (formData.Bedrooms ?? '')}
                            onChange={handleOptionalNumberChange}
                            disabled={isLoading || bedsFixed}
                            placeholder="Any"
                            className="w-full font-spicy text-center text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2 focus:ring-2 focus:ring-brand-accent outline-none bg-white/20 dark:bg-black/20 text-brand-dark dark:text-white disabled:opacity-60 transition-colors"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Baths</label>
                        <input
                            type="number" name="Bathrooms"
                            min={bathBounds.bathMin} max={bathBounds.bathMax}
                            value={bathsFixed ? activeConfig.bathDefault : (formData.Bathrooms ?? '')}
                            onChange={handleOptionalNumberChange}
                            disabled={isLoading || bathsFixed}
                            placeholder="Any"
                            className="w-full font-spicy text-center text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2 focus:ring-2 focus:ring-brand-accent outline-none bg-white/20 dark:bg-black/20 text-brand-dark dark:text-white disabled:opacity-60 transition-colors"
                        />
                    </div>
                </div>
            </div>

            {/* [ADDED] Explanatory note for fixed-bed/bath types, same pattern as EstimatorForm */}
            {activeConfig?.note ? (
                <p className="text-xs italic opacity-70 mb-6 text-brand-dark dark:text-gray-300">{activeConfig.note}</p>
            ) : (
                <p className="text-xs opacity-60 mb-8 text-brand-dark dark:text-gray-300">
                    Leave Beds/Baths blank for no preference -- each property type will use its own typical default.
                </p>
            )}

            <div className="mb-6">
                <h2 className="text-xl font-bold text-brand-dark dark:text-white mb-3 font-rye">Preferred Features</h2>
                <div className="flex flex-wrap gap-2.5">
                    {FEATURES_LIST.map((feature) => {
                        const isActive = formData.preferred_features.includes(feature.id);
                        return (
                            <Tooltip key={feature.id} content={feature.description}>
                                <button
                                    type="button"
                                    disabled={isLoading}
                                    onClick={() => toggleFeature(feature.id)}
                                    className={`px-3 py-1.5 rounded-md text-sm font-bold transition-all duration-200 border 
                    ${isActive
                                            ? 'bg-brand-gray text-brand-bg border-brand-gray shadow-sm dark:bg-white dark:text-brand-gray dark:border-white scale-[1.02]'
                                            : 'bg-transparent text-brand-gray border-brand-gray/40 hover:bg-white/20 dark:bg-[#1a2530] dark:text-gray-300 dark:border-white/10 dark:hover:border-white/30 dark:hover:bg-[#233140]'}
                    disabled:opacity-50 disabled:cursor-not-allowed
                  `}
                                >
                                    {feature.label}
                                </button>
                            </Tooltip>
                        );
                    })}
                </div>
            </div>

            <div className="mb-8">
                <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">
                    Sort Results By
                </label>
                <div className="inline-flex rounded-lg border border-brand-gray/30 dark:border-white/20 overflow-hidden">
                    {[
                        { value: 'value', label: 'Best Value' },
                        { value: 'closest_to_budget', label: 'Closest to Budget' },
                        { value: 'cheapest', label: 'Cheapest First' },
                    ].map((option) => (
                        <button
                            key={option.value}
                            type="button"
                            disabled={isLoading}
                            onClick={() => onSortChange(option.value)}
                            className={`px-4 py-2 text-sm font-bold transition-colors disabled:opacity-50
                ${formData.sort_by === option.value
                                    ? 'bg-brand-accent text-white'
                                    : 'bg-transparent text-brand-dark dark:text-white hover:bg-white/20 dark:hover:bg-white/10'}
              `}
                        >
                            {option.label}
                        </button>
                    ))}
                </div>
                <p className="text-xs opacity-60 mt-1.5 text-brand-dark dark:text-gray-300">
                    {formData.sort_by === 'value' && "Ranks by how much you're saving relative to what that area/property type normally costs."}
                    {formData.sort_by === 'closest_to_budget' && "Ranks by properties that use as much of your budget as possible without going over."}
                    {formData.sort_by === 'cheapest' && "Ranks by lowest price first, for maximum savings."}
                </p>
            </div>

            {apiError && (
                <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/30 border-l-4 border-brand-accent text-brand-dark dark:text-white rounded animate-scale-in">
                    <p className="font-medium text-sm">{apiError}</p>
                </div>
            )}

            {/* [ADDED] Clear Selections button, mirroring EstimatorForm */}
            <div className="flex flex-col sm:flex-row justify-end items-center gap-4 pt-6 border-t border-brand-gray/20 dark:border-white/10">
                <button
                    type="button"
                    onClick={onClear}
                    disabled={isLoading}
                    className="w-full sm:w-auto px-5 py-2.5 rounded-lg border border-brand-gray/40 dark:border-white/30 text-brand-dark dark:text-gray-300 bg-transparent hover:bg-white/20 dark:hover:bg-white/10 font-bold text-sm transition-colors disabled:opacity-50"
                >
                    Clear Selections
                </button>
                <button
                    type="submit"
                    disabled={isLoading || !formData.budget}
                    className="w-full sm:w-auto px-8 py-2.5 rounded-lg bg-brand-accent hover:bg-brand-accent-hover text-white font-bold tracking-wider capitalize text-lg font-rye shadow-lg transition-all active:scale-[0.98] disabled:opacity-80 disabled:cursor-not-allowed flex justify-center items-center h-13 min-w-50"
                >
                    {isLoading ? (
                        <div className="sound-wave">
                            <div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div>
                        </div>
                    ) : (
                        "Find Properties"
                    )}
                </button>
            </div>
        </form>
    );
}