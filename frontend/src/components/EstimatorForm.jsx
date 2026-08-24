// src/components/EstimatorForm.jsx
import SearchableSelect from './ui/SearchableSelect';
import Tooltip from './ui/Tooltip';
import { LAGOS_DISTRICTS, LAGOS_LOCATIONS, FEATURES_LIST, PROPERTY_TYPES } from '../constants/appConstants';
// [ADDED] Property-type-aware bed/bath bounds
import { getBedBathConfig, clamp } from '../constants/propertyTypeConstraints';

const PROPERTY_TYPE_OPTIONS = PROPERTY_TYPES.map(type => ({ label: type, value: type }));

export default function EstimatorForm({ formData, setFormData, onSubmit, isLoading, apiError, onClear }) {
  // [ADDED] Bounds for whatever Property_Type is currently selected
  const bedBathConfig = getBedBathConfig(formData.Property_Type);
  const bedsFixed = bedBathConfig.bedMin === bedBathConfig.bedMax;
  const bathsFixed = bedBathConfig.bathMin === bedBathConfig.bathMax;

  const handleNumberChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: Number(value) }));
  };

  // [ADDED] Clamps current Bedrooms/Bathrooms into the new type's valid
  // range whenever Property_Type changes -- e.g. switching from Flat/
  // Apartment (6 beds) to Studio auto-corrects to 1 rather than letting an
  // invalid combination sit in the form until submission fails.
  const handlePropertyTypeChange = (selected) => {
    const newType = selected.value;
    const config = getBedBathConfig(newType);
    setFormData(p => ({
      ...p,
      Property_Type: newType,
      Bedrooms: clamp(p.Bedrooms, config.bedMin, config.bedMax),
      Bathrooms: clamp(p.Bathrooms, config.bathMin, config.bathMax),
    }));
  };

  const toggleFeature = (featureId) => {
    setFormData(prev => ({ ...prev, [featureId]: prev[featureId] === 1 ? 0 : 1 }));
  };

  return (
    <form onSubmit={onSubmit} className="p-6 sm:p-8">
      <div className="max-w-4xl flex flex-col md:flex-row gap-6 mb-2 items-start">
        <div className="w-full md:flex-1">
          <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Location</label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <SearchableSelect
              options={LAGOS_DISTRICTS}
              value={formData.Main_District ? { label: formData.Main_District } : null}
              onChange={(selected) => setFormData(p => ({
                ...p,
                Main_District: selected.district,
                Official_Sub_Location: ''
              }))}
              placeholder="Select district..."
            />

            <SearchableSelect
              options={
                formData.Main_District
                  ? LAGOS_LOCATIONS
                      .filter(loc => loc.district === formData.Main_District)
                      .map(loc => ({ label: loc.subLocation, subLocation: loc.subLocation, district: loc.district }))
                  : []
              }
              value={formData.Official_Sub_Location ? { label: formData.Official_Sub_Location } : null}
              onChange={(selected) => setFormData(p => ({
                ...p,
                Official_Sub_Location: selected.subLocation
              }))}
              placeholder={formData.Main_District ? "Select sub-location..." : "Select a district first"}
            />
          </div>
        </div>

        <div className="w-full md:w-56 shrink-0">
          <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Property Type</label>
          <SearchableSelect
            options={PROPERTY_TYPE_OPTIONS}
            value={formData.Property_Type ? { label: formData.Property_Type } : null}
            onChange={handlePropertyTypeChange}
            placeholder="Select property type..."
          />
        </div>
        
        <div className="w-20 shrink-0">
          <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Beds</label>
          <input 
            type="number" name="Bedrooms"
            min={bedBathConfig.bedMin} max={bedBathConfig.bedMax}
            value={formData.Bedrooms} onChange={handleNumberChange}
            disabled={isLoading || bedsFixed}
            className="w-full font-spicy text-center text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2 focus:ring-2 focus:ring-brand-accent outline-none bg-white/20 dark:bg-black/20 text-brand-dark dark:text-white disabled:opacity-60 transition-colors"
          />
        </div>

        <div className="w-20 shrink-0">
          <label className="block text-sm font-semibold uppercase text-brand-dark dark:text-white mb-1.5 opacity-80">Baths</label>
          <input 
            type="number" name="Bathrooms"
            min={bedBathConfig.bathMin} max={bedBathConfig.bathMax}
            value={formData.Bathrooms} onChange={handleNumberChange}
            disabled={isLoading || bathsFixed}
            className="w-full font-spicy text-center text-lg border border-brand-gray/30 dark:border-white/20 rounded-md p-2 focus:ring-2 focus:ring-brand-accent outline-none bg-white/20 dark:bg-black/20 text-brand-dark dark:text-white disabled:opacity-60 transition-colors"
          />
        </div>
      </div>

      {/* [ADDED] Explanatory note for property types with fixed or unusual bed/bath rules */}
      {bedBathConfig.note && (
        <p className="text-xs italic opacity-70 mb-6 text-brand-dark dark:text-gray-300">{bedBathConfig.note}</p>
      )}
      {!bedBathConfig.note && <div className="mb-8" />}

      <div className="mb-6">
        <h2 className="text-xl font-bold text-brand-dark dark:text-white mb-3 font-rye">Property Features</h2>
        <div className="flex flex-wrap gap-2.5">
          {FEATURES_LIST.map((feature) => {
            const isActive = formData[feature.id] === 1;
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

      {apiError && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/30 border-l-4 border-brand-accent text-brand-dark dark:text-white rounded animate-scale-in">
          <p className="font-medium text-sm">{apiError}</p>
        </div>
      )}

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
          disabled={isLoading}
          className="w-full sm:w-auto px-8 py-2.5 rounded-lg bg-brand-accent hover:bg-brand-accent-hover text-white font-bold tracking-wider capitalize text-lg font-rye shadow-lg transition-all active:scale-[0.98] disabled:opacity-80 disabled:cursor-not-allowed flex justify-center items-center h-13 min-w-50"
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