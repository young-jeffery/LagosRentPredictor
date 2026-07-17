import { formatNaira, getApproximatedRanges, getBreakdownRange } from '../utils/helpers';
import { FEATURES_LIST } from '../constants/appConstants';

export default function ResultsDisplay({ result, formData }) {
  if (!result) return null;

  const { estimated_tct, negotiation_lower, negotiation_upper, breakdown } = result;
  const selectedFeatures = FEATURES_LIST.filter(f => formData[f.id] === 1);

  const { approxLower, approxUpper } = getApproximatedRanges(negotiation_lower, negotiation_upper);

  const breakdownItems = [
    { label: 'Base Rent', value: breakdown.base_rent, delay: 'delay-100', ...getBreakdownRange(breakdown.base_rent, estimated_tct, approxLower, approxUpper) },
    { label: 'Agency Fee', value: breakdown.agency_fee, delay: 'delay-200', ...getBreakdownRange(breakdown.agency_fee, estimated_tct, approxLower, approxUpper) },
    { label: 'Legal Fee', value: breakdown.legal_fee, delay: 'delay-300', ...getBreakdownRange(breakdown.legal_fee, estimated_tct, approxLower, approxUpper) },
    { label: 'Caution Deposit', value: breakdown.caution_deposit, delay: 'delay-400', ...getBreakdownRange(breakdown.caution_deposit, estimated_tct, approxLower, approxUpper) }
  ];

  return (
    <section aria-label="Rent Estimation Results" className="border-t border-brand-gray/20 flex flex-col lg:flex-row bg-transparent">
      <div className="bg-[#141D26] px-8 py-12 text-center lg:w-1/3 flex flex-col justify-center relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 bg-brand-accent"></div>
        <div className="animate-scale-in">
          <h2 className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">
            Estimated Total Cost to Tenant
          </h2>
          <p className="text-brand-accent font-spicy text-lg mb-4">
            {formData.Bedrooms} Bed, {formData.Bathrooms} Bath in {formData.Consolidated_Location}
          </p>
          <div className="text-5xl lg:text-6xl font-spicy mb-6 text-white drop-shadow-md">
            {formatNaira(estimated_tct)}
          </div>
          <div className="inline-flex flex-col mx-auto items-center gap-1 bg-brand-bg/20 rounded-xl px-6 py-3 border border-gray-600 shadow-inner">
            <span className="text-gray-300 text-xs uppercase tracking-wide">Negotiation Range</span>
            <span className="font-spicy text-lg text-white">
              {formatNaira(approxLower)} - {formatNaira(approxUpper)}
            </span>
          </div>
        </div>
      </div>

      <div className="p-8 lg:w-2/3">
        <h3 className="text-2xl font-spicy mb-6 text-brand-dark dark:text-white animate-fade-up">Cost Breakdown</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {breakdownItems.map((item, index) => (
            <div 
              key={index} 
              className={`p-4 rounded-xl border border-brand-gray/20 shadow-sm bg-white/20 dark:bg-black/20 animate-fade-up ${item.delay}`}
            >
              <p className="text-xs font-bold uppercase opacity-70 mb-1 text-brand-dark dark:text-gray-300">{item.label}</p>
              <p className="text-xl font-spicy text-brand-dark dark:text-white mb-1">{formatNaira(item.value)}</p>
              
              <div className="text-[11px] font-semibold text-brand-accent pt-2 border-t border-brand-gray/10 dark:border-white/10">
                Range: {formatNaira(item.itemLower)} - {formatNaira(item.itemUpper)}
              </div>
            </div>
          ))}
        </div>

        <div className="pt-6 border-t border-brand-gray/20 animate-fade-up delay-400">
          <h4 className="text-sm font-bold uppercase mb-3 text-brand-dark dark:text-white">Included Features</h4>
          <div className="flex flex-wrap gap-2">
            {selectedFeatures.length > 0 ? selectedFeatures.map(f => (
              <span key={f.id} className="text-xs font-semibold px-3 py-1 rounded-md bg-white/20 dark:bg-black/40 text-brand-dark dark:text-white border border-brand-gray/20 dark:border-white/5">
                {f.label}
              </span>
            )) : (
              <span className="text-sm opacity-60 italic text-brand-dark dark:text-gray-300">No additional amenities selected.</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}