// src/components/RecommenderResults.jsx
import { formatNaira } from '../utils/helpers';

// [ADDED] Badge label depends on how results were actually sorted
const BADGE_LABELS = {
  value: 'Best Value',
  closest_to_budget: 'Closest to Budget',
  cheapest: 'Cheapest',
};

export default function RecommenderResults({ result, isLoading, sortBy }) {
  if (isLoading) return null;
  if (!result) return null;

  const { matches, total_matches_before_limit, no_match_suggestion } = result;
  const badgeLabel = BADGE_LABELS[sortBy] || 'Top Pick';

  if (no_match_suggestion) {
    return (
      <section className="border-t border-brand-gray/20 dark:border-white/10 p-8">
        <div className="max-w-xl mx-auto py-8">
          <h3 className="text-2xl font-spicy mb-2 text-center text-brand-dark dark:text-white">No Matches Found</h3>
          <p className="text-center text-brand-dark dark:text-gray-300 mb-6">
            {no_match_suggestion.headline}
          </p>
          <ul className="space-y-3">
            {no_match_suggestion.tips.map((tip, i) => (
              <li
                key={i}
                className="flex items-start gap-3 p-4 rounded-xl border border-brand-gray/20 bg-white/20 dark:bg-black/20"
              >
                <span className="shrink-0 w-6 h-6 rounded-full bg-brand-accent text-white text-xs font-bold flex items-center justify-center mt-0.5">
                  {i + 1}
                </span>
                <span className="text-sm text-brand-dark dark:text-gray-200 leading-relaxed">{tip}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    );
  }

  return (
    <section className="border-t border-brand-gray/20 dark:border-white/10">
      <div className="px-8 py-6 border-b border-brand-gray/20 dark:border-white/5">
        <h2 className="text-2xl font-spicy text-brand-dark dark:text-white">
          {matches.length} Recommended {matches.length === 1 ? 'Property' : 'Properties'}
        </h2>
        {total_matches_before_limit > matches.length && (
          <p className="text-xs font-semibold opacity-70 text-brand-dark dark:text-gray-400 mt-1">
            {total_matches_before_limit} total matched your budget and filters -- showing the top {matches.length}, spread across different areas.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 p-8">
        {matches.map((prop, index) => (
          <div
            key={`${prop.Official_Sub_Location}-${prop.Property_Type}-${index}`}
            className="p-5 rounded-xl border border-brand-gray/20 shadow-sm bg-white/20 dark:bg-black/20 animate-fade-up relative"
          >
            {index === 0 && (
              <span className="absolute -top-2.5 right-4 text-xs font-bold uppercase px-2 py-0.5 rounded-full bg-brand-accent text-white shadow">
                {badgeLabel}
              </span>
            )}
            <p className="text-xs font-bold uppercase opacity-60 mb-1 text-brand-dark dark:text-gray-300">
              #{index + 1}
            </p>
            <h3 className="text-lg font-spicy text-brand-dark dark:text-white mb-0.5">
              {prop.Official_Sub_Location}, {prop.Main_District}
            </h3>
            <p className="text-sm font-semibold text-brand-accent mb-3">
              {prop.Property_Type} &bull; {prop.Bedrooms} Bed, {prop.Bathrooms} Bath
            </p>
            <p className="text-2xl font-spicy text-brand-dark dark:text-white">
              {formatNaira(prop.estimated_tct)}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}