import { useState } from 'react';
import { formatNaira } from '../utils/helpers';

export default function HistoryTable({ history = [], onSelect, onDelete }) {
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  if (history.length === 0) return null;

  const totalItems = history.length;
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, totalItems);
  const currentData = history.slice(startIndex, endIndex);

  const formatDate = (isoString) => {
    return new Intl.DateTimeFormat('en-NG', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    }).format(new Date(isoString));
  };

  // [ADDED] v3 -- entries saved before Property_Type existed won't have
  // `propertyType` on the record, so fall back to fullFormData if present,
  // else show a plain dash rather than "undefined".
  const getPropertyType = (item) => item.propertyType || item.fullFormData?.Property_Type || '—';

  return (
    <section aria-label="Recent Estimates" className="border-t border-brand-gray/20 dark:border-white/10 bg-transparent">
      <div className="px-8 py-6 flex flex-col sm:flex-row justify-between items-start sm:items-end gap-2 border-b border-brand-gray/20 dark:border-white/5">
        <div>
          <h2 className="text-2xl font-spicy text-brand-dark dark:text-white">Search History</h2>
          <p className="text-xs font-semibold opacity-70 text-brand-dark dark:text-gray-400 mt-1">
            This table securely holds a maximum of your 30 most recent estimates.
          </p>
        </div>
        <span className="text-sm font-bold opacity-80 text-brand-dark dark:text-gray-300">
          Showing {startIndex + 1}-{endIndex} of {totalItems}
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="text-xs uppercase tracking-wider bg-transparent dark:bg-white/5 text-brand-dark dark:text-gray-300 border-b border-brand-gray/20 dark:border-transparent">
              <th className="px-8 py-4 font-bold">Location</th>
              <th className="px-8 py-4 font-bold">Type</th>
              <th className="px-8 py-4 font-bold">Size</th>
              <th className="px-8 py-4 font-bold">Est. Rent (TCT)</th>
              <th className="px-8 py-4 font-bold">Date</th>
              <th className="px-8 py-4 font-bold text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-brand-gray/20 dark:divide-white/5">
            {currentData.map((item) => (
              <tr key={item.id} className="transition-colors group hover:bg-white/30 dark:hover:bg-white/5 text-brand-dark dark:text-white">
                <td onClick={() => onSelect(item)} className="px-8 py-5 font-bold cursor-pointer group-hover:text-brand-accent transition-colors">
                  {item.location}
                </td>
                <td onClick={() => onSelect(item)} className="px-8 py-5 font-medium cursor-pointer">
                  {getPropertyType(item)}
                </td>
                <td onClick={() => onSelect(item)} className="px-8 py-5 font-medium cursor-pointer">
                  {item.beds} Bed, {item.baths} Bath
                </td>
                <td onClick={() => onSelect(item)} className="px-8 py-5 font-spicy text-lg text-brand-accent cursor-pointer">
                  {formatNaira(item.estimated_tct)}
                </td>
                <td onClick={() => onSelect(item)} className="px-8 py-5 text-sm font-semibold opacity-70 cursor-pointer">
                  {formatDate(item.timestamp)}
                </td>
                <td className="px-8 py-5 text-center">
                  <button 
                    onClick={(e) => { e.stopPropagation(); onDelete(item.id); }}
                    className="p-2 text-red-500 hover:bg-red-500/10 rounded-full transition-colors"
                    title="Delete Entry"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="px-8 py-4 border-t border-brand-gray/20 dark:border-white/5 flex justify-between items-center bg-transparent">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="px-5 py-2 text-sm font-bold rounded-md border border-brand-gray/40 dark:border-white/20 disabled:opacity-50 transition-colors bg-transparent text-brand-dark dark:text-white"
          >
            Previous
          </button>
          <span className="text-sm font-bold opacity-80 text-brand-dark dark:text-white">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="px-5 py-2 text-sm font-bold rounded-md border border-brand-gray/40 dark:border-white/20 disabled:opacity-50 transition-colors bg-transparent text-brand-dark dark:text-white"
          >
            Next
          </button>
        </div>
      )}
    </section>
  );
}