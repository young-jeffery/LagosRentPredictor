import { useState, useRef, useEffect } from 'react';

// [MODIFIED] Now accepts an array of {label, ...rest} objects instead of
// plain strings, so we can carry the sub-location's district alongside
// its display label. onChange now returns the full selected object, not
// just a string, so the caller can pull out both subLocation and district
// in one selection.
export default function SearchableSelect({ options, value, onChange, placeholder = "Select an option..." }) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const wrapperRef = useRef(null);

  const filteredOptions = options.filter(option =>
    option.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) setIsOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (option) => {
    onChange(option); setSearchTerm(''); setIsOpen(false);
  };

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div
        className="w-full bg-white/20 dark:bg-black/20 border border-brand-gray/30 dark:border-white/20 rounded-md p-2.5 text-left cursor-pointer flex justify-between items-center focus-within:ring-2 focus-within:ring-brand-accent transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className={value ? "text-brand-dark dark:text-white font-medium" : "text-brand-gray/70 dark:text-gray-400"}>
          {value ? value.label : placeholder}
        </span>
        <svg className="w-5 h-5 text-brand-gray dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-[#141D26] border border-brand-gray/20 dark:border-white/10 rounded-md shadow-xl max-h-60 overflow-auto">
          <div className="sticky top-0 bg-white dark:bg-[#141D26] p-2 border-b border-brand-gray/10 dark:border-white/10">
            <input
              type="text"
              className="w-full px-3 py-2 border border-brand-gray/30 dark:border-white/20 rounded focus:outline-none focus:border-brand-accent bg-transparent text-brand-dark dark:text-white text-sm"
              placeholder="Type to search"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
          <ul className="py-1">
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <li
                  key={option.label}
                  className={`px-4 py-2 cursor-pointer hover:bg-brand-accent hover:text-white transition-colors duration-150 ${value?.label === option.label ? 'bg-brand-gray dark:bg-white/20 text-white' : 'text-brand-dark dark:text-gray-200'}`}
                  onClick={() => handleSelect(option)}
                >
                  {option.label}
                </li>
              ))
            ) : (
              <li className="px-4 py-2 text-brand-gray dark:text-gray-400 text-sm">No locations found.</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}