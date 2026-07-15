// src/components/ui/SearchableSelect.jsx
import { useState, useRef, useEffect } from 'react';

export default function SearchableSelect({ options, value, onChange, placeholder = "Select an option..." }) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const wrapperRef = useRef(null);

  const filteredOptions = options.filter(option =>
    option.toLowerCase().includes(searchTerm.toLowerCase())
  );

  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (option) => {
    onChange(option);
    setSearchTerm('');
    setIsOpen(false);
  };

  return (
    <div ref={wrapperRef} className="relative w-full">
      <div
        className="w-full bg-white dark:bg-[#0b1116] border border-brand-gray/30 dark:border-white/20 rounded-md p-2.5 text-left cursor-pointer flex justify-between items-center focus-within:ring-2 focus-within:ring-brand-accent transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className={value ? "text-brand-dark font-medium" : "text-gray-500 dark:text-gray-400"}>
          {value || placeholder}
        </span>
        <svg className="w-5 h-5 text-brand-gray dark:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-[#141D26] border border-brand-gray/20 dark:border-white/10 rounded-md shadow-lg max-h-60 overflow-auto">
          <div className="sticky top-0 bg-white dark:bg-[#141D26] p-2 border-b border-gray-100 dark:border-white/10">
            <input
              type="text"
              className="w-full px-3 py-2 border border-brand-gray/30 dark:border-white/20 rounded focus:outline-none focus:border-brand-accent bg-white dark:bg-[#0b1116] text-brand-dark dark:text-white text-sm"
              placeholder="Type to search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onClick={(e) => e.stopPropagation()}
            />
          </div>

          <ul className="py-1">
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <li
                  key={option}
                  className={`px-4 py-2 cursor-pointer hover:bg-brand-accent hover:text-white transition-colors duration-150 ${value === option ? 'bg-brand-gray dark:bg-white/20 text-white' : 'text-brand-dark dark:text-gray-200'}`}
                  onClick={() => handleSelect(option)}
                >
                  {option}
                </li>
              ))
            ) : (
              <li className="px-4 py-2 text-gray-500 dark:text-gray-400 text-sm">No locations found.</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}