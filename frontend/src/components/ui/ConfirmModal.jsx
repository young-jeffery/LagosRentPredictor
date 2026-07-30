// src/components/ui/ConfirmModal.jsx
import { useEffect } from 'react';

export default function ConfirmModal({ isOpen, type, title, message, onConfirm, onCancel }) {
  // Prevent scrolling on the main body when the modal is open
  useEffect(() => {
    if (isOpen) document.body.style.overflow = 'hidden';
    else document.body.style.overflow = 'unset';
    return () => { document.body.style.overflow = 'unset'; };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-100 flex items-center justify-center p-4 bg-brand-dark/40 dark:bg-black/60 backdrop-blur-sm">
      <div className="bg-brand-bg dark:bg-[#16202b] rounded-2xl shadow-2xl max-w-md w-full p-6 sm:p-8 border border-brand-gray/20 dark:border-white/10 animate-scale-in">
        
        <h3 className="text-2xl font-rye text-brand-dark dark:text-brand-dark mb-3">
          {title}
        </h3>
        
        <p className="text-brand-gray dark:text-gray-300 mb-8 text-sm sm:text-base leading-relaxed">
          {message}
        </p>
        
        <div className="flex justify-end items-center gap-3">
          {type === 'confirm' && (
            <button 
              onClick={onCancel}
              className="px-5 py-2.5 rounded-lg font-bold text-sm text-brand-gray dark:text-gray-300 hover:bg-brand-gray/10 dark:hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>
          )}
          <button 
            onClick={onConfirm}
            className="px-6 py-2.5 rounded-lg font-bold text-sm bg-brand-accent hover:bg-brand-accent-hover text-white shadow-md transition-colors"
          >
            {type === 'confirm' ? 'Yes, Proceed' : 'Okay, Got it'}
          </button>
        </div>
        
      </div>
    </div>
  );
}