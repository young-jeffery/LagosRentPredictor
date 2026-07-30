// src/components/ui/Tooltip.jsx
import { useState, useRef, useEffect } from 'react';

// Added a delay prop (default is 700 milliseconds)
export default function Tooltip({ content, children, delay = 700 }) {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef(null);

  // Starts the timer when the mouse enters
  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };

  // Clears the timer and hides tooltip if mouse leaves
  const handleMouseLeave = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  // Cleanup to prevent memory leaks if the component unmounts while hovered
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div 
      className="relative inline-flex"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleMouseEnter} // For keyboard accessibility
      onBlur={handleMouseLeave}
    >
      {children}
      
      {isVisible && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2.5 w-56 p-3 bg-brand-dark dark:bg-brand-gray text-brand-bg dark:text-brand-dark text-xs sm:text-sm font-medium rounded-lg shadow-2xl z-50 text-center pointer-events-none animate-scale-in origin-bottom">
          {content}
          {/* Tooltip pointer arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-[6px] border-transparent border-t-brand-dark dark:border-t-white"></div>
        </div>
      )}
    </div>
  );
}