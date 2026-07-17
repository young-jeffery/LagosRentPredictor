export default function Navbar({ isDarkMode, toggleTheme }) {
  return (
    <nav className="sticky top-0 z-50 transition-colors bg-white/70 dark:bg-[#0b1116]/80 backdrop-blur-md border-b border-brand-gray/10 dark:border-white/10 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16 items-center">
          
          <div className="flex-1 flex justify-start">
            <span className="text-sm font-spicy text-brand-accent">Matric: 2209...</span>
          </div>

          <div className="flex-1 flex justify-center items-center space-x-3">
            <div className="w-8 h-8 rounded bg-brand-accent flex items-center justify-center font-bold text-white shadow-md">
              LR
            </div>
            <span className="font-spicy text-xl tracking-wide hidden sm:block whitespace-nowrap text-brand-dark dark:text-white">
              Lagos Rent Estimator
            </span>
          </div>

          <div className="flex-1 flex justify-end">
            <button 
              onClick={toggleTheme} 
              className="p-2 rounded bg-brand-gray/5 dark:bg-white/10 hover:bg-brand-gray/10 dark:hover:bg-white/20 transition-colors"
            >
              {isDarkMode ? (
                <svg className="w-5 h-5 text-yellow-300" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4.22 2.22a1 1 0 011.415 0l.707.707a1 1 0 01-1.414 1.414l-.707-.707a1 1 0 010-1.414zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zm-3.364 4.95l.707.707a1 1 0 11-1.414 1.414l-.707-.707a1 1 0 011.414-1.414zM10 18a1 1 0 01-1-1v-1a1 1 0 112 0v1a1 1 0 01-1 1zm-4.95-3.364l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM2 10a1 1 0 011-1h1a1 1 0 110 2H3a1 1 0 01-1-1zm2.22-4.22l-.707-.707a1 1 0 011.414-1.414l.707.707a1 1 0 01-1.414 1.414zM10 5a5 5 0 100 10 5 5 0 000-10z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}