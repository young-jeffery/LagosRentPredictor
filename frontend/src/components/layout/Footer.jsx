export default function Footer() {
  return (
    <footer className="w-full py-12 mt-auto border-t border-brand-gray/20 bg-transparent px-6 sm:px-8">
      <div className="max-w-6xl mx-auto flex flex-col lg:flex-row gap-8 lg:gap-12 items-center lg:items-stretch text-center lg:text-left">
        <div className="lg:w-2/5 flex flex-col justify-center space-y-3">
          <h3 className="font-spicy text-xl sm:text-2xl text-brand-dark dark:text-white uppercase tracking-wide">
            Development of an AI-Powered Web Application for Transparent Residential Rent Valuation in the Lagos Property Market
          </h3>
          <div className="pt-2">
            <p className="font-bold text-brand-accent text-lg">By: Idowu Mayowa Joseph</p>
            <p className="text-brand-dark dark:text-gray-300 font-medium text-sm sm:text-base">
              University of Lagos • Final Year Project
            </p>
          </div>
        </div>

        <div className="lg:w-3/5 p-5 sm:p-6 rounded-xl bg-white/20 dark:bg-white/5 border border-brand-gray/20 dark:border-white/10 text-brand-dark dark:text-gray-300 text-sm sm:text-base leading-relaxed text-left flex flex-col justify-center shadow-sm">
          <p className="mb-3">
            <strong className="text-brand-dark dark:text-white">Project Premise:</strong> This application utilizes an advanced XGBoost Machine Learning model (R² Score: 71%) to estimate rent in sixteen(16) different locations in the Lagos State, Nigeria. The underlying data was gathered via web scraping from <em>PropertyProNG</em> and <em>Jiji</em>.
          </p>
          <p className="text-brand-accent dark:text-red-400 font-semibold uppercase">
            Disclaimer: This platform is developed purely for educational and academic purposes. The market data is static and based on data collected by May 2026. It should not be used as an active, deployed financial tool by the general public at this stage.
          </p>
        </div>
      </div>
    </footer>
  );
}