// src/components/layout/Footer.jsx

export default function Footer() {
    return (
        <footer className="w-full py-12 mt-8 border-t border-brand-gray/20 bg-brand-bg/50 dark:bg-[#0b1116]/80 text-center px-4">
            <div className="max-w-4xl mx-auto space-y-4">
                <h3 className="font-rye text-xl sm:text-2xl text-brand-dark mb-2 uppercase tracking-wide">
                    Development of an AI-Powered Web Application for Transparent Residential Rent Valuation in the Lagos Property Market
                </h3>

                <p className="font-bold text-brand-accent text-lg">
                    By: Idowu Mayowa Joseph
                </p>
                <p className="text-brand-dark font-medium text-sm sm:text-base">
                    University of Lagos • Final Year Project
                </p>

                <div className="mt-6 p-4 rounded-lg bg-brand-gray/5 dark:bg-white/5 border border-brand-gray/10 text-brand-dark/80 text-sm sm:text-base leading-relaxed">
                    <p className="mb-2">
                        <strong>Project Premise:</strong> This application utilizes an advanced XGBoost Machine Learning model (R² Score: 71%) to estimate property rents in Lagos. The underlying data was gathered via web scraping from <em>propertypro.ng</em> and <em>Jiji</em>.
                    </p>
                    <p className="text-brand-accent/80 dark:text-red-400 font-semibold italic">
                        Disclaimer: This platform is developed purely for educational and academic purposes. The market data is static (based on data collected up to May 2026) and should not be used as an active, deployed financial tool by the general public.
                    </p>
                </div>
            </div>
        </footer>
    );
}