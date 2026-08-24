// src/constants/propertyTypeConstraints.js
//
// Bedroom/bathroom bounds per property type, derived from actual
// distributions in lagos_rent_spatially_capped_v2.csv (median/p90/max per
// type, extracted the same way feature_engineering_pipeline_v3.py does).
// Studio and Self-contain are ~100% single-room-plus-bath units in the
// real data (151/151 and 18/18 non-missing extractions = 1 bedroom), so
// both are fixed rather than ranged.
//
// [CRITICAL] Must stay in sync with PROPERTY_TYPE_BED_BATH_CONFIG in
// backend/schemas.py -- the backend enforces these same bounds
// server-side (defense in depth against direct API calls bypassing the
// UI), so if one changes, update the other.

export const PROPERTY_TYPE_BED_BATH_CONFIG = {
  "Studio":         { bedMin: 1, bedMax: 1, bedDefault: 1, bathMin: 1, bathMax: 1, bathDefault: 1,
                       note: "Studios are single-room units -- bedrooms and bathrooms are fixed at 1." },
  "Self-contain":   { bedMin: 1, bedMax: 1, bedDefault: 1, bathMin: 1, bathMax: 1, bathDefault: 1,
                       note: "Self-contain units are single-room-plus-bath -- bedrooms and bathrooms are fixed at 1." },
  "Mini flat":      { bedMin: 1, bedMax: 2, bedDefault: 1, bathMin: 1, bathMax: 2, bathDefault: 1 },
  "Flat/Apartment": { bedMin: 1, bedMax: 6, bedDefault: 2, bathMin: 1, bathMax: 6, bathDefault: 2 },
  "Block of flats": { bedMin: 1, bedMax: 4, bedDefault: 2, bathMin: 1, bathMax: 4, bathDefault: 2 },
  "Bungalow":       { bedMin: 1, bedMax: 6, bedDefault: 3, bathMin: 1, bathMax: 6, bathDefault: 3 },
  "Penthouse":      { bedMin: 2, bedMax: 6, bedDefault: 3, bathMin: 2, bathMax: 6, bathDefault: 3,
                       note: "Penthouses are premium multi-room units -- bedrooms start at 2." },
  "Detached":       { bedMin: 2, bedMax: 6, bedDefault: 4, bathMin: 2, bathMax: 6, bathDefault: 4 },
  "Duplex":         { bedMin: 2, bedMax: 7, bedDefault: 4, bathMin: 2, bathMax: 7, bathDefault: 4 },
  "Semi-detached":  { bedMin: 2, bedMax: 6, bedDefault: 4, bathMin: 2, bathMax: 6, bathDefault: 4 },
  "Terrace":        { bedMin: 2, bedMax: 6, bedDefault: 3, bathMin: 2, bathMax: 6, bathDefault: 3 },
  "Maisonette":     { bedMin: 2, bedMax: 4, bedDefault: 3, bathMin: 2, bathMax: 4, bathDefault: 3 },
};

const FALLBACK_CONFIG = { bedMin: 1, bedMax: 6, bedDefault: 2, bathMin: 1, bathMax: 6, bathDefault: 2 };

export const getBedBathConfig = (propertyType) =>
  PROPERTY_TYPE_BED_BATH_CONFIG[propertyType] || FALLBACK_CONFIG;

export const clamp = (value, min, max) => Math.min(Math.max(value, min), max);