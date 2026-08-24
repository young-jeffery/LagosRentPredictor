// src/constants/appConstants.js

// [MODIFIED] Old flat 16-district LOCATIONS array replaced. Now
// re-exporting the full 335-entry sub-location dictionary from its own
// file, since it's generated data and shouldn't live inline here.
// [ADDED] Re-exporting the derived districts list alongside the sub-locations
export { LAGOS_LOCATIONS, LAGOS_DISTRICTS } from './lagosLocations';

// [ADDED] v3 -- the exact 12 categories property_type_classifier.py
// produces. Must match schemas.py's PROPERTY_TYPES list exactly, since
// this string gets sent straight to the backend and looked up by name.
export const PROPERTY_TYPES = [
  "Flat/Apartment",
  "Mini flat",
  "Self-contain",
  "Studio",
  "Duplex",
  "Terrace",
  "Semi-detached",
  "Detached",
  "Bungalow",
  "Block of flats",
  "Maisonette",
  "Penthouse",
];

export const FEATURES_LIST = [
  {
    id: "Is_Serviced",
    label: "Fully Serviced",
    description:
      "Includes facility management like cleaning, security, and waste disposal.",
  },
  {
    id: "Is_Furnished",
    label: "Fully Furnished",
    description: "Property comes with furniture and essential appliances.",
  },
  {
    id: "has_prepaid_meter",
    label: "Prepaid Meter",
    description:
      "Independent electricity meter for self-managed power billing.",
  },
  {
    id: "has_screeded_walls",
    label: "Screeded Walls",
    description: "Smooth, evenly plastered walls for a premium paint finish.",
  },
  {
    id: "has_fitted_kitchen",
    label: "Fitted Kitchen",
    description:
      "Kitchen equipped with built-in cabinets and basic appliances.",
  },
  {
    id: "has_clean_water",
    label: "Clean Water",
    description: "Reliable access to treated, clean running water.",
  },
  {
    id: "has_steady_power",
    label: "Steady Power",
    description:
      "Guaranteed 24/7 electricity supply, often backed by generators.",
  },
  {
    id: "has_ac",
    label: "Air Conditioning",
    description: "Air conditioning units are pre-installed.",
  },
  {
    id: "has_water_heater",
    label: "Water Heater",
    description: "Hot water heaters installed in bathrooms.",
  },
  {
    id: "has_internet",
    label: "Internet/WiFi",
    description: "Dedicated internet or WiFi setup available.",
  },
  {
    id: "is_ensuite",
    label: "All Rooms Ensuite",
    description: "Every bedroom has its own attached private bathroom.",
  },
  {
    id: "has_bq",
    label: "Boys Quarters (BQ)",
    description:
      "Includes a Boys Quarters (BQ) for staff or guest accommodation.",
  },
  {
    id: "has_pop_ceiling",
    label: "POP Ceiling",
    description:
      "Modern Plaster of Paris (POP) ceiling with recessed lighting.",
  },
  {
    id: "has_wardrobe",
    label: "Wardrobe",
    description: "Built-in wardrobes in the bedrooms.",
  },
  {
    id: "has_balcony",
    label: "Balcony/Terrace",
    description: "Includes a private balcony or terrace area.",
  },
  {
    id: "is_secured",
    label: "Secured/Guards",
    description: "Property has dedicated security guards or personnel.",
  },
  {
    id: "in_gated_estate",
    label: "Gated Estate",
    description: "Located within a secure, gated community or estate.",
  },
  {
    id: "has_cctv",
    label: "CCTV",
    description: "Equipped with CCTV surveillance cameras for security.",
  },
  {
    id: "has_pool",
    label: "Swimming Pool",
    description: "Access to a swimming pool on the premises.",
  },
  {
    id: "has_gym",
    label: "Gym/Fitness",
    description: "Access to a fitness center or gym on the premises.",
  },
  {
    id: "has_smart_home",
    label: "Smart Home",
    description: "Features smart home automation (lighting, security, etc.).",
  },
  {
    id: "has_elevator",
    label: "Elevator",
    description: "Building includes an elevator/lift.",
  },
  {
    id: "has_parking",
    label: "Parking Space",
    description: "Dedicated off-street parking space available.",
  },
  {
    id: "is_tarred",
    label: "Tarred Road",
    description: "Access roads to the property are tarred and well-maintained.",
  },
  {
    id: "has_easy_access",
    label: "Easy Access/Bus Stop",
    description: "Close proximity to major roads or bus stops.",
  },
  {
    id: "prefers_working_class",
    label: "Working Class Preferred",
    description: "Landlord strictly prefers renting to working professionals.",
  },
  // [ADDED] New amenities found via keyword_discovery.py
  {
    id: "has_jacuzzi",
    label: "Jacuzzi",
    description: "Property includes a jacuzzi.",
  },
  {
    id: "is_premium_marketed",
    label: "Premium/Luxury Listing",
    description: "Listing is marketed using premium or luxury language.",
  },
  {
    id: "is_urgent_listing",
    label: "Urgent/First-Come-First-Served",
    description: "Listing is marketed as urgent or fastest-finger-first.",
  },
  {
    id: "is_standalone_fenced",
    label: "Standalone Fenced Compound",
    description: "A fenced standalone compound, as opposed to a gated estate.",
  },
];