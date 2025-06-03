import streamlit as st
import json
import requests
from typing import Dict, List
import time
import pyperclip
import random
import ollama
from fastapi import FastAPI
import uvicorn
from threading import Thread
import concurrent.futures
import asyncio

app = FastAPI(
    title="Property Remarks Generator API",
    description="API for generating AI-powered property descriptions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class PropertyRemarksGenerator:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = "tinyllama"
    
    def generate_single_description(self, property_data: Dict, style_preferences: Dict, focus: str, attempt: int) -> str:
        """Generate a single property description"""
        try:
            prompt = self._create_prompt(property_data, style_preferences, attempt=attempt, focus=focus)
            
            # Clear model state
            requests.post(
                self.api_url,
                json={
                    "model": "tinyllama",
                    "prompt": "clear",
                    "stream": False
                },
                timeout=1
            )
            
            response = requests.post(
                self.api_url,
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 150,  # Reduced from 200
                        "context_length": 80,  # Reduced from 100
                        "stop": ["###"],
                        "num_thread": 2,
                        "num_batch": 1,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=10  # Reduced from 15
            )
            
            if response.status_code == 200:
                content = response.json()
                if 'response' in content and content['response']:
                    description = content['response'].strip()
                    description = self._clean_description(description)
                    if self._verify_description_completeness(description, property_data):
                        return description
            
            # If generation fails, return default description
            return self._generate_default_descriptions(property_data, style_preferences)[attempt]
            
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return self._generate_default_descriptions(property_data, style_preferences)[attempt]

    def generate_remarks(self, property_data: Dict, style_preferences: Dict) -> List[str]:
        """Generate property descriptions using TinyLlama"""
        try:
            # Validate input data
            if not isinstance(property_data, dict) or not isinstance(style_preferences, dict):
                return ["Error: Invalid input data type"]
            
            if not property_data or not style_preferences:
                return ["Error: Empty input data"]
            
            # Ensure required fields are present
            required_fields = ['address', 'price', 'square_footage', 'bedrooms', 'bathrooms']
            missing_fields = [field for field in required_fields if field not in property_data or property_data.get(field) is None]
            if missing_fields:
                return [f"Error: Missing required fields: {', '.join(missing_fields)}"]
                 
            # Validate field types
            try:
                property_data['price'] = float(property_data['price'])
                property_data['square_footage'] = float(property_data['square_footage'])
                property_data['bedrooms'] = float(property_data['bedrooms'])
                property_data['bathrooms'] = float(property_data['bathrooms'])
            except (ValueError, TypeError):
                return ["Error: Invalid numeric values in property data"]
            
            # Different focus points for each description
            focus_points = [
                "luxury and lifestyle",
                "modern technology and comfort",
                "location and investment value"
            ]
            
            # Generate descriptions in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        self.generate_single_description,
                        property_data,
                        style_preferences,
                        focus,
                        i
                    )
                    for i, focus in enumerate(focus_points)
                ]
                
                # Get results as they complete
                descriptions = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            return descriptions
            
        except Exception as e:
            print(f"Error in generate_remarks: {str(e)}")
            return [f"Error: {str(e)}"]

    def _clean_description(self, description: str) -> str:
        """Clean up the description formatting"""
        # Remove extra newlines and spaces
        description = ' '.join(description.split())
        
        # Remove any meta-instructions or notes
        description = description.replace("Base Description:", "").replace("Enhancing the Property Description:", "")
        
        # Format paragraphs properly
        paragraphs = description.split(". ")
        formatted_paragraphs = []
        for p in paragraphs:
            if p.strip():
                formatted_paragraphs.append(p.strip() + ".")
        
        # Join paragraphs with proper spacing
        return "\n\n".join(formatted_paragraphs)

    def _create_prompt(self, property_data: Dict, style_preferences: Dict, attempt: int = 0, focus: str = "") -> str:
        # Determine if it's a commercial property
        is_commercial = property_data.get('property_type', '').lower() == 'commercial'
        
        # Context starters for commercial properties
        commercial_starters = [
            "Discover an exceptional investment opportunity at",
            "Presenting a prime commercial space at",
            "Capitalize on this outstanding commercial property at"
        ]
        
        # Context starters for residential properties
        residential_starters = [
            "Experience the pinnacle of",
            "Welcome to an exceptional",
            "Discover a masterpiece of",
            "Immerse yourself in"
        ]
        
        # Select different starters for each attempt
        if is_commercial:
            starter = commercial_starters[attempt % len(commercial_starters)]
        else:
            starter = residential_starters[attempt % len(residential_starters)]
        
        # Create the appropriate prompt based on property type
        if is_commercial:
            prompt = f"""Generate a commercial property description:
            Starter: {starter}
            Location: {property_data.get('address', '')}
            Tone: {style_preferences.get('tone')}
            Theme: {style_preferences.get('theme')}
            Focus: {focus}
            
            Include: Size: {property_data.get('square_footage', 0):,} sq ft, Price: ${property_data.get('price', 0):,}, Units: {property_data.get('units', 0)}, Parking: {property_data.get('parking_spots', 0)}, Revenue: ${property_data.get('annual_revenue', 0):,}, Tax: ${property_data.get('tax', 0):,}, Zoning: {property_data.get('zoning', 'N/A')}
            Features: {', '.join(property_data.get('features', []))}
            Tenant Info: Occupancy: {property_data.get('tenant_info', {}).get('current_occupancy', 'N/A')}, Lease: {property_data.get('tenant_info', {}).get('lease_terms', 'N/A')}
            
            Guidelines: 200-250 words, professional tone, focus on {focus}, include ALL details."""
        else:
            # Property features formatting
            features_list = property_data.get('features', [])
            amenities = property_data.get('amenities', {})
            location_features = property_data.get('location_features', {})
            
            prompt = f"""Generate a property description:
            Starter: {starter}
            Location: {property_data.get('address', '')}
            Tone: {style_preferences.get('tone')}
            Theme: {style_preferences.get('theme')}
            Focus: {focus}
            
            Include: Price: ${property_data.get('price', 0):,}, Size: {property_data.get('square_footage', 0):,} sq ft, Beds: {property_data.get('bedrooms', 0)}, Baths: {property_data.get('bathrooms', 0)}, Year: {property_data.get('year_built', 'N/A')}, Lot: {property_data.get('lot_size', 'N/A')}, Tax: ${property_data.get('tax', 0):,}
            Features: {', '.join(features_list)}
            Amenities: {', '.join([f'{k}: {v}' for k, v in amenities.items()])}
            Location: {', '.join([f'{k}: {v}' for k, v in location_features.items()])}
            
            Guidelines: 200-250 words, {style_preferences.get('tone')} tone, focus on {focus}, include ALL details."""
        
        return prompt

    def _verify_description_completeness(self, description: str, property_data: Dict) -> bool:
        """Verify that all required property details are included in the description"""
        try:
            # Check if description is None or empty
            if not description or not isinstance(description, str):
                return False
                
            required_elements = []
            
            # Add basic property details with null checks
            basic_details = [
                ('address', 'address'),
                ('price', 'price'),
                ('square_footage', 'square_footage'),
                ('bedrooms', 'bedrooms'),
                ('bathrooms', 'bathrooms'),
                ('year_built', 'year_built'),
                ('lot_size', 'lot_size'),
                ('tax', 'tax')
            ]
            
            for key, value in basic_details:
                val = property_data.get(key)
                if val is not None:
                    try:
                        required_elements.append(str(val))
                    except Exception:
                        continue
            
            # Add features with null check
            features = property_data.get('features', [])
            if features and isinstance(features, list):
                for feature in features:
                    if feature and isinstance(feature, (str, int, float)):
                        required_elements.append(str(feature))
            
            # Add amenities with null check
            amenities = property_data.get('amenities', {})
            if amenities and isinstance(amenities, dict):
                for k, v in amenities.items():
                    if v is not None and isinstance(v, (str, int, float)):
                        required_elements.append(str(v))
            
            # Add location features with null check
            location_features = property_data.get('location_features', {})
            if location_features and isinstance(location_features, dict):
                for k, v in location_features.items():
                    if v is not None and isinstance(v, (str, int, float)):
                        required_elements.append(str(v))
            
            # Check if all elements are present in the description
            description_lower = description.lower()
            for element in required_elements:
                if element and isinstance(element, str) and element.lower() not in description_lower:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error in _verify_description_completeness: {str(e)}")
            return False

    def _generate_default_descriptions(self, property_data: Dict, style_preferences: Dict) -> List[str]:
        """Generate fallback descriptions based on tone and theme"""
        price = f"${property_data.get('price'):,}"
        beds = property_data.get('bedrooms')
        baths = property_data.get('bathrooms')
        sqft = property_data.get('square_footage')
        features = property_data.get('features', [])
        address = property_data.get('address')
        tone = style_preferences.get('tone')
        theme = style_preferences.get('theme')
        year_built = property_data.get('year_built')
        lot_size = property_data.get('lot_size')
        tax = f"${property_data.get('tax'):,}"
        amenities = property_data.get('amenities', {})
        location_features = property_data.get('location_features', {})
        
        # Format amenities and location features more naturally
        amenities_list = []
        for k, v in amenities.items():
            if k == 'pool':
                amenities_list.append(f"an {v.lower()}")
            elif k == 'gym':
                amenities_list.append(f"a {v.lower()}")
            elif k == 'parking':
                amenities_list.append(f"a {v.lower()}")
            elif k == 'security':
                amenities_list.append(f"{v.lower()}")
        
        amenities_str = ', '.join(amenities_list[:-1]) + f", and {amenities_list[-1]}" if len(amenities_list) > 1 else amenities_list[0]
        
        location_list = []
        for k, v in location_features.items():
            if k == 'school_distance':
                location_list.append(f"an international school {v.replace('to international school', 'away')}")
            elif k == 'shopping':
                location_list.append(f"a shopping mall {v.replace('to mall', 'away')}")
            elif k == 'transport':
                location_list.append(f"a {v.lower()}")
            elif k == 'medical':
                location_list.append(f"a hospital {v.replace('Hospital ', '')}")
        
        location_str = ', '.join(location_list[:-1]) + f", and {location_list[-1]}" if len(location_list) > 1 else location_list[0]
        
        return [
            f"Experience the pinnacle of {tone.lower()} living at {address}. This extraordinary {sqft} sq ft residence, built in {year_built} on a generous {lot_size}, showcases the epitome of modern design. The {beds} thoughtfully designed bedrooms and {baths} luxurious bathrooms offer the perfect balance of style and functionality. Step into the spacious rooms, where soaring ceilings and abundant natural light create an atmosphere of unrestricted elegance. The property's crowning glory is its breathtaking valley view, visible from the expansive windows that frame the landscape like living artwork. Priced at {price}, this home represents the perfect fusion of location, luxury, and lifestyle. The modern design elements extend to every corner, from the sophisticated fixtures to the premium finishes that define each space. Smart home technology and energy-efficient systems ensure modern convenience and sustainability. The property offers {amenities_str}. The prime location provides exceptional convenience with {location_str}, making this a truly exceptional investment with an annual tax of {tax}. Whether you're entertaining in the generous living areas or enjoying quiet moments gazing at the valley vista, this residence offers an unparalleled living experience.",
            
            f"Welcome to an architectural masterpiece in {address}, where {theme.lower()} takes center stage. This remarkable {sqft} sq ft home, built in {year_built} on a spacious {lot_size}, features {beds} generously proportioned bedrooms and {baths} elegantly appointed bathrooms, each space thoughtfully crafted to maximize comfort and style. The modern design philosophy is evident throughout, from the cutting-edge kitchen to the sophisticated living spaces. Floor-to-ceiling windows showcase the spectacular valley view, creating a seamless connection between indoor luxury and natural beauty. At {price}, this property offers an exceptional opportunity to embrace a lifestyle of distinction. The spacious rooms provide versatile living spaces that adapt to your needs, while premium finishes and contemporary touches elevate every interaction. Smart home technology and energy-efficient systems ensure modern convenience and sustainability. The property features {amenities_str}, while the prime location offers {location_str}, making this a truly exceptional investment with an annual tax of {tax}. This isn't just a home; it's a statement of success and refined taste.",
            
            f"Discover unparalleled luxury at {address}, where every detail has been meticulously curated for the discerning buyer. This prestigious {sqft} sq ft residence, built in {year_built} on a generous {lot_size}, harmoniously combines modern design principles with practical luxury. The heart of this {price} masterpiece lies in its {beds} expansive bedrooms and {baths} designer bathrooms, each space reflecting a commitment to excellence. The spacious rooms flow seamlessly from one to another, creating an atmosphere of sophisticated comfort. Perhaps the most captivating feature is the stunning valley view, a natural masterpiece that serves as a constant backdrop to your daily life. The modern architectural elements are complemented by high-end finishes and premium materials throughout. Smart home technology and energy-efficient systems ensure modern convenience and sustainability. The property offers {amenities_str}, while the prime location provides {location_str}, making this a truly exceptional investment with an annual tax of {tax}. This exceptional property offers more than just a home – it's an opportunity to elevate your lifestyle in an environment where luxury, comfort, and natural beauty converge."
        ]

@app.post("/api/generate")
async def generate_descriptions_api(property_data: dict, style_preferences: dict):
    try:
        generator = PropertyRemarksGenerator()
        descriptions = generator.generate_remarks(property_data, style_preferences)
        return {
            "status": "success",
            "descriptions": descriptions
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
async def root():
    return {"message": "Welcome to Property Remarks Generator API"}

def run_api():
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_api()
  