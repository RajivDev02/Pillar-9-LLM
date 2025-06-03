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

# Add this near the top of the file, after imports
app = FastAPI(
    title="Property Remarks Generator API",
    description="API for generating AI-powered property descriptions",
    version="1.0.0",
    docs_url="/docs",  # Explicitly set the Swagger UI path
    redoc_url="/redoc"  # Explicitly set the ReDoc path
)

class PropertyRemarksGenerator:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = "tinyllama"
    
    def generate_remarks(self, property_data: Dict, style_preferences: Dict) -> List[str]:
        """Generate property descriptions using TinyLlama"""
        prompt = self._create_prompt(property_data, style_preferences)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_gpu": 0,
                        "num_thread": 4,
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "max_tokens": 1024,
                        "context_length": 512,
                        "batch_size": 1
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                content = response.json()
                if 'response' in content:
                    descriptions = self._parse_descriptions(
                        content['response'],
                        property_data,
                        style_preferences
                    )
                    if descriptions == ["No valid descriptions generated"]:
                        return self._generate_default_descriptions(property_data, style_preferences)
                    return descriptions
                else:
                    print(f"Ollama API returned unexpected response format: {content}")
                    return ["Error: Ollama API unexpected response format."]
            else:
                print(f"Ollama API returned status code {response.status_code}: {response.text}")
                return [f"Error: Ollama API returned status code {response.status_code}: {response.text}"]
            
        except requests.exceptions.ConnectionError as e:
            print(f"Error: Could not connect to Ollama API. Make sure Ollama is running. Details: {e}")
            return ["Error: Could not connect to Ollama API."]
        except requests.exceptions.Timeout as e:
            print(f"Error: Ollama API request timed out after 60 seconds. Details: {e}")
            return ["Error: Ollama API request timed out."]
        except Exception as e:
            print(f"An unexpected error occurred during API call: {str(e)}")
            return [f"Error: An unexpected error occurred: {str(e)}"]

    def _create_prompt(self, property_data: Dict, style_preferences: Dict) -> str:
        # Determine if it's a commercial property
        is_commercial = property_data.get('property_type', '').lower() == 'commercial'
        
        # Context starters for commercial properties
        commercial_starters = [
            "Discover an exceptional investment opportunity at",
            "Presenting a prime commercial space at",
            "Capitalize on this outstanding commercial property at",
            "Seize this premium business opportunity at",
            "Welcome to a distinguished commercial establishment at"
        ]
        
        # Create the appropriate prompt based on property type
        if is_commercial:
            prompt = f"""
            Generate a compelling commercial property description with the following requirements:
            
            1. Use this context starter: {random.choice(commercial_starters)}
            2. Location: {property_data.get('address', '')}
            3. Tone: {style_preferences.get('tone')}
            4. Theme: {style_preferences.get('theme')}
            
            Essential Details to Include:
            - Property Size: {property_data.get('square_footage', 0):,} sq ft
            - Investment Value: ${property_data.get('price', 0):,}
            - Number of Units: {property_data.get('units', 0)} units
            - Parking Capacity: {property_data.get('parking_spots', 0)} parking spots
            - Annual Revenue: ${property_data.get('annual_revenue', 0):,}
            - Property Tax: ${property_data.get('tax', 0):,}
            - Zoning: {property_data.get('zoning', 'N/A')}
            
            Key Features to Highlight:
            {', '.join(property_data.get('features', []))}
            
            Tenant Information:
            - Current Occupancy: {property_data.get('tenant_info', {}).get('current_occupancy', 'N/A')}
            - Lease Terms: {property_data.get('tenant_info', {}).get('lease_terms', 'N/A')}
            
            Guidelines:
            1. Start with the location and property overview
            2. Discuss the investment potential and annual revenue
            3. Highlight ALL infrastructure features (power backup, connectivity, etc.)
            4. Mention occupancy and lease terms
            5. Emphasize parking and accessibility
            6. Include zoning and commercial advantages
            7. End with a strong investment appeal
            8. Keep the description between 200-250 words
            9. Use professional, investment-focused language
            10. Ensure EVERY feature and detail is incorporated
            """
        else:
            # Context starters for variety
            context_starters = [
                "Experience the pinnacle of",
                "Welcome to an exceptional",
                "Discover a masterpiece of",
                "Immerse yourself in",
                "Presenting an extraordinary",
                "Step into the elegance of",
                "Embrace the sophistication of",
                "Unveil the grandeur of"
            ]
            
            # Property features formatting
            features_list = property_data.get('features', [])
            amenities = property_data.get('amenities', {})
            location_features = property_data.get('location_features', {})
            
            # Combine all features for comprehensive description
            all_features = {
                'Property Details': {
                    'Price': f"${property_data.get('price', 0):,}",
                    'Square Footage': f"{property_data.get('square_footage', 0):,} sq ft",
                    'Bedrooms': property_data.get('bedrooms', 'N/A'),
                    'Bathrooms': property_data.get('bathrooms', 'N/A'),
                    'Year Built': property_data.get('year_built', 'N/A'),
                    'Lot Size': property_data.get('lot_size', 'N/A'),
                    'Tax': f"${property_data.get('tax', 0):,}"
                },
                'Special Features': features_list,
                'Amenities': amenities,
                'Location Highlights': location_features
            }

            prompt = f"""
            Generate a compelling property description with the following requirements:
            
            1. Use this context starter format (but vary it): {random.choice(context_starters)}
            2. Property Location: {property_data.get('address', '')}
            3. Tone: {style_preferences.get('tone')}
            4. Theme: {style_preferences.get('theme')}
            
            CRITICAL: You MUST include ALL of these details in your description:
            
            Property Details:
            - Price: ${property_data.get('price', 0):,}
            - Square Footage: {property_data.get('square_footage', 0):,} sq ft
            - Bedrooms: {property_data.get('bedrooms', 0)}
            - Bathrooms: {property_data.get('bathrooms', 0)}
            - Year Built: {property_data.get('year_built', 'N/A')}
            - Lot Size: {property_data.get('lot_size', 'N/A')}
            - Annual Tax: ${property_data.get('tax', 0):,}
            
            Special Features (MUST include ALL):
            {', '.join(features_list)}
            
            Amenities (MUST include ALL):
            {', '.join([f'{k}: {v}' for k, v in amenities.items()])}
            
            Location Benefits (MUST include ALL):
            {', '.join([f'{k}: {v}' for k, v in location_features.items()])}
            
            Guidelines:
            1. Create a flowing narrative that incorporates ALL provided features
            2. Emphasize lifestyle benefits and emotional appeal
            3. Use sophisticated language appropriate for {style_preferences.get('tone')} tone
            4. Focus on the {style_preferences.get('theme')} aspects
            5. Keep the description between 200-250 words
            6. Avoid generic phrases and ensure specific property details are mentioned
            7. Include a compelling closing statement about the property's unique value
            8. CRITICAL: Every single detail listed above MUST be included in the description
            9. Format the description with proper paragraphs and flow
            10. Ensure the description reads naturally while including all details
            """
        
        return prompt

    def _parse_descriptions(self, text: str, property_data: Dict, style_preferences: Dict) -> List[str]:
        """Parse the model's response into separate descriptions"""
        if '###' not in text:
            return self._generate_default_descriptions(property_data, style_preferences)
        
        parts = text.split('###')
        descriptions = []
        
        for part in parts[1:]:  # Skip everything before first ###
            desc = part.strip()
            # Skip any placeholders or short descriptions
            if (desc and 
                len(desc) > 100 and  # Ensure minimum length
                not desc.startswith('[') and 
                not desc.endswith(']')):
                    # Verify all required details are included
                    if self._verify_description_completeness(desc, property_data):
                        descriptions.append(desc)
                    else:
                        # If any details are missing, use default description
                        descriptions.append(self._generate_default_descriptions(property_data, style_preferences)[0])
        
        # If we don't get exactly 3 valid descriptions, use defaults
        if len(descriptions) != 3:
            return self._generate_default_descriptions(property_data, style_preferences)
        
        return descriptions

    def _verify_description_completeness(self, description: str, property_data: Dict) -> bool:
        """Verify that all required property details are included in the description"""
        required_elements = [
            str(property_data.get('address')),
            str(property_data.get('price')),
            str(property_data.get('square_footage')),
            str(property_data.get('bedrooms')),
            str(property_data.get('bathrooms')),
            str(property_data.get('year_built')),
            str(property_data.get('lot_size')),
            str(property_data.get('tax'))
        ]
        
        # Add features
        required_elements.extend(property_data.get('features', []))
        
        # Add amenities
        for k, v in property_data.get('amenities', {}).items():
            required_elements.append(str(v))
        
        # Add location features
        for k, v in property_data.get('location_features', {}).items():
            required_elements.append(str(v))
        
        # Check if all elements are present in the description
        description_lower = description.lower()
        for element in required_elements:
            if element.lower() not in description_lower:
                return False
        
        return True

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

def main():
    st.set_page_config(page_title="Property Descriptions Generator", layout="wide")
    
    st.title("🏠 Property Descriptions Generator")
    st.markdown("Generate property descriptions using AI")

    # First, verify model availability
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            st.error("Cannot connect to Ollama server")
            st.info("Please make sure Ollama is running and TinyLlama model is installed")
            st.code("ollama pull tinyllama")
            return
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return

    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Property Details")
        example_json = {
            "address": "Tech Park, Mohali",
            "price": 1800000,
            "bedrooms": 4,
            "bathrooms": 3,
            "square_footage": 3000,
            "year_built": 2023,
            "lot_size": "0.25 acre",
            "tax": 12000,
            "features": [
                "Modern design",
                "Valley view",
                "High ceilings",
                "Smart home technology",
                "Energy efficient"
            ],
            "amenities": {
                "pool": "Infinity pool",
                "gym": "24/7 fitness center",
                "parking": "2 car garage",
                "security": "24/7 surveillance"
            },
            "location_features": {
                "school_distance": "0.5 miles to international school",
                "shopping": "5 minutes to mall",
                "transport": "Metro station nearby",
                "medical": "Hospital within 2 km"
            }
        }
        
        # Show example JSON
        with st.expander("Show Example Format"):
            st.code(json.dumps(example_json, indent=2), language="json")
        
        json_input = st.text_area(
            "Enter Property Details (JSON format)",
            value=json.dumps(example_json, indent=2),
            height=300
        )
    
    with col2:
        st.subheader("Description Style")
        tone = st.selectbox(
            "Select Tone",
            ["Professional", "Luxury", "Casual", "Modern", "Traditional"]
        )
        
        theme = st.selectbox(
            "Select Theme",
            ["Modern Living", "Luxury Lifestyle", "Family-Focused", "Investment", "Urban Living"]
        )

    # Initialize session state for descriptions if not exists
    if 'descriptions' not in st.session_state:
        st.session_state.descriptions = []

    # Generate button
    if st.button("Generate Descriptions", type="primary"):
        start_time = time.time()  # Start timing
        try:
            property_data = json.loads(json_input)
            property_name = property_data.get('address', 'Property')  # Get property name
            
            with st.spinner("Generating descriptions..."):
                generator = PropertyRemarksGenerator()
                st.session_state.descriptions = generator.generate_remarks(
                    property_data,
                    {"tone": tone, "theme": theme}
                )
            
            # Calculate time taken
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            # Display title and timing
            st.header(f"🏠 Remarks for {property_name}")
            st.info(f"⏱️ Generation completed in {time_taken} seconds")
            
            # Display descriptions
            display_descriptions(st.session_state.descriptions, property_data)
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your input.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def display_descriptions(descriptions, property_data):
    for i, desc in enumerate(descriptions, 1):
        with st.expander(f"Description {i}", expanded=True):
            edited_desc = st.text_area(
                f"Edit Description {i}",
                value=desc,
                key=f"description_{i}",
                height=150
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Copy Description {i}", key=f"copy_{i}"):
                    pyperclip.copy(edited_desc)
                    st.success("Copied to clipboard!")
            with col2:
                if st.button(f"Save Changes {i}", key=f"save_{i}"):
                    st.session_state[f"description_{i}"] = edited_desc
                    st.success("Changes saved!")
            with col3:
                if st.button(f"Clear Messages {i}", key=f"clear_{i}"):
                    st.session_state[f"description_{i}"] = desc
                    st.rerun()

def format_property_details(property_data):
    """Organize all input fields into structured categories"""
    details = {
        'basic_info': {
            'address': property_data.get('address'),
            'price': property_data.get('price'),
            'square_footage': property_data.get('square_footage'),
            'bedrooms': property_data.get('bedrooms'),
            'bathrooms': property_data.get('bathrooms'),
            'year_built': property_data.get('year_built'),
            'lot_size': property_data.get('lot_size'),
            'tax': property_data.get('tax'),
            'property_type': property_data.get('property_type', 'residential')
        },
        'features': property_data.get('features', []),
        'amenities': property_data.get('amenities', {}),
        'location_features': property_data.get('location_features', {}),
        'commercial_info': {
            'units': property_data.get('units'),
            'parking_spots': property_data.get('parking_spots'),
            'annual_revenue': property_data.get('annual_revenue'),
            'zoning': property_data.get('zoning'),
            'tenant_info': property_data.get('tenant_info', {})
        }
    }
    return details

def create_field_requirements(details):
    """Create mandatory coverage requirements for all available fields"""
    requirements = []
    
    # Basic Info Requirements
    for key, value in details['basic_info'].items():
        if value:
            requirements.append(f"- Must mention {key}: {value}")
    
    # Features Requirements
    if details['features']:
        requirements.append("- Must incorporate ALL features:")
        for feature in details['features']:
            requirements.append(f"  * {feature}")
    
    # Amenities Requirements
    if details['amenities']:
        requirements.append("- Must describe ALL amenities:")
        for k, v in details['amenities'].items():
            requirements.append(f"  * {k}: {v}")
    
    # Location Features Requirements
    if details['location_features']:
        requirements.append("- Must highlight ALL location benefits:")
        for k, v in details['location_features'].items():
            requirements.append(f"  * {k}: {v}")
    
    # Commercial Info Requirements (if applicable)
    if details['basic_info']['property_type'] == 'commercial':
        for key, value in details['commercial_info'].items():
            if value:
                requirements.append(f"- Must include {key}: {value}")
    
    return "\n".join(requirements)

def create_prompt(property_data, tone, theme):
    """Enhanced prompt to ensure comprehensive descriptions"""
    details = format_property_details(property_data)
    
    prompt = f"""
    Create a luxury property description that seamlessly incorporates ALL of these elements into a natural, flowing narrative:

    Essential Details (weave ALL into the description):
    - Location: {details['basic_info']['address']}
    - Price: ${details['basic_info']['price']:,}
    - Living Space: {details['basic_info']['square_footage']} sq ft
    - Configuration: {details['basic_info']['bedrooms']} bedrooms, {details['basic_info']['bathrooms']} bathrooms
    - Construction: Built in {details['basic_info']['year_built']}
    - Property Size: {details['basic_info']['lot_size']}
    - Annual Tax: ${details['basic_info']['tax']:,}

    Features to Include:
    {', '.join(details['features'])}

    Amenities to Highlight:
    {', '.join([f'{k}: {v}' for k, v in details['amenities'].items()])}

    Location Benefits:
    {', '.join([f'{k}: {v}' for k, v in details['location_features'].items()])}

    Writing Requirements:
    1. Create a cohesive narrative that naturally incorporates ALL elements
    2. Maintain a {tone} tone and {theme} theme throughout
    3. Structure the content in this order:
       - Opening with location and primary features
       - Interior details and smart features
       - Amenities and security features
       - Location benefits and accessibility
       - Closing with investment value
    4. Length: 200-250 words
    5. Use sophisticated language while maintaining clarity
    6. Ensure every feature flows naturally in the narrative

    Remember: Every single detail provided must be woven into the description naturally.
    """
    
    return prompt

def verify_description_completeness(description, property_data):
    """Verify that all input fields are mentioned in the description"""
    details = format_property_details(property_data)
    missing_elements = []
    
    # Check each field
    for category, items in details.items():
        if isinstance(items, dict):
            for key, value in items.items():
                if value and str(value).lower() not in description.lower():
                    missing_elements.append(f"{key}: {value}")
        elif isinstance(items, list):
            for item in items:
                if item.lower() not in description.lower():
                    missing_elements.append(item)
    
    return missing_elements

def setup_llm():
    try:
        # Initialize Ollama client
        llm = ollama.Client()
        return lambda prompt: ollama.generate(model='tinyllama', prompt=prompt)['response']
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return None

# At the start of your script, initialize the LLM
llm = setup_llm()

def create_structured_description(property_data):
    """Create a structured description template ensuring all elements are covered"""
    
    # Format currency and numbers
    price = f"${property_data.get('price', 0):,}"
    tax = f"${property_data.get('tax', 0):,}"
    sqft = f"{property_data.get('square_footage', 0):,}"
    
    description_template = f"""
    Experience the pinnacle of luxury living at {property_data.get('address')}. This extraordinary {sqft} sq ft residence, built in {property_data.get('year_built')} on a pristine {property_data.get('lot_size')}, showcases the epitome of modern design. The {property_data.get('bedrooms')} thoughtfully designed bedrooms and {property_data.get('bathrooms')} luxurious bathrooms offer the perfect balance of style and functionality.

    Step into a world of sophisticated technology and comfort. {', '.join(property_data.get('features', []))}. The property's modern design elements extend to every corner, creating an atmosphere of unrestricted elegance. Priced at {price}, with an annual tax of {tax}, this investment represents the perfect fusion of luxury and practical value.

    Premium amenities elevate daily living: {', '.join([f'{k} featuring {v}' for k, v in property_data.get('amenities', {}).items()])}

    The location offers unparalleled convenience: {', '.join([f'{v}' for k, v in property_data.get('location_features', {}).items()])}

    Whether you're entertaining in the generous living areas or enjoying quiet moments, this residence offers an unparalleled living experience that combines luxury, technology, and convenience in one of Mohali's most sought-after locations.
    """
    return description_template

def generate_property_descriptions(property_data, tone, theme):
    """Generate descriptions using template and enhance with LLM"""
    descriptions = []
    base_template = create_structured_description(property_data)
    
    for i in range(3):
        prompt = f"""
        Enhance and rephrase this property description while maintaining ALL details. 
        CRITICAL: EVERY single feature and detail MUST be included in the final output.

        Base Description:
        {base_template}

        Requirements:
        1. Maintain {tone} tone and {theme} theme
        2. Keep ALL property details, prices, and features
        3. Rephrase while ensuring NO information is lost
        4. Create natural flow between elements
        5. Use sophisticated language
        6. Maintain professional real estate style
        
        Property Details to Verify (ALL must be included):
        - Address: {property_data.get('address')}
        - Price: ${property_data.get('price'):,}
        - Square Footage: {property_data.get('square_footage')} sq ft
        - Bedrooms: {property_data.get('bedrooms')}
        - Bathrooms: {property_data.get('bathrooms')}
        - Year Built: {property_data.get('year_built')}
        - Lot Size: {property_data.get('lot_size')}
        - Annual Tax: ${property_data.get('tax'):,}
        
        Features: {', '.join(property_data.get('features', []))}
        
        Amenities: {', '.join([f'{k}: {v}' for k, v in property_data.get('amenities', {}).items()])}
        
        Location Features: {', '.join([f'{k}: {v}' for k, v in property_data.get('location_features', {}).items()])}

        Generate a unique version that includes EVERY detail listed above.
        """
        
        try:
            # First generate using template
            base_description = base_template
            
            # Then enhance with LLM while maintaining all details
            enhanced_description = llm(prompt)
            
            # Verify all elements are present
            missing_elements = verify_description_completeness(enhanced_description, property_data)
            
            # If anything is missing, fall back to template version
            if missing_elements:
                descriptions.append(base_description)
            else:
                descriptions.append(enhanced_description)
                
        except Exception as e:
            st.error(f"Error in generation: {str(e)}")
            descriptions.append(base_description)
        
        time.sleep(1)
    
    return descriptions

def verify_description_completeness(description, property_data):
    """Strict verification of all elements"""
    required_elements = [
        str(property_data.get('address')),
        str(property_data.get('price')),
        str(property_data.get('square_footage')),
        str(property_data.get('bedrooms')),
        str(property_data.get('bathrooms')),
        str(property_data.get('year_built')),
        str(property_data.get('lot_size')),
        str(property_data.get('tax'))
    ]
    
    # Add features
    required_elements.extend(property_data.get('features', []))
    
    # Add amenities
    for k, v in property_data.get('amenities', {}).items():
        required_elements.append(str(v))
    
    # Add location features
    for k, v in property_data.get('location_features', {}).items():
        required_elements.append(str(v))
    
    missing = []
    for element in required_elements:
        if element.lower() not in description.lower():
            missing.append(element)
    
    return missing

# Add these API endpoints before the run_api() function
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
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    import asyncio
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start API server in a separate thread
    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Run Streamlit app
    main() 