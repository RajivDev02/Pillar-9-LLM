# Property Remarks Generator API

A microservice that generates professional property descriptions using AI.

## Prerequisites

- Python 3.8+
- Ollama running locally with the TinyLlama model installed

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
```bash
# Follow instructions at https://ollama.ai to install Ollama
ollama pull tinyllama
```

## Running the API

Start the API server:
```bash
python api.py
```

The API will be available at:
- API Root: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST /api/generate

Generate property descriptions based on provided data and style preferences.

Request body:
```json
{
    "property_data": {
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
    },
    "style_preferences": {
        "tone": "Professional",
        "theme": "Modern Living"
    }
}
```

Response:
```json
{
    "status": "success",
    "property_id": "PROP_1234567890",
    "descriptions": [
        "Description 1...",
        "Description 2...",
        "Description 3..."
    ],
    "generation_time": "2.34 seconds"
}
```

## Testing with Postman

1. Open Postman and create a new request
2. Set the request type to POST
3. Enter the URL: http://localhost:8000/api/generate
4. Set Headers:
   - Content-Type: application/json
5. In the Body tab:
   - Select "raw"
   - Select "JSON" from the dropdown
   - Paste the example request body from above
6. Click "Send" to test the API

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (invalid input)
- 500: Internal Server Error
- 503: Service Unavailable (Ollama not running)
- 504: Gateway Timeout (Ollama request timeout)

## Development

To modify the API:
1. Edit `api.py`
2. Restart the server
3. Test changes using Postman or Swagger UI

## Project Structure
```
property-remarks-generator/
├── .venv/           # Virtual environment
├── api.py           # Main API implementation
├── requirements.txt # Dependencies
└── README.md        # Documentation
```

## Troubleshooting

1. **Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama if needed
ollama restart
```

2. **Model Not Found**
```bash
# Pull the model again
ollama pull tinyllama
```

3. **Port Already in Use**
```bash
# Kill the process using port 8000
# Windows
netstat -ano | findstr 8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

## License
MIT License

## Support
For issues and feature requests, please create an issue in the repository.

{
    "address": "Tech Park, Mohali",
    "price": 1800000,
    "square_footage": 3000,
    "property_type": "Commercial",
    "units": 4,
    "parking_spots": 10,
    "annual_revenue": 180000,
    "tax": 15000,
    "zoning": "Mixed-use",
    "features": [
        "Corner location",
        "High foot traffic",
        "Modern infrastructure",
        "Fiber optic connectivity",
        "24/7 power backup"
    ],
    "tenant_info": {
        "current_occupancy": "80%",
        "lease_terms": "3-5 years"
    }
}


