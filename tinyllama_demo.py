from fastapi import FastAPI, HTTPException
import uvicorn
import requests
from typing import Dict
from pydantic import BaseModel
import socket
import sys

app = FastAPI(
    title="Pilla9 LLM Q&A Demo",
    description="Demonstration of LLM capabilities and why it was chosen for the project",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class TinyLlamaDemo:
    def __init__(self):
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = "tinyllama"
        
        # Why TinyLlama was chosen
        self.model_advantages = {
            "efficiency": "Lightweight and fast inference",
            "quality": "Good balance of quality and performance",
            "cost": "Free and open-source",
            "deployment": "Easy to deploy locally",
            "customization": "Highly customizable parameters"
        }
        
        # Model configuration
        self.model_config = {
            "temperature": 0.7,
            "max_tokens": 150,
            "context_length": 80,
            "num_thread": 2,
            "num_batch": 1,
            "repeat_penalty": 1.1
        }

        # Known facts database for validation
        self.known_facts = {
            "virat_kohli": {
                "ipl_team": "Royal Challengers Bangalore (RCB)",
                "role": "Batsman",
                "nationality": "Indian",
                "format": "Right-handed"
            }
        }

    def check_ollama_connection(self) -> Dict:
        """Check if LLM is running and accessible"""
        try:
            # First check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                # Then check if our specific model is loaded
                model_status = self.check_model_status()
                return {
                    "status": "connected",
                    "message": "LLM is running and accessible",
                    "available_models": response.json().get("models", []),
                    "model_status": model_status
                }
            else:
                return {
                    "status": "error",
                    "message": f"LLM is running but returned status code: {response.status_code}"
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "disconnected",
                "message": "LLM is not running or not accessible. Please start Ollama with 'ollama serve'"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking LLM connection: {str(e)}"
            }

    def check_model_status(self) -> Dict:
        """Check if the specific model is loaded and ready"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {"max_tokens": 1}
                },
                timeout=5
            )
            if response.status_code == 200:
                return {
                    "status": "loaded",
                    "message": f"Model {self.model_name} is loaded and ready"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Model {self.model_name} is not properly loaded"
                }
        except requests.exceptions.Timeout:
            return {
                "status": "loading",
                "message": f"Model {self.model_name} is still loading. Please wait a few minutes."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking model status: {str(e)}"
            }

    def validate_response(self, question: str, answer: str) -> Dict:
        """Validate the response against known facts"""
        validation_result = {
            "is_valid": True,
            "corrections": [],
            "confidence": "high"
        }

        # Check for Virat Kohli related information
        if "virat kohli" in question.lower():
            if "kkr" in answer.lower() or "kolkata" in answer.lower():
                validation_result["is_valid"] = False
                validation_result["corrections"].append({
                    "incorrect": "KKR/Kolkata Knight Riders",
                    "correct": "Royal Challengers Bangalore (RCB)",
                    "context": "IPL team"
                })
                validation_result["confidence"] = "high"

        return validation_result

    def ask_question(self, question: str) -> Dict:
        """Ask a question to TinyLlama"""
        # First check if model is ready
        model_status = self.check_model_status()
        if model_status["status"] != "loaded":
            return {
                "status": "error",
                "message": f"Cannot process question: {model_status['message']}"
            }

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": question,
                    "stream": False,
                    "options": self.model_config
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                
                # Validate the response
                validation = self.validate_response(question, answer)
                
                if not validation["is_valid"]:
                    # Apply corrections
                    for correction in validation["corrections"]:
                        answer = answer.replace(
                            correction["incorrect"],
                            correction["correct"]
                        )
                
                return {
                    "status": "success",
                    "question": question,
                    "answer": answer,
                    "validation": validation,
                    "model_info": {
                        "name": self.model_name,
                        "parameters": self.model_config
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to get response: {response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "message": "Request timed out. The model might be still loading or the request is too complex. Please try again in a few minutes."
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }

    def get_model_info(self) -> Dict:
        """Get information about why LLM was chosen"""
        connection_status = self.check_ollama_connection()
        return {
            "model_name": self.model_name,
            "advantages": self.model_advantages,
            "configuration": self.model_config,
            "use_case": "Property description generation",
            "implementation": {
                "api_integration": "Using Ollama API",
                "parallel_processing": "Concurrent description generation",
                "error_handling": "Comprehensive error management",
                "response_formatting": "Structured output formatting"
            },
            "connection_status": connection_status
        }

def find_available_port(start_port: int = 8001, max_port: int = 8999) -> int:
    """Find an available port to use"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

# Initialize demo
demo = TinyLlamaDemo()

@app.get("/model-info")
async def get_model_info():
    """Get information about TinyLlama and why it was chosen"""
    return demo.get_model_info()

@app.post("/ask")
async def ask(question: QuestionRequest):
    """Ask a question to LLM"""
    return demo.ask_question(question.question)

def run_demo():
    """Run the demo server"""
    try:
        # Find an available port
        port = find_available_port()
        print(f"Starting server on port {port}")
        
        # Run the server
        uvicorn.run(app, host="127.0.0.1", port=port)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_demo()
#hi this is a test comment to see if it works