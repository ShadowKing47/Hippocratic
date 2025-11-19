from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import logging
from main import generate_and_refine_story

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bedtime Story Generator API",
    description="API for generating AI-powered bedtime stories using Gemini",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class StoryRequest(BaseModel):
    text: str

# Response model structure (for documentation)
class StoryResponse(BaseModel):
    story: str
    judge: Dict[str, Any]
    trading_cards: Dict[str, Any]
    soundtrack: Dict[str, Any]
    mood: str
    themes: list
    setting: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/generate", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    """
    Generate a bedtime story based on user input
    
    Args:
        request: StoryRequest containing the user's story prompt
        
    Returns:
        StoryResponse containing the generated story and metadata
    """
    try:
        logger.info(f"Received story generation request: {request.text[:100]}...")
        
        # Call the story generation function from main.py
        story, judge_json, cards_json, soundtrack_json, mood, themes, setting = generate_and_refine_story(request.text)
        
        # Prepare response
        response = {
            "story": story,
            "judge": judge_json,
            "trading_cards": cards_json,
            "soundtrack": soundtrack_json,
            "mood": mood,
            "themes": themes,
            "setting": setting
        }
        
        logger.info("Story generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to generate story: {str(e)}"
            }
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bedtime Story Generator API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)