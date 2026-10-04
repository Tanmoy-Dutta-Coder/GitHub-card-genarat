import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add current folder to sys.path to guarantee local imports work
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)

# Load environment variables
load_dotenv(os.path.join(root_dir, ".env"), override=True)
load_dotenv(override=True)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

from agent import github_card_agent, mcp_toolset

# Ensure the static directories exist
static_dir = os.path.join(base_dir, "static")
cards_dir = os.path.join(static_dir, "cards")
os.makedirs(cards_dir, exist_ok=True)

# Initialize Session and Memory Services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Create ADK Runner
runner = Runner(
    agent=github_card_agent,
    session_service=session_service,
    memory_service=memory_service,
    app_name="github_dev_card_app",
    auto_create_session=True
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("FastAPI Application starting up...")
    yield
    # Shutdown actions
    print("FastAPI Application shutting down. Cleaning up toolset connections...")
    try:
        await mcp_toolset.close()
    except Exception as e:
        print(f"Error during shutdown toolset cleanup: {e}", file=sys.stderr)

app = FastAPI(title="GitHub Dev Card Generator Backend", lifespan=lifespan)

# CORS middleware config to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CardRequest(BaseModel):
    username: str

@app.post("/generate")
@app.post("/api/generate")
async def generate_card(request: CardRequest):
    username = request.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    
    # Construct types.Content for ADK Runner
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"Generate a dev card for {username}")]
    )
    
    # 4. Stream events and execute agent
    used_fallback = False
    try:
        print(f"Driving agent execution for: {username}")
        async for event in runner.run_async(
            user_id="default_user",
            session_id=username,
            new_message=message
        ):
            print(f"[ADK Event] ID: {event.id} | Turn Complete: {event.turn_complete}")
            if event.error_message:
                print(f"[ADK Error] {event.error_message}")
                raise HTTPException(status_code=500, detail=event.error_message)
    except Exception as e:
        print(f"Error during ADK agent execution: {e}. Falling back to direct tool execution.")
        # Fallback to direct tool execution (handles missing GEMINI_API_KEY gracefully)
        try:
            from mcp_server import scrape_github, analyze_profile, generate_card_html, save_card
            github_data = scrape_github(username)
            analysis = analyze_profile(github_data)
            html_content = generate_card_html(username, github_data, analysis)
            save_card(username, html_content)
            used_fallback = True
        except Exception as fallback_err:
            raise HTTPException(
                status_code=500,
                detail=f"Agent failed: {str(e)}. Fallback failed: {str(fallback_err)}"
            )

    # 5. Serve saved cards
    file_path = os.path.join(cards_dir, f"{username.lower()}.html")
    if not os.path.exists(file_path):
        file_path = os.path.join(cards_dir, f"{username}.html")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=500,
            detail=f"Card file for {username} was not generated or saved."
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read generated card file: {read_err}"
        )

    return {
        "success": True,
        "card_url": f"/card/{username}",
        "html": html_content,
        "direct_run": used_fallback
    }

# 5. Serve saved cards via GET endpoint
@app.get("/card/{username}", response_class=HTMLResponse)
async def get_card(username: str):
    file_path = os.path.join(cards_dir, f"{username.lower()}.html")
    if not os.path.exists(file_path):
        file_path = os.path.join(cards_dir, f"{username}.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Card not found")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading card: {e}")

# 6. Health check endpoint for Cloud Run
@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # 8. Start with port 8080 as requested
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
