import json
import os
from typing import Dict, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from gpt_researcher import GPTResearcher  # Add this import at the top with other imports
from httpx import AsyncClient  # Add this import at the top

from backend.server.websocket_manager import WebSocketManager
from backend.server.server_utils import (
    get_config_dict,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication
)
from backend.chat.chat import ChatAgentWithMemory

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str
    record_id: str


class ConfigRequest(BaseModel):
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    DOC_PATH: str
    RETRIEVER: str
    GOOGLE_API_KEY: str = ''
    GOOGLE_CX_KEY: str = ''
    BING_API_KEY: str = ''
    SEARCHAPI_API_KEY: str = ''
    SERPAPI_API_KEY: str = ''
    SERPER_API_KEY: str = ''
    SEARX_URL: str = ''


# App initialization
app = FastAPI()

# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# WebSocket manager
manager = WebSocketManager()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://hook.eu1.make.com/qdzrmrc77ggt1psu88gyi4rqk7pkleal")

# Startup event


@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    os.makedirs(DOC_PATH, exist_ok=True)

# Routes


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/files/")
async def list_files():
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        async with ChatAgentWithMemory(report="", config_path="", headers={}) as agent:
            await handle_websocket_communication(websocket, manager, agent)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    finally:
        if 'agent' in locals():
            await agent.cleanup()

async def process_research_in_background(request: ResearchRequest):
    try:
        researcher = GPTResearcher(
            query=request.task,
            report_type=request.report_type
        )
        
        await researcher.conduct_research()
        report = await researcher.write_report()
        
        webhook_payload = {
            "status": "success",
            "record_id": request.record_id,
            "report": report,
            "sources": researcher.get_source_urls(),
            "costs": researcher.get_costs(),
            "original_task": request.task,
            "report_type": request.report_type
        }
        
        async with AsyncClient() as client:
            webhook_response = await client.post(
                WEBHOOK_URL,
                json=webhook_payload,
                timeout=30.0
            )
            webhook_response.raise_for_status()
            
    except Exception as e:
        error_payload = {
            "status": "error",
            "message": str(e),
            "record_id": request.record_id,
            "original_task": request.task
        }
        
        try:
            async with AsyncClient() as client:
                await client.post(WEBHOOK_URL, json=error_payload)
        except Exception:
            pass

@app.post("/api/research")
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    # Add the research task to background tasks
    background_tasks.add_task(process_research_in_background, request)
    
    # Return immediately with acceptance message
    return {
        "status": "accepted", 
        "message": "Research task started. Results will be sent to webhook.",
        "record_id": request.record_id
    }