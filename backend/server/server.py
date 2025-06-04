import json
import os
from typing import Dict, List
import time
import httpx

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.server.websocket_manager import WebSocketManager
from backend.server.server_utils import (
    get_config_dict, sanitize_filename,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication
)

from backend.server.websocket_manager import run_agent
from backend.utils import write_md_to_word, write_md_to_pdf
from gpt_researcher.utils.logging_config import setup_research_logging
from gpt_researcher.utils.enum import Tone
from backend.chat.chat import ChatAgentWithMemory

import logging

# Get logger instance
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    report_source: str
    tone: str
    headers: dict | None = None
    repo_name: str
    branch_name: str
    generate_in_background: bool = True
    webhook_url: str | None = None


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
    XAI_API_KEY: str
    DEEPSEEK_API_KEY: str


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

# Startup event


@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    # os.makedirs(DOC_PATH, exist_ok=True)  # Commented out to avoid creating the folder if not needed
    

# Routes


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/report/{research_id}")
async def read_report(request: Request, research_id: str):
    docx_path = os.path.join('outputs', f"{research_id}.docx")
    if not os.path.exists(docx_path):
        return {"message": "Report not found."}
    return FileResponse(docx_path)


async def write_report(research_request: ResearchRequest, research_id: str = None):
    report_information = await run_agent(
        task=research_request.task,
        report_type=research_request.report_type,
        report_source=research_request.report_source,
        source_urls=[],
        document_urls=[],
        tone=Tone[research_request.tone],
        websocket=None,
        stream_output=None,
        headers=research_request.headers,
        query_domains=[],
        config_path="",
        return_researcher=True
    )

    actual_markdown_report = ""
    researcher_object_for_details = None

    # Determine the structure of report_information and extract markdown
    if research_request.report_type == "multi_agents":
        if isinstance(report_information, str):
            actual_markdown_report = report_information
        elif isinstance(report_information, dict) and "report" in report_information: # Based on multi_agents.main.run_research_task
            actual_markdown_report = report_information.get("report", "")
        else:
            logger.error(f"Unexpected structure for multi_agents report_information: {type(report_information)}")
            actual_markdown_report = "Error: Could not extract multi_agents report content."
    elif isinstance(report_information, tuple) and len(report_information) == 2:
        actual_markdown_report = report_information[0]
        researcher_object_for_details = report_information[1]
        if not isinstance(actual_markdown_report, str): # Ensure it's a string
             logger.error(f"Report content is not a string: {type(actual_markdown_report)}")
             actual_markdown_report = "Error: Report content is not in expected string format."
    else:
        logger.error(f"Unexpected structure for report_information: {type(report_information)}")
        actual_markdown_report = "Error: Could not extract report content."


    # Send to webhook if URL is provided and markdown content is available and not an error message
    if research_request.webhook_url and actual_markdown_report and not actual_markdown_report.startswith("Error:"):
        try:
            webhook_payload = {
                "research_id": research_id,
                "task": research_request.task,
                "report_type": research_request.report_type,
                "markdown_report": actual_markdown_report
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(research_request.webhook_url, json=webhook_payload, timeout=10.0)
                response.raise_for_status()
                logger.info(f"Successfully sent report to webhook: {research_request.webhook_url} for research_id: {research_id}")
        except httpx.RequestError as e:
            logger.error(f"HTTP error sending report to webhook {research_request.webhook_url} for research_id {research_id}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred when sending report to webhook {research_request.webhook_url} for research_id {research_id}: {e}")

    # Continue with existing logic for DOCX, PDF, and constructing the main API response
    docx_path = ""
    pdf_path = ""

    if actual_markdown_report and not actual_markdown_report.startswith("Error:"):
        # The first argument to write_md_to_word and write_md_to_pdf should be the markdown string
        docx_path = await write_md_to_word(actual_markdown_report, research_id)
        pdf_path = await write_md_to_pdf(actual_markdown_report, research_id)

    response_payload = {}
    if research_request.report_type != "multi_agents" and researcher_object_for_details:
        response_payload = {
            "research_id": research_id,
            "research_information": {
                "source_urls": researcher_object_for_details.get_source_urls(),
                "research_costs": researcher_object_for_details.get_costs(),
                "visited_urls": list(researcher_object_for_details.visited_urls),
                "research_images": researcher_object_for_details.get_research_images(),
            },
            "report": actual_markdown_report, # Return the actual markdown
            "docx_path": docx_path,
            "pdf_path": pdf_path
        }
    else: # Covers multi_agents or cases where researcher_object_for_details might be None
          # For multi_agents, the main 'report' field in the API response was previously empty,
          # now it will contain the markdown report if successfully extracted.
        response_payload = {
            "research_id": research_id,
            "report": actual_markdown_report,
            "docx_path": docx_path,
            "pdf_path": pdf_path
        }
    
    return response_payload

@app.post("/report/")
async def generate_report(research_request: ResearchRequest, background_tasks: BackgroundTasks):
    research_id = sanitize_filename(f"task_{int(time.time())}_{research_request.task}")

    if research_request.generate_in_background:
        background_tasks.add_task(write_report, research_request=research_request, research_id=research_id)
        return {"message": "Your report is being generated in the background. Please check back later.",
                "research_id": research_id}
    else:
        response = await write_report(research_request, research_id)
        return response


@app.get("/files/")
async def list_files():
    if not os.path.exists(DOC_PATH):
        os.makedirs(DOC_PATH, exist_ok=True)
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
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
