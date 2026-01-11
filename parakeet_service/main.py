from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from .model import lifespan
from .routes import router
from .config import logger

from parakeet_service.stream_routes import router as stream_router

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet-TDT 0.6B v3 STT service",
        version="0.1.0",
        description=(
            "High-accuracy English speech-to-text (FastConformer-TDT) "
            "with optional word/char/segment timestamps."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    # TODO: improve streaming and add support for other audio formats (maybe)
    app.include_router(stream_router)

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui(request: Request):
        """Serve the web UI."""
        return templates.TemplateResponse("index.html", {"request": request})

    logger.info("FastAPI app initialised")
    return app


app = create_app()
