"""
FastAPI Application - Enterprise Marketing AI Agents API.

Production-ready REST API for multi-agent marketing system.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import logging
from typing import Dict, Any

from config.settings import get_settings
from api.dependencies import get_orchestrator, get_memory_manager, get_message_bus
from api.routes import agents, workflows, health, prompts, chat
from src.marketing_agents.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.project_name} v{settings.version}")
    logger.info(f"Environment: {settings.system.environment}")

    # Initialize core components
    try:
        orchestrator = get_orchestrator()
        logger.info(f"Orchestrator initialized with {len(orchestrator.agents)} agents")

        memory_manager = get_memory_manager()
        if memory_manager:
            logger.info("Memory manager initialized")
        else:
            logger.warning("Memory manager not available")

        message_bus = get_message_bus()
        logger.info("Message bus initialized")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        raise

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")
    # Clean up resources if needed


# Initialize FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# CORS Middleware
origins = settings.security.allowed_origins
if isinstance(origins, str):
    origins = [o.strip() for o in origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{origin}" for origin in origins]
    + [f"https://{origin}" for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests and their processing time.
    """
    start_time = time.time()

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        },
    )

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - Duration: {duration:.3f}s",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration": duration,
        },
    )

    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors with detailed messages.
    """
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    logger.warning(f"Validation error: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request parameters",
            "details": errors,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.
    """
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.system.debug else None,
        },
    )


# Include routers with API v1 prefix
app.include_router(chat.router, prefix="/api/v1")  # NEW: Unified chat endpoint
app.include_router(health.router, prefix="/api/v1")
app.include_router(agents.router, prefix="/api/v1")
app.include_router(workflows.router, prefix="/api/v1")
app.include_router(prompts.router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": settings.project_name,
        "version": settings.version,
        "description": settings.description,
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }


# API v1 root
@app.get("/api/v1")
async def api_v1_root():
    """
    API v1 root endpoint with available routes.
    """
    return {
        "version": "v1",
        "endpoints": {
            "agents": "/api/v1/agents",
            "workflows": "/api/v1/workflows",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn for development
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
