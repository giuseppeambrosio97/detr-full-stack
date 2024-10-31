import logging

from backend import ASSETS_DATA_LOCATION, IMAGES_DATA_LOCATION
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routers import health, inference
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DATA_LOCATION), name="images")
app.mount("/assets", StaticFiles(directory=ASSETS_DATA_LOCATION), name="assets")


app.include_router(health.router)
app.include_router(inference.router)
