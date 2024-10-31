import pathlib as pl
import logging.config

__version__ = "1.0.0"

PACKAGE_LOCATION = pl.Path(__file__).parent.resolve()
ROOT_LOCATION = PACKAGE_LOCATION.parent
RESOURCES_LOCATION = PACKAGE_LOCATION / "resources"
CONFIG_LOCATION = RESOURCES_LOCATION / "config"
LOGGING_CONF_LOCATION = CONFIG_LOCATION / "logging.conf"

DATA_LOCATION = ROOT_LOCATION / "data"
IMAGES_DATA_LOCATION = DATA_LOCATION / "images"
ASSETS_DATA_LOCATION = DATA_LOCATION / "assets"

logging.config.fileConfig(LOGGING_CONF_LOCATION)