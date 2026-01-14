import os
from dotenv import load_dotenv

# Loaded environment variables from a .env
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
DEFAULT_LANGUAGE = "en-US"

if not API_KEY:
    raise ValueError("TMDB_API_KEY is not set in the environment variables.")


# -----------------------------------------------
# Request parameters
# ------------------------------------------------
REQUEST_TIMEOUT = 10        # seconds
MAX_RETRIES = 5           # number of retry attempts
RETRY_BACKOFF = 2           # seconds (wait time between retries)