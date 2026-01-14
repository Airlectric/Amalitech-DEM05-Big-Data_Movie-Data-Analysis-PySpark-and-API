import requests
from utils.config import (
    API_KEY, BASE_URL, DEFAULT_LANGUAGE,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF,
)
from utils.logger_config import logger
import time

def fetch_with_retry(url, params):
    """
    GET request handler with retry logic, timeout,
    logging, and API error handling
    """

    for attempt in range(MAX_RETRIES+1):
        try:
            response = requests.get(url, params, timeout=REQUEST_TIMEOUT)

            # this raises an error for HTTP 400 and above
            response.raise_for_status()

            data = response.json()

            # TMBD returns status code 34 if the resourse is not found 
            if data.get('status_code') == 34:
                logger.warning(f"Movie not found: {params.get('movie_id')}")
                return {'success': False, 'error': 'Movie not found'}

            # This will return the result if successfil
            return {'success': True, 'data': data}

        except requests.exceptions.Timeout:
            logger.error(f'Timeout on attempt {attempt} for URL: {url}')

        except requests.exceptions.HTTPError as e:
            logger.error(f'HTTP error on attempt {attempt}: {str(e)}')

        except requests.exceptions.RequestException as e:
            logger.error(f'Request error on attempt {attempt}: {str(e)}')

        # Retrying for 5 times with a backof
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)

    logger.error(f'Failed after {MAX_RETRIES} attempts: {url}')
    return {'success': False, 'error': 'Max retries reached.'}


def get_movie_details(movie_id):
    """Fetches detailed information about a specific movie by its ID."""
    url = f"{BASE_URL}/movie/{movie_id}?append_to_response=credits"
    params = {"api_key": API_KEY, "language": DEFAULT_LANGUAGE, 'movie_id': movie_id}
    
    # Using the fetch_with_retry function to get the data
    result = fetch_with_retry(url, params)

    # Logging error if fetch failed
    if not result['success']:
        logger.error(f"Failed to fetch movies {movie_id}: {result['error']}")

    return result

def get_all_movies_by_ids(movie_ids):
    """Fetches detailed information for multiple movies
       by their IDs."""
    movies = []

    for movie_id in movie_ids:
        logger.info(f'Fetching movie ID: {movie_id}')

        result = get_movie_details(movie_id)

        # including only valid data
        if result['success']:
            movies.append(result['data'])
        else:
            logger.warning(f"Skipping movie ID {movie_id} due to fetch error: {result.get('error')}")

    return movies
