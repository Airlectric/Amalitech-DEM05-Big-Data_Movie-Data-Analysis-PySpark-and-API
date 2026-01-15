# File: tmdb_data_preprocessor.py

import logging
from typing import List, Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)

def normalize_tmdb_movie(movie: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes and flattens a single raw TMDB movie dictionary.
    Makes nested fields much easier to handle in Spark.
    
    Returns a flatter dictionary with extracted useful fields.
    """
    if not movie:
        return {}

    normalized = deepcopy(movie)  # avoid mutating original

    # 1. belongs_to_collection → extract name if exists
    if isinstance(movie.get("belongs_to_collection"), dict):
        normalized["belongs_to_collection_name"] = movie["belongs_to_collection"].get("name")
        normalized["belongs_to_collection_id"] = movie["belongs_to_collection"].get("id")
        # Optional: keep poster/backdrop if useful
    else:
        normalized["belongs_to_collection_name"] = None

    # 2. genres → list of names
    if isinstance(movie.get("genres"), list):
        normalized["genres_names"] = "|".join(
            g.get("name", "") for g in movie["genres"] if g.get("name")
        )
        normalized["genres_ids"] = "|".join(
            str(g.get("id", "")) for g in movie["genres"] if g.get("id") is not None
        )
    else:
        normalized["genres_names"] = ""
        normalized["genres_ids"] = ""

    # 3. production_countries → names joined
    if isinstance(movie.get("production_countries"), list):
        normalized["production_countries_names"] = "|".join(
            c.get("name", "") for c in movie["production_countries"] if c.get("name")
        )
        normalized["production_countries_codes"] = "|".join(
            c.get("iso_3166_1", "") for c in movie["production_countries"] if c.get("iso_3166_1")
        )
    else:
        normalized["production_countries_names"] = ""
        normalized["production_countries_codes"] = ""

    # 4. production_companies → names joined
    if isinstance(movie.get("production_companies"), list):
        normalized["production_companies_names"] = "|".join(
            c.get("name", "") for c in movie["production_companies"] if c.get("name")
        )
    else:
        normalized["production_companies_names"] = ""

    # 5. spoken_languages → english_name or name
    if isinstance(movie.get("spoken_languages"), list):
        normalized["spoken_languages_names"] = "|".join(
            lang.get("english_name") or lang.get("name", "") 
            for lang in movie["spoken_languages"] if lang.get("english_name") or lang.get("name")
        )
    else:
        normalized["spoken_languages_names"] = ""

    # 6. credits → flatten cast & crew (most important part!)
    credits = movie.get("credits", {})
    
    # Cast
    cast_list = credits.get("cast", [])
    if isinstance(cast_list, list):
        normalized["cast_names"] = "|".join(
            member.get("name", "") for member in cast_list if member.get("name")
        )
        normalized["cast_size"] = len([m for m in cast_list if m.get("name")])
    else:
        normalized["cast_names"] = ""
        normalized["cast_size"] = 0

    # Directors (may be multiple)
    crew_list = credits.get("crew", [])
    directors = []
    if isinstance(crew_list, list):
        directors = [
            member.get("name", "")
            for member in crew_list
            if member.get("job") == "Director" and member.get("name")
        ]
    normalized["directors"] = "|".join(directors)
    normalized["crew_size"] = len(crew_list) if isinstance(crew_list, list) else 0

    # 7. origin_country (simple array of strings)
    if isinstance(movie.get("origin_country"), list):
        normalized["origin_country_codes"] = "|".join(movie["origin_country"])
    else:
        normalized["origin_country_codes"] = ""

    # 8. Keep other useful top-level fields (optional cleanup)
    for field in ["budget", "revenue", "runtime", "vote_average", "vote_count", "popularity"]:
        if field in normalized and normalized[field] == 0:
            normalized[field] = None  # consistent with later cleaning

    # Remove the original nested fields to reduce size & confusion
    nested_fields_to_remove = [
        "belongs_to_collection", "genres", "production_countries",
        "production_companies", "spoken_languages", "credits"
    ]
    for field in nested_fields_to_remove:
        normalized.pop(field, None)

    return normalized


def preprocess_tmdb_movies(raw_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of raw TMDB movie dictionaries.
    Returns a list ready to be passed to spark.createDataFrame()
    """
    logger.info(f"Preprocessing {len(raw_movies)} raw TMDB movies...")
    
    processed = []
    for movie in raw_movies:
        try:
            norm_movie = normalize_tmdb_movie(movie)
            if norm_movie.get("id"):  # at least have an id
                processed.append(norm_movie)
        except Exception as e:
            logger.error(f"Error processing movie {movie.get('id', 'unknown')}: {e}")
            continue

    logger.info(f"Successfully preprocessed {len(processed)} movies")
    return processed