from pyspark.sql import functions as F
from pyspark.sql.types import *
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Drop irrelevant columns
# =============================================================================
def drop_irrelevant_columns(movies_df, columns_to_drop):
    """Drops irrelevant columns from the movie data."""
    logger.info(f"Dropping columns: {columns_to_drop}")
    existing = [c for c in columns_to_drop if c in movies_df.columns]
    if existing:
        movies_df = movies_df.drop(*existing)
    return movies_df


# =============================================================================
# 2. Extract field from nested array<struct> or struct
# =============================================================================
def extract_json_field(df, col, key, join_with="|"):
    """
    Extract field from:
      - struct → .key
      - array<struct> → join of all .key values
    """
    logger.info(f"Extracting field '{key}' from column '{col}'")

    if col not in df.columns:
        return df

    dtype = df.schema[col].dataType

    # Single struct (e.g. belongs_to_collection)
    if isinstance(dtype, StructType):
        return df.withColumn(col, F.col(col).getField(key))

    # Array of structs (genres, production_companies, spoken_languages, etc.)
    if isinstance(dtype, ArrayType) and isinstance(dtype.elementType, StructType):
        return df.withColumn(
            col,
            F.when(
                F.size(F.col(col)) > 0,
                F.concat_ws(join_with, F.transform(F.col(col), lambda x: F.coalesce(x.getField(key), F.lit(""))))
            ).otherwise(F.lit(None))
        )

    # Fallback
    logger.warning(f"Unsupported type for extraction: {dtype}")
    return df.withColumn(col, F.lit(None))


# =============================================================================
# 3. Special handling for credits (cast + director + sizes)
# =============================================================================
def extract_credit_json_fields(df, col='credits', join_with="|"):
    """Extract cast names, director(s), cast_size, crew_size from credits"""
    logger.info(f"Extracting cast & crew from column '{col}'")

    if col not in df.columns:
        return df

    # Cast names
    df = df.withColumn(
        "cast",
        F.when(
            F.size(F.col(f"{col}.cast")) > 0,
            F.concat_ws(
                join_with,
                F.transform(
                    F.col(f"{col}.cast"),
                    lambda x: F.coalesce(x.getField("name"), F.lit(""))
                )
            )
        ).otherwise(F.lit(None))
    )

    # Cast size
    df = df.withColumn("cast_size", F.size(F.coalesce(F.col(f"{col}.cast"), F.array())))

    # Director(s) - support multiple directors
    df = df.withColumn(
        "director",
        F.when(
            F.size(F.col(f"{col}.crew")) > 0,
            F.concat_ws(
                join_with,
                F.transform(
                    F.filter(
                        F.col(f"{col}.crew"),
                        lambda x: x.getField("job") == "Director"
                    ),
                    lambda x: F.coalesce(x.getField("name"), F.lit(""))
                )
            )
        ).otherwise(F.lit(None))
    )

    # Crew size
    df = df.withColumn("crew_size", F.size(F.coalesce(F.col(f"{col}.crew"), F.array())))

    # Drop original credits
    df = df.drop(col)

    logger.info(f"Finished extracting credits – dropped original '{col}' column")
    return df


# =============================================================================
# 4. Origin / Production countries (already array types)
# =============================================================================
def extract_production_countries(df, col='origin_country', join_with="|"):
    """Join array<string> or array<struct> country fields"""
    logger.info(f"Extracting/joining countries from column '{col}'")

    if col not in df.columns:
        return df

    dtype = df.schema[col].dataType

    if isinstance(dtype, ArrayType):
        if isinstance(dtype.elementType, StringType):
            # origin_country: array<string>
            return df.withColumn(
                col,
                F.when(F.size(F.col(col)) > 0, F.concat_ws(join_with, F.col(col))).otherwise(None)
            )
        elif isinstance(dtype.elementType, StructType):
            # production_countries: array<struct>
            return df.withColumn(
                col,
                F.when(F.size(F.col(col)) > 0,
                       F.concat_ws(join_with, F.transform(F.col(col), lambda x: x.getField("name")))
                ).otherwise(None)
            )

    return df.withColumn(col, F.lit(None))


# =============================================================================
# 5. Inspect value counts (driver action - small data only!)
# =============================================================================
def inspect_categorical_columns_using_value_counts(df, cols):
    """Show value counts (use only on small/medium data!)"""
    logger.info(f"Inspecting value counts for columns: {cols}")
    for c in cols:
        if c in df.columns:
            print(f"\nValue counts for ====== {c} ======")
            df.groupBy(c).count().orderBy(F.desc("count")).show(20, truncate=False)
        else:
            print(f"Column '{c}' not found.\n")


# =============================================================================
# 6. Convert numeric columns
# =============================================================================
def convert_numeric(df, cols):
    """Cast columns to double (safe)"""
    logger.info(f"Converting columns to numeric: {cols}")
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
    return df


# =============================================================================
# 7. Convert to date
# =============================================================================
def convert_to_datetime(df, cols):
    """Convert string dates to date type"""
    logger.info(f"Converting columns to date: {cols}")
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.to_date(F.col(c), "yyyy-MM-dd"))
    return df


# =============================================================================
# 8. Main cleaning pipeline
# =============================================================================
def clean_movie_data(movies_df):
    """Main cleaning pipeline - PySpark version"""
    logger.info("Starting clean_movie_data pipeline")

    # JSON-like extractions
    json_columns = {
        'belongs_to_collection': 'name',
        'genres': 'name',
        'spoken_languages': 'english_name',
        'production_companies': 'name',
    }

    for col, key in json_columns.items():
        if col in movies_df.columns:
            movies_df = extract_json_field(movies_df, col, key)

    # Countries
    if 'origin_country' in movies_df.columns:
        movies_df = extract_production_countries(movies_df, 'origin_country')
    if 'production_countries' in movies_df.columns:
        movies_df = extract_production_countries(movies_df, 'production_countries')

    # Credits
    if 'credits' in movies_df.columns:
        movies_df = extract_credit_json_fields(movies_df, 'credits')

    # Type conversions
    numeric_columns = ['budget', 'popularity', 'id', 'revenue', 'runtime', 'vote_average', 'vote_count']
    movies_df = convert_numeric(movies_df, numeric_columns)

    date_columns = ['release_date']
    movies_df = convert_to_datetime(movies_df, date_columns)

    logger.info("Finished clean_movie_data pipeline")
    return movies_df


# =============================================================================
# 9. Replace unrealistic / placeholder values
# =============================================================================
def replace_zero_values(df):
    """Replace 0 → null in money & runtime"""
    logger.info("Replacing zero values → null in budget/revenue/runtime")
    for c in ['budget', 'revenue', 'runtime']:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c) == 0, None).otherwise(F.col(c)))
    return df


def convert_budget_to_millions(df):
    """Convert budget & revenue to millions USD"""
    logger.info("Converting budget & revenue to millions USD")
    if 'budget' in df.columns:
        df = df.withColumn('budget_musd', F.round(F.col('budget') / 1_000_000.0, 4))
    if 'revenue' in df.columns:
        df = df.withColumn('revenue_musd', F.round(F.col('revenue') / 1_000_000.0, 4))
    df = df.drop(*[c for c in ['budget', 'revenue'] if c in df.columns])
    return df


def clean_text_placeholders(df):
    """Clean placeholder text in tagline & overview"""
    logger.info("Cleaning placeholder text in tagline & overview")
    placeholders = ["no tagline", "no overview", "no data", ""]
    for col in ['tagline', 'overview']:
        if col in df.columns:
            df = df.withColumn(
                col,
                F.when(
                    F.lower(F.trim(F.coalesce(F.col(col), F.lit("")))).isin(placeholders),
                    None
                ).otherwise(F.col(col))
            )
    return df


def adjust_vote_average(df):
    """Set vote_average to null when vote_count = 0"""
    logger.info("Adjusting vote_average when vote_count == 0")
    if all(c in df.columns for c in ['vote_count', 'vote_average']):
        df = df.withColumn(
            'vote_average',
            F.when(F.col('vote_count') == 0, None).otherwise(F.col('vote_average'))
        )
    return df


def replace_unrealistic_values(df):
    """Pipeline for unrealistic value replacements"""
    logger.info("Starting replace_unrealistic_values pipeline")
    df = replace_zero_values(df)
    df = convert_budget_to_millions(df)
    df = clean_text_placeholders(df)
    df = adjust_vote_average(df)
    logger.info("Finished replace_unrealistic_values pipeline")
    return df


# =============================================================================
# 10. Remove duplicates & NA filtering
# =============================================================================
def remove_duplicates(df):
    """Remove duplicate rows (prefer by id)"""
    logger.info("Removing duplicate rows")
    before = df.count()
    if "id" in df.columns:
        df = df.dropDuplicates(["id"])
    else:
        df = df.dropDuplicates()
    after = df.count()
    logger.info(f"After deduplication: {after} rows (removed {before - after})")
    return df


def drop_rows_with_na_in_critical_columns(df, critical_columns):
    """Drop rows with null in critical columns"""
    present = [c for c in critical_columns if c in df.columns]
    if present:
        before = df.count()
        df = df.dropna(subset=present)
        after = df.count()
        logger.info(f"After critical NA drop: {after} rows (removed {before - after})")
    return df


def keep_rows_with_min_non_nan(df, min_non_nan=10):
    """Keep rows with at least N non-null values (expensive - use carefully)"""
    logger.info(f"Filtering rows with >= {min_non_nan} non-null values")
    before = df.count()
    non_null_expr = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in df.columns)
    df = df.withColumn("_non_null_count", non_null_expr) \
           .filter(F.col("_non_null_count") >= min_non_nan) \
           .drop("_non_null_count")
    after = df.count()
    logger.info(f"After min non-null filter: {after} rows (removed {before - after})")
    return df


def filter_released_movies(df):
    """Keep only 'Released' movies"""
    if "status" in df.columns:
        before = df.count()
        df = df.filter(F.col("status") == "Released").drop("status")
        after = df.count()
        logger.info(f"Kept Released movies: {after} (removed {before - after})")
    else:
        logger.info("No 'status' column → skipping released filter")
    return df


def removing_na_and_duplicates(df):
    """Full NA + duplicate removal pipeline"""
    logger.info("Starting removing_na_and_duplicates pipeline")
    df = remove_duplicates(df)
    df = drop_rows_with_na_in_critical_columns(df, ["title", "id"])
    df = keep_rows_with_min_non_nan(df, min_non_nan=10)  # ← comment out if too slow
    df = filter_released_movies(df)
    logger.info(f"Final size after cleaning: {df.count()} rows")
    return df


# =============================================================================
# 11. Finalize (ordering + reset index equivalent)
# =============================================================================
def reorder_columns(df, desired_order):
    """Reorder columns - keep only existing ones"""
    logger.info("Reordering columns")
    existing = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df.select(*(existing + remaining))


def reset_index(df):
    """Spark DataFrames don't have index → no-op or add monotonic id if needed"""
    logger.info("Resetting index (no-op in Spark)")
    return df  # or: .withColumn("row_id", F.monotonically_increasing_id())


def finalize_dataframe(df):
    """Final reordering + logging"""
    logger.info("Finalizing DataFrame")
    desired_order = [
        'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
        'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
        'production_countries', 'vote_count', 'vote_average', 'popularity', 'runtime',
        'overview', 'spoken_languages', 'poster_path', 'cast', 'cast_size', 'director', 'crew_size'
    ]
    df = reorder_columns(df, desired_order)
    df = reset_index(df)
    logger.info(f"DataFrame finalized – rows: {df.count()}, columns: {len(df.columns)}")
    return df