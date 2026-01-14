from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, lit, size, concat_ws, explode, collect_list,
    lower, trim, coalesce, isnan
)
from pyspark.sql.types import StringType
import logging

logger = logging.getLogger(__name__)


def drop_irrelevant_columns(movies_df: DataFrame, columns_to_drop: list) -> DataFrame:
    """Drops irrelevant columns from the movie data."""
    logger.info(f"Dropping columns: {columns_to_drop}")
    
    existing_cols = set(movies_df.columns)
    cols_to_drop = [c for c in columns_to_drop if c in existing_cols]
    
    if cols_to_drop:
        return movies_df.drop(*cols_to_drop)
    return movies_df


def extract_json_field(df: DataFrame, col: str, key: str, join_with: str = "|") -> DataFrame:
    """
    Extract values from nested JSON-like structures in a DataFrame column.
    - dict  -> value of key
    - list  -> joined values of key from each item
    - else  -> null
    """
    from pyspark.sql.functions import udf
    
    logger.info(f"Extracting field '{key}' from column '{col}'")
    
    @udf(returnType=StringType())
    def extract_struct(value):
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        if isinstance(value, list):
            values = [item.get(key, "") for item in value if isinstance(item, dict) and key in item]
            if values:
                return join_with.join(values)
        return None
    
    return df.withColumn(col, extract_struct(col(col)))


def extract_credit_json_fields(df: DataFrame, col: str = 'credits', join_with: str = "|") -> DataFrame:
    """Extracts cast and crew information from the credits column."""
    logger.info(f"Extracting cast & crew from column '{col}'")
    
    # Cast names - explode + collect_list + concat_ws
    df = df.withColumn("cast_temp", explode(col(f"{col}.cast").getField("name")))\
           .groupBy([c for c in df.columns if c != col and c != "cast_temp"])\
           .agg(concat_ws(join_with, collect_list("cast_temp")).alias("cast"))
    
    # Sizes + director using native functions + minimal UDF
    from pyspark.sql.functions import udf
    
    @udf(returnType=StringType())
    def get_directors(crew_list):
        if not isinstance(crew_list, list):
            return None
        dirs = [m.get("name", "") for m in crew_list 
                if isinstance(m, dict) and m.get("job") == "Director"]
        return join_with.join(dirs) if dirs else None
    
    df = df.withColumn("cast_size", size(coalesce(col(f"{col}.cast"), lit([]))))\
           .withColumn("director", get_directors(col(f"{col}.crew")))\
           .withColumn("crew_size", size(coalesce(col(f"{col}.crew"), lit([]))))\
           .drop(col)
    
    logger.info(f"Finished extracting credits – dropped original '{col}' column")
    return df


def extract_production_countries(df: DataFrame, col: str = 'origin_country', join_with: str = "|") -> DataFrame:
    """Extracts country codes/names from the production_countries/origin_country column."""
    logger.info(f"Extracting production countries from column '{col}'")
    
    return df.withColumn(
        col,
        concat_ws(join_with, col(col))
    )


def inspect_categorical_columns_using_value_counts(df: DataFrame, cols: list):
    """Prints value counts for specified categorical columns (driver-side operation)."""
    logger.info(f"Inspecting value counts for columns: {cols}")
    
    for c in cols:
        if c in df.columns:
            print(f"Value counts for column: ====== {c} ======")
            df.groupBy(c).count().orderBy("count", ascending=False).show(truncate=False)
            print("\n")
        else:
            print(f"Column '{c}' not found in DataFrame\n")


def convert_numeric(df: DataFrame, cols: list) -> DataFrame:
    """Converts specified columns to numeric types, null on error."""
    logger.info(f"Converting columns to numeric: {cols}")
    
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast("double"))
    return df


def convert_to_datetime(df: DataFrame, cols: list) -> DataFrame:
    """Converts specified columns to date/timestamp, null on error."""
    logger.info(f"Converting columns to datetime: {cols}")
    
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast("date"))  # or "timestamp" if needed
    return df


def clean_movie_data(movies_df: DataFrame) -> DataFrame:
    """
    Cleans and preprocesses the movie data - PySpark version
    
    This function orchestrates all the individual cleaning steps
    using the previously defined Spark-compatible functions.
    """
    logger.info("Starting clean_movie_data pipeline (Spark version)")

    # ─── JSON-like column extractions ───────────────────────────────────────
    json_columns = {
        'belongs_to_collection': 'name',
        'genres': 'name',
        'spoken_languages': 'english_name',
        'production_companies': 'name',
        'production_countries': 'name',     # Note: will be overridden below
    }

    for col_name, key in json_columns.items():
        if col_name in movies_df.columns:
            movies_df = extract_json_field(movies_df, col_name, key)
            logger.debug(f"Extracted '{key}' from column '{col_name}'")

    # Special handling for origin_country/production_countries (list of strings/codes)
    if 'origin_country' in movies_df.columns:
        movies_df = extract_production_countries(movies_df, col='origin_country')
    elif 'production_countries' in movies_df.columns:
        movies_df = extract_production_countries(movies_df, col='production_countries')

    # ─── Credits extraction (cast, director, sizes) ──────────────────────────
    if 'credits' in movies_df.columns:
        movies_df = extract_credit_json_fields(movies_df, col='credits')

    # ─── Type conversions ────────────────────────────────────────────────────
    numeric_columns = [
        'budget', 'popularity', 'id', 'revenue',
        'runtime', 'vote_average', 'vote_count'
    ]
    movies_df = convert_numeric(movies_df, numeric_columns)

    date_columns = ['release_date']
    movies_df = convert_to_datetime(movies_df, date_columns)

    logger.info("Finished clean_movie_data pipeline (Spark version)")
    
    # If this DataFrame will be reused multiple times downstream:
    # return movies_df.cache()
    
    return movies_df


def replace_zero_values(df: DataFrame) -> DataFrame:
    """Replaces unrealistic placeholder values (0) with null."""
    logger.info("Replacing zero values in budget, revenue, runtime with NaN")
    
    return df.withColumn("budget",
                when(col("budget") == 0, lit(None)).otherwise(col("budget")))\
             .withColumn("revenue",
                when(col("revenue") == 0, lit(None)).otherwise(col("revenue")))\
             .withColumn("runtime",
                when(col("runtime") == 0, lit(None)).otherwise(col("runtime")))


def convert_budget_to_millions(df: DataFrame) -> DataFrame:
    """Converts budget & revenue from dollars to millions of dollars."""
    logger.info("Converting budget & revenue to millions of USD")
    
    return df.withColumn("budget_musd", col("budget") / 1000000.0)\
             .withColumn("revenue_musd", col("revenue") / 1000000.0)\
             .drop("budget", "revenue")


def clean_text_placeholders(df: DataFrame) -> DataFrame:
    """Cleans text placeholders in tagline & overview."""
    logger.info("Cleaning placeholder text in tagline & overview")
    
    bad_values = ["no tagline", "no overview", "no data", ""]
    
    for field in ["tagline", "overview"]:
        if field in df.columns:
            df = df.withColumn(
                field,
                when(
                    lower(trim(coalesce(col(field), lit("")))).isin(bad_values),
                    lit(None)
                ).otherwise(col(field))
            )
    return df


def adjust_vote_average(df: DataFrame) -> DataFrame:
    """Sets vote_average to null where vote_count is zero."""
    logger.info("Setting vote_average to NaN when vote_count == 0")
    
    if "vote_count" in df.columns and "vote_average" in df.columns:
        df = df.withColumn(
            "vote_average",
            when(col("vote_count") == 0, lit(None)).otherwise(col("vote_average"))
        )
    return df


def remove_duplicates(df: DataFrame) -> DataFrame:
    """Removes duplicate rows from the DataFrame."""
    logger.info("Removing duplicate rows")
    
    if "id" in df.columns:
        before = df.count()
        df = df.dropDuplicates(["id"])
        logger.info(f"After id-based deduplication: {df.count()} rows (removed {before - df.count()})")
    else:
        before = df.count()
        df = df.dropDuplicates()
        logger.info(f"After full-row deduplication: {df.count()} rows (removed {before - df.count()})")
    
    return df


def drop_rows_with_na_in_critical_columns(df: DataFrame, critical_columns: list) -> DataFrame:
    """Drops rows with null values in critical columns."""
    present_crits = [c for c in critical_columns if c in df.columns]
    
    if not present_crits:
        logger.info("No critical columns found to drop NA on")
        return df
    
    logger.info(f"Dropping rows with NA in critical columns: {present_crits}")
    
    before = df.count()
    df = df.dropna(subset=present_crits)
    after = df.count()
    
    logger.info(f"Rows removed: {before - after} | Remaining: {after}")
    return df


def keep_rows_with_min_non_nan(df: DataFrame, min_non_nan: int = 10) -> DataFrame:
    """
    Keeps rows with at least min_non_nan non-null values.
    WARNING: Expensive on very wide tables - use carefully
    """
    logger.info(f"Keeping only rows with >= {min_non_nan} non-null values")
    
    from pyspark.sql.functions import sum as sum_
    
    before = df.count()
    
    non_null_expr = sum_(when(col(c).isNotNull(), 1).otherwise(0) for c in df.columns)
    
    df = df.withColumn("_non_null_count", non_null_expr)\
           .filter(col("_non_null_count") >= min_non_nan)\
           .drop("_non_null_count")
    
    after = df.count()
    logger.info(f"Rows removed: {before - after} | Remaining: {after}")
    
    return df


def filter_released_movies(df: DataFrame) -> DataFrame:
    """Filters for released movies only and drops status column."""
    logger.info("Filtering for released movies only")
    
    if "status" in df.columns:
        before = df.count()
        df = df.filter(col("status") == "Released").drop("status")
        after = df.count()
        logger.info(f"Kept only Released movies: {after} (removed {before - after})")
    else:
        logger.info("No 'status' column found – skipping released-movie filter")
    
    return df


def reorder_columns(df: DataFrame, desired_order: list) -> DataFrame:
    """Reorders DataFrame columns based on the desired order."""
    logger.info("Reordering columns according to desired order")
    
    existing = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    
    return df.select(*(existing + remaining))


def reset_index(df: DataFrame) -> DataFrame:
    """In Spark there is no index like pandas -> this function is usually a no-op"""
    logger.info("reset_index() called - no effect in Spark DataFrame (no pandas-style index)")
    return df


def finalize_dataframe(df: DataFrame) -> DataFrame:
    """Finalizes the DataFrame by reordering columns (reset_index is no-op in Spark)."""
    logger.info("Finalizing DataFrame (reordering columns)")
    
    desired_order = [
        'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
        'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
        'production_countries', 'vote_count', 'vote_average', 'popularity', 'runtime',
        'overview', 'spoken_languages', 'poster_path', 'cast', 'cast_size',
        'director', 'crew_size'
    ]
    
    df = reorder_columns(df, desired_order)
    
    logger.info(f"DataFrame finalized – {df.count()} rows × {len(df.columns)} columns")
    return df