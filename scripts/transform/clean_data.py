from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType
from ..utils.logger_config import logger
from pyspark.sql import DataFrame


def drop_irrelevant_columns(movies_df, columns_to_drop):
    """Drops irrelevant columns from the movie data."""
    logger.info(f"Dropping columns: {columns_to_drop}")
    existing = [c for c in columns_to_drop if c in movies_df.columns]
    if existing:
        movies_df = movies_df.drop(*existing)
    return movies_df


def extract_json_field(
    df: DataFrame,
    col: str,
    key: str,
    join_with: str = "|"
) -> DataFrame:
    """
    - map -> extract value by key
    - array<map> -> extract key from each element and join
    """

    return df.withColumn(
        col,
        F.when(
            F.col(col).isNull(),
            F.lit(None)
        )
        # array<map>
        .when(
            F.expr(f"typeof({col}) LIKE 'array%'"),
            F.concat_ws(
                join_with,
                F.expr(f"transform({col}, x -> x['{key}'])")
            )
        )
        # map
        .otherwise(F.col(col)[key])
    )


def extract_credit_json_fields(
    df: DataFrame,
    col: str = "credits",
    join_with: str = "|"
) -> DataFrame:

    df = (
        df
        # Cast names
        .withColumn(
            "cast",
            F.concat_ws(
                join_with,
                F.expr(
                    f"""
                    transform(
                        {col}['cast'],
                        x -> x['name']
                    )
                    """
                )
            )
        )

        # Cast size
        .withColumn(
            "cast_size",
            F.size(F.col(col)["cast"])
        )

        # Directors only
        .withColumn(
            "director",
            F.concat_ws(
                join_with,
                F.expr(
                    f"""
                    transform(
                        filter({col}['crew'], x -> x['job'] = 'Director'),
                        x -> x['name']
                    )
                    """
                )
            )
        )

        # Crew size
        .withColumn(
            "crew_size",
            F.size(F.col(col)["crew"])
        )

        .drop(col)
    )

    return df

def extract_production_countries(
    df: DataFrame,
    col: str = "origin_country",
    join_with: str = "|"
) -> DataFrame:

    return df.withColumn(
        col,
        F.concat_ws(join_with, F.col(col))
    )



def inspect_categorical_columns_using_value_counts(df, cols):
    """Show value counts (driver action - use only for small/medium data!)"""
    logger.info(f"Inspecting value counts for: {cols}")
    for c in [col for col in cols if col in df.columns]:
        print(f"\nValue counts for ====== {c} ======")
        df.groupBy(c).count().orderBy(F.desc("count")).show(20, truncate=False)


def convert_numeric(df, cols):
    """Cast to double (safer than float)"""
    logger.info(f"Converting to numeric: {cols}")
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
    return df


def convert_to_datetime(df, cols):
    """release_date is string → convert to date"""
    logger.info(f"Converting to date: {cols}")
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.to_date(F.col(c), "yyyy-MM-dd"))
    return df


def clean_movie_data(movies_df: DataFrame) -> DataFrame:
    """
    Full PySpark cleaning pipeline
    """

    json_columns = {
        "belongs_to_collection": "name",
        "genres": "name",
        "spoken_languages": "english_name",
        "production_companies": "name",
        "production_countries": "name",
    }

    for col, key in json_columns.items():
        movies_df = extract_json_field(movies_df, col, key)

    movies_df = extract_production_countries(
        movies_df, col="origin_country"
    )

    movies_df = extract_credit_json_fields(
        movies_df, col="credits"
    )

    # Numeric casting (already numeric but enforced)
    numeric_columns = [
        "budget", "popularity", "id", "revenue",
        "runtime", "vote_average", "vote_count"
    ]

    for c in numeric_columns:
        movies_df = movies_df.withColumn(c, F.col(c).cast("double"))

    # Date conversion
    movies_df = movies_df.withColumn(
        "release_date",
        F.to_date("release_date", "yyyy-MM-dd")
    )

    return movies_df



def replace_zero_values(df):
    logger.info("Replacing 0 → NULL in budget/revenue/runtime")
    for c in ['budget', 'revenue', 'runtime']:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c) == 0, None).otherwise(F.col(c)))
    return df


def convert_budget_to_millions(df):
    logger.info("Converting budget & revenue to millions USD")
    df = df.withColumn("budget_musd", F.round(F.col("budget") / 1000000, 4)) \
           .withColumn("revenue_musd", F.round(F.col("revenue") / 1000000, 4)) \
           .drop("budget", "revenue")
    return df


def clean_text_placeholders(df):
    logger.info("Cleaning placeholder texts")
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
    logger.info("Nullifying vote_average when vote_count = 0")
    if all(c in df.columns for c in ['vote_count', 'vote_average']):
        df = df.withColumn(
            "vote_average",
            F.when(F.col("vote_count") == 0, None).otherwise(F.col("vote_average"))
        )
    return df


def replace_unrealistic_values(df):
    logger.info("Starting replace_unrealistic_values pipeline")
    df = replace_zero_values(df)
    df = convert_budget_to_millions(df)
    df = clean_text_placeholders(df)
    df = adjust_vote_average(df)
    logger.info("Finished replace_unrealistic_values pipeline")
    return df


def remove_duplicates(df):
    logger.info("Removing duplicates")
    before = df.count()
    if "id" in df.columns:
        df = df.dropDuplicates(["id"])
    else:
        df = df.dropDuplicates()
    logger.info(f"After deduplication: {df.count()} rows (removed {before - df.count()})")
    return df


def drop_rows_with_na_in_critical_columns(df, critical_columns):
    present = [c for c in critical_columns if c in df.columns]
    if present:
        before = df.count()
        df = df.dropna(subset=present)
        logger.info(f"After dropping NA in critical: {df.count()} rows (removed {before - df.count()})")
    return df


def keep_rows_with_min_non_nan(df, min_non_nan=10):
    """Warning: expensive on very large datasets — consider skipping or sampling"""
    logger.info(f"Filtering rows with >= {min_non_nan} non-null values")
    before = df.count()
    count_non_null = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in df.columns)
    df = df.withColumn("_nn_count", count_non_null)\
           .filter(F.col("_nn_count") >= min_non_nan)\
           .drop("_nn_count")
    after = df.count()
    logger.info(f"After min non-null filter: {after} rows (removed {before - after})")
    return df


def filter_released_movies(df):
    if "status" in df.columns:
        before = df.count()
        df = df.filter(F.col("status") == "Released").drop("status")
        logger.info(f"Kept Released only: {df.count()} (removed {before - df.count()})")
    return df


def removing_na_and_duplicates(df):
    logger.info("Starting NA & duplicate removal pipeline")
    df = remove_duplicates(df)
    df = drop_rows_with_na_in_critical_columns(df, ["title", "id"])
    df = keep_rows_with_min_non_nan(df, 10)          # ← consider disabling on huge data
    df = filter_released_movies(df)
    logger.info(f"Final size after cleaning: {df.count()} rows")
    return df


def reorder_columns(df, desired_order):
    existing = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df.select(*(existing + remaining))


def reset_index(df):
    # Spark DataFrames don't have indexes like pandas → usually no-op
    # If you really need row number:
    # df = df.withColumn("row_id", F.monotonically_increasing_id())
    return df


def finalize_dataframe(df):
    desired_order = [
        'id', 'title', 'tagline', 'release_date', 'genres', 'belongs_to_collection',
        'original_language', 'budget_musd', 'revenue_musd', 'production_companies',
        'production_countries', 'vote_count', 'vote_average', 'popularity', 'runtime',
        'overview', 'spoken_languages', 'poster_path', 'cast', 'cast_size', 'director', 'crew_size'
    ]
    df = reorder_columns(df, desired_order)
    df = reset_index(df)
    logger.info(f"Finalized – rows: {df.count()}, columns: {len(df.columns)}")
    return df