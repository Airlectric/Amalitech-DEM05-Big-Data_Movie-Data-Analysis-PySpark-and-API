from pyspark.sql import functions as F
from pyspark.sql.types import *
import logging

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------
# 1. Profit Calculation
#--------------------------------------------------------------------
def calculate_profit(df):
    """Calculates profit as revenue minus budget in millions USD"""
    logger.info("Calculating profit (revenue_musd - budget_musd)")
    return df.withColumn(
        "profit",
        F.col("revenue_musd") - F.col("budget_musd")
    )


#--------------------------------------------------------------------
# 2. ROI Calculation (only for budget >= 10M)
#--------------------------------------------------------------------
def calculate_roi(df):
    """
    Calculates ROI = revenue_musd / budget_musd
    Only for movies with budget_musd >= 10
    """
    logger.info("Calculating ROI (only for movies with budget >= 10M USD)")
    return df.withColumn(
        "roi",
        F.when(
            F.col("budget_musd") >= 10,
            F.col("revenue_musd") / F.col("budget_musd")
        ).otherwise(None)
    )


#--------------------------------------------------------------------
# 3. Rank top/bottom N movies by a metric
#--------------------------------------------------------------------
def rank_movies(df, metric, ascending=False, limit=10):
    """
    Generic ranking helper
    """
    order_col = F.col(metric).asc() if ascending else F.col(metric).desc()
    return df.orderBy(order_col).limit(limit)


#----------------------------------------------------------
# KPI CALCULATIONS FOR TMDB MOVIE DATA 
#----------------------------------------------------


#-------------------------------------------------------
# Revenue & Budget KPIs
#----------------------------------------------------

def highest_revenue(df, limit=10):
    return rank_movies(df, "revenue_musd", ascending=False, limit=limit)


def highest_budget(df, limit=10):
    return rank_movies(df, "budget_musd", ascending=False, limit=limit)


#----------------------------------------------------
# Profit KPIs
#--------------------------------------------------
def highest_profit(df, limit=10):
    df = calculate_profit(df)
    return rank_movies(df, "profit", ascending=False, limit=limit)


def lowest_profit(df, limit=10):
    df = calculate_profit(df)
    return rank_movies(df, "profit", ascending=True, limit=limit)


#--------------------------------------------------------------------
# ROI KPIs (only movies with budget >= min_budget)
#--------------------------------------------------------------------
def highest_roi(df, min_budget=10, limit=10):
    df = calculate_profit(df)
    df = calculate_roi(df)
    return rank_movies(
        df.filter(F.col("budget_musd") >= min_budget),
        "roi",
        ascending=False,
        limit=limit
    )


def lowest_roi(df, min_budget=10, limit=10):
    df = calculate_profit(df)
    df = calculate_roi(df)
    return rank_movies(
        df.filter(F.col("budget_musd") >= min_budget),
        "roi",
        ascending=True,
        limit=limit
    )


#--------------------------------------------------------------------
# Ratings & Votes KPIs
#--------------------------------------------------------------------
def most_voted(df, limit=10):
    return rank_movies(df, "vote_count", ascending=False, limit=limit)


def highest_rated(df, min_votes=10, limit=10):
    return rank_movies(
        df.filter(F.col("vote_count") >= min_votes),
        "vote_average",
        ascending=False,
        limit=limit
    )


def lowest_rated(df, min_votes=10, limit=10):
    return rank_movies(
        df.filter(F.col("vote_count") >= min_votes),
        "vote_average",
        ascending=True,
        limit=limit
    )


#--------------------------------------------------------------------
# Popularity KPI
#--------------------------------------------------------------------

def most_popular(df, limit=10):
    return rank_movies(df, "popularity", ascending=False, limit=limit)



#--------------------------------------------------------------------
# 5. Search best Sci-Fi + Action movies with Bruce Willis
#--------------------------------------------------------------------
def search_best_scifi_action_bruce(df):
    """
    Searches for best Sci-Fi + Action movies starring Bruce Willis
    Works with genres and cast as pipe-separated strings
    """
    logger.info("Searching for best Sci-Fi + Action movies starring Bruce Willis")

    mask = (
        F.col("genres").contains("Science Fiction") |
        F.col("genres").contains("Sci-Fi")
    ) & (
        F.col("genres").contains("Action")
    ) & (
        F.lower(F.col("cast")).contains("bruce willis")
    )

    result = df.filter(mask) \
               .orderBy(F.desc("vote_average"))

    logger.info(f"Found {result.count()} Sci-Fi/Action movies with Bruce Willis")
    return result


#--------------------------------------------------------------------
# 6. Search Uma Thurman + Quentin Tarantino movies
#--------------------------------------------------------------------
def search_uma_thurman_tarentino(df):
    """
    Searches for movies starring Uma Thurman directed by Quentin Tarantino
    Works with cast and director as pipe-separated strings
    """
    logger.info("Searching for movies with Uma Thurman directed by Quentin Tarantino")

    mask = (
        F.lower(F.col("cast")).contains("uma thurman")
    ) & (
        F.lower(F.col("director")).contains("quentin tarantino")
    )

    result = df.filter(mask) \
               .orderBy(F.desc("vote_average"))

    logger.info(f"Found {result.count()} Uma Thurman + Quentin Tarantino collaborations")
    return result


#--------------------------------------------------------------------
# 7. Franchise vs Standalone comparison
#--------------------------------------------------------------------
def franchise_vs_standalone(df):
    """
    Compares average performance metrics between franchise and standalone movies
    Uses belongs_to_collection_name as franchise indicator
    """
    logger.info("Comparing franchise vs standalone movie performance")

    df = calculate_roi(df)

    df = df.withColumn(
        "is_franchise",
        F.when(F.col("belongs_to_collection").isNotNull(), True).otherwise(False)
    )

    grouped = df.groupBy("is_franchise").agg(
        F.mean("revenue_musd").alias("mean_revenue"),
        F.expr("percentile_approx(roi, 0.5)").alias("median_roi"),
        F.mean("budget_musd").alias("mean_budget"),
        F.mean("popularity").alias("mean_popularity"),
        F.mean("vote_average").alias("mean_rating")
    )

    logger.info("Franchise vs standalone comparison completed")
    return grouped


#--------------------------------------------------------------------
# 8. Most successful franchises by total revenue
#--------------------------------------------------------------------
def franchise_success(df):
    """
    Analyzes most successful franchises by total revenue
    Uses belongs_to_collection_name for grouping
    """
    logger.info("Analyzing most successful movie franchises by total revenue")

    df = df.filter(F.col("belongs_to_collection").isNotNull())

    grouped = df.groupBy("belongs_to_collection").agg(
        F.count("id").alias("count_movies"),
        F.sum("budget_musd").alias("total_budget_musd"),
        F.mean("budget_musd").alias("mean_budget_musd"),
        F.sum("revenue_musd").alias("total_revenue_musd"),
        F.mean("revenue_musd").alias("mean_revenue_musd"),
        F.mean("vote_average").alias("mean_rating")
    ).orderBy(F.desc("total_revenue_musd"))

    logger.info(f"Franchise success analysis completed – {grouped.count()} franchises ranked")
    return grouped


#--------------------------------------------------------------------
# 9. Most successful directors by total revenue
#--------------------------------------------------------------------
def director_success(df, top_n=10):
    """
    Analyzes most successful directors by total revenue
    Handles multiple directors (pipe-separated)
    """
    logger.info(f"Calculating top {top_n} most successful directors by total revenue")

    # Explode directors
    exploded = df.withColumn(
        "director",
        F.explode(F.split(F.col("director"), "\\|"))
    ).withColumn(
        "director",
        F.trim(F.col("director"))
    ).filter(F.col("director") != "")

    grouped = exploded.groupBy("director").agg(
        F.count("id").alias("total_movies_directed"),
        F.sum("revenue_musd").alias("total_revenue_musd"),
        F.mean("vote_average").alias("mean_rating")
    ).orderBy(F.desc("total_revenue_musd")) \
     .limit(top_n)

    logger.info(f"Director success ranking completed – top {top_n} returned")
    return grouped