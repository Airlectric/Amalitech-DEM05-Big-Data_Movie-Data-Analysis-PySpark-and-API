import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import *
from ..utils.logger_config import logger
from ..analysis.kpi_analysis import calculate_roi

#--------------------------------------------------------------------
# 1. Revenue vs Budget Scatter Plot
#--------------------------------------------------------------------
def plot_revenue_vs_budget(df, sample_size=10000):
    """
    Plots Revenue vs Budget scatter plot using sampled data.
    Uses Spark to sample and collect only necessary points.
    
    Parameters:
        sample_size (int): Number of points to sample for plotting (default: 10000)
    """
    logger.info(f"Preparing Revenue vs Budget scatter plot (sampling {sample_size} rows)")
    
    # Select and sample data (sampling prevents driver OOM on huge datasets)
    plot_data = df.select("budget_musd", "revenue_musd") \
                  .na.drop() \
                  .sample(fraction=1.0, seed=42) \
                  .limit(sample_size) \
                  .toPandas()

    if plot_data.empty:
        logger.warning("No valid data for Revenue vs Budget plot")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data['budget_musd'], plot_data['revenue_musd'], alpha=0.6, s=20)
    plt.xlabel('Budget (Millions USD)')
    plt.ylabel('Revenue (Millions USD)')
    plt.title('Revenue vs Budget (Sampled Data)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    logger.info("Revenue vs Budget plot completed")


#--------------------------------------------------------------------
# 2. Average ROI by Genre (bar plot)
#--------------------------------------------------------------------
def plot_roi_by_genre(df):
    """
    Plots average ROI distribution by genre.
    Explodes genres pipe-separated string and computes mean ROI per genre.
    """
    logger.info("Preparing ROI by Genre bar plot")

    # Compute ROI and keep the returned DataFrame
    df_with_roi = calculate_roi(df)

    # Explode genres (pipe-separated string)
    df_exploded = df_with_roi.withColumn(
        "genre",
        F.explode(F.split(F.col("genres"), "\\|"))
    ).withColumn(
        "genre",
        F.trim(F.col("genre"))
    ).filter(
        (F.col("genre") != "") & F.col("roi").isNotNull()
    )

    # Compute average ROI per genre
    genre_roi = df_exploded.groupBy("genre").agg(
        F.mean("roi").alias("avg_roi")
    ).orderBy("avg_roi") \
     .toPandas()

    if genre_roi.empty:
        logger.warning("No valid ROI data by genre for plotting")
        return

    plt.figure(figsize=(12, 6))
    plt.bar(genre_roi['genre'], genre_roi['avg_roi'], color='skyblue')
    plt.title('Average ROI by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average ROI')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    logger.info("ROI by Genre plot completed")


#--------------------------------------------------------------------
# 3. Popularity vs Rating Scatter Plot
#--------------------------------------------------------------------
def plot_popularity_vs_rating(df, sample_size=10000):
    """
    Plots Popularity vs Rating scatter plot using sampled data.
    """
    logger.info(f"Preparing Popularity vs Rating scatter plot (sampling {sample_size} rows)")

    plot_data = df.select("vote_average", "popularity") \
                  .na.drop() \
                  .sample(fraction=1.0, seed=42) \
                  .limit(sample_size) \
                  .toPandas()

    if plot_data.empty:
        logger.warning("No valid data for Popularity vs Rating plot")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_data['vote_average'], plot_data['popularity'], alpha=0.6, s=20)
    plt.xlabel('Rating (Vote Average)')
    plt.ylabel('Popularity')
    plt.title('Popularity vs Rating (Sampled Data)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logger.info("Popularity vs Rating plot completed")


#--------------------------------------------------------------------
# 4. Yearly Box Office Performance (line plot)
#--------------------------------------------------------------------
def plot_yearly_box_office(df):
    """
    Plots total yearly box office revenue.
    Groups by release year and sums revenue_musd.
    """
    logger.info("Preparing Yearly Box Office Performance plot")

    yearly = df.withColumn("release_year", F.year(F.col("release_date"))) \
               .groupBy("release_year") \
               .agg(F.sum("revenue_musd").alias("total_revenue_musd")) \
               .orderBy("release_year") \
               .toPandas()

    if yearly.empty:
        logger.warning("No valid yearly revenue data for plotting")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(yearly['release_year'], yearly['total_revenue_musd'], marker='o', linestyle='-', color='b')
    plt.title('Yearly Box Office Performance')
    plt.xlabel('Year')
    plt.ylabel('Total Revenue (Millions USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logger.info("Yearly Box Office plot completed")


#--------------------------------------------------------------------
# 5. Franchise vs Standalone Success (bar plot)
#--------------------------------------------------------------------
def plot_franchise_vs_standalone(df):
    """
    Plots average revenue comparison between franchise and standalone movies.
    Uses belongs_to_collection to classify.
    """
    logger.info("Preparing Franchise vs Standalone Success plot")

    df = df.withColumn(
        "is_franchise",
        F.when(F.col("belongs_to_collection").isNotNull(), "Franchise").otherwise("Standalone")
    )

    comparison = df.groupBy("is_franchise").agg(
        F.mean("revenue_musd").alias("avg_revenue")
    ).toPandas()

    if comparison.empty:
        logger.warning("No data for Franchise vs Standalone comparison")
        return

    plt.figure(figsize=(8, 6))
    plt.bar(comparison['is_franchise'], comparison['avg_revenue'], color=['#1f77b4', '#ff7f0e'])
    plt.title('Franchise vs Standalone Success')
    plt.xlabel('Movie Type')
    plt.ylabel('Average Revenue (Millions USD)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    logger.info("Franchise vs Standalone plot completed")