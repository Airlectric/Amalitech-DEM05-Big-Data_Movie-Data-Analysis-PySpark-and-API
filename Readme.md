# Project Summary: Movie Data Analysis

## Methodology

This analysis leverages the TMDB API to fetch detailed metadata for a curated list of high-profile movies (e.g., Avengers series, Star Wars, Jurassic World). The process involves:

1. **Data Collection**: Using `tmdb_client` to retrieve movie details, including credits, for specified movie IDs. Data is compiled into a Pandas DataFrame.

2. **Data Cleaning and Preprocessing** (via `clean_data` module):
   - Extract nested JSON fields (e.g., genres, cast, crew, production countries).
   - Convert columns to appropriate types (numeric, datetime).
   - Replace unrealistic placeholders (e.g., zero budgets/revenues with NaN).
   - Scale budget and revenue to millions USD.
   - Remove duplicates, drop rows with missing critical data (e.g., title, ID), and filter for released movies only.
   - Finalize DataFrame structure with desired column order.

3. **KPI Calculations and Analysis** (via `kpi_analysis` module):
   - Compute profit (revenue - budget) and ROI (revenue / budget, for budgets ≥ $10M).
   - Rank movies by metrics like revenue, profit, ROI, votes, ratings, and popularity.
   - Perform advanced filters (e.g., Sci-Fi/Action movies with Bruce Willis, collaborations like Uma Thurman and Quentin Tarantino).
   - Compare franchise vs. standalone performance (e.g., mean revenue, median ROI).
   - Aggregate franchise success (e.g., total/mean revenue, movie count).
   - Identify top directors by total revenue.

4. **Visualizations** (via `visualizations` module):
   - Scatter plots: Revenue vs. Budget, Popularity vs. Rating.
   - Bar plots: ROI Distribution by Genre, Franchise vs. Standalone Success.
   - Line plot: Yearly Box Office Performance.

Logging is integrated throughout for traceability, and errors (e.g., API fetch failures) are handled with retries.