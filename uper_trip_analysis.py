import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import logging

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    "pickup_time": ["2024-03-20 08:30:00", "2024-03-20 09:15:00", "2024-03-20 10:45:00"],
    "fare_amount": [15.5, 8.2, 12.0],
    "trip_duration": [20, 10, 15],
    "distance": [5.2, 2.8, 3.5],
    "pickup_lat": [40.7128, 40.7528, 40.7336],
    "pickup_lon": [-74.0060, -73.9772, -74.0027],
    "dropoff_lat": [40.7359, 40.7614, 40.7488],
    "dropoff_lon": [-73.9911, -73.9827, -73.9680]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save as CSV in the same directory as the script
df.to_csv("uber_trip_data.csv", index=False)

print("CSV file saved successfully!")


# Configure logging for detailed debugging output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data(filepath):
    """
    Loads the dataset, handles missing values, and converts columns to appropriate types.
    Also derives new features such as hour of pickup and day-of-week if not available.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

    # Drop rows with missing crucial values (or consider imputation)
    df.dropna(subset=['pickup_time', 'fare_amount', 'trip_duration', 'distance'], inplace=True)

    # Convert types: pickup_time and dropoff_time to datetime, numeric conversions for fare and trip_duration
    df['pickup_time'] = pd.to_datetime(df['pickup_time'], errors='coerce')
    if 'dropoff_time' in df.columns:
        df['dropoff_time'] = pd.to_datetime(df['dropoff_time'], errors='coerce')
    df['fare_amount'] = pd.to_numeric(df['fare_amount'], errors='coerce')
    df['trip_duration'] = pd.to_numeric(df['trip_duration'], errors='coerce')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    # Create additional time-based features
    df['hour'] = df['pickup_time'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['pickup_time'].dt.day_name()

    logging.info("Data preprocessing complete.")
    return df

# =============================================================================
# 2. Exploratory Data Analysis (EDA)
# =============================================================================
def perform_eda(df):
    """
    Performs EDA by printing summaries and generating plots:
      - Distribution of trip duration and fare
      - Correlation heatmap
      - Peak hour and day analysis
      - Trip type distribution (if available)
    """
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Plot distributions for trip duration and fare
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['trip_duration'], kde=True, bins=30, color='skyblue')
    plt.title('Trip Duration Distribution')
    plt.xlabel('Trip Duration')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['fare_amount'], kde=True, bins=30, color='salmon')
    plt.title('Fare Amount Distribution')
    plt.xlabel('Fare Amount')
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Peak hours analysis
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour', data=df, palette='viridis')
    plt.title('Trips by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Trips')
    plt.show()

    # Peak days analysis
    plt.figure(figsize=(10, 6))
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='day_of_week', data=df, order=order, palette='magma')
    plt.title('Trips by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Trips')
    plt.show()

    # Trip type distribution (if available)
    if 'trip_type' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='trip_type', data=df, palette='Set2')
        plt.title('Trip Type Distribution')
        plt.xlabel('Trip Type')
        plt.ylabel('Count')
        plt.show()

    logging.info("EDA completed.")

# =============================================================================
# 3. Geospatial Analysis with Folium
# =============================================================================
def create_geospatial_map(df, output_html="uber_trip_map.html"):
    """
    Creates an interactive map using Folium to plot pickup and dropoff locations.
    """
    if 'pickup_lat' in df.columns and 'pickup_lon' in df.columns:
        center_lat = df['pickup_lat'].mean()
        center_lon = df['pickup_lon'].mean()
        ride_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Add pickup and dropoff markers (if dropoff exists)
        for idx, row in df.iterrows():
            folium.CircleMarker(location=[row['pickup_lat'], row['pickup_lon']],
                                radius=2,
                                color='blue',
                                fill=True,
                                fill_opacity=0.6,
                                tooltip="Pickup").add_to(ride_map)
            if 'dropoff_lat' in df.columns and 'dropoff_lon' in df.columns:
                folium.CircleMarker(location=[row['dropoff_lat'], row['dropoff_lon']],
                                    radius=2,
                                    color='red',
                                    fill=True,
                                    fill_opacity=0.6,
                                    tooltip="Dropoff").add_to(ride_map)
        ride_map.save(output_html)
        logging.info(f"Geospatial map saved as {output_html}")
    else:
        logging.warning("Pickup location data missing; geospatial map not created.")

# =============================================================================
# 4. Advanced Analysis – Predictive Modeling and Clustering
# =============================================================================
def predict_trip_duration(df):
    """
    Predicts trip duration using Linear Regression based on distance, hour, and fare_amount.
    """
    required_cols = ['distance', 'hour', 'fare_amount', 'trip_duration']
    if not all(col in df.columns for col in required_cols):
        logging.error("Missing required columns for trip duration prediction.")
        return

    X = df[['distance', 'hour', 'fare_amount']]
    y = df['trip_duration']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mse = mean_squared_error(y, y_pred)
    
    logging.info(f"Trip Duration Regression Model: R-squared = {r2:.2f}, MSE = {mse:.2f}")
    print(f"Trip Duration Regression Model: R-squared = {r2:.2f}, MSE = {mse:.2f}")

def cluster_pickup_locations(df, n_clusters=5):
    """
    Uses KMeans clustering to group pickup locations into demand zones.
    """
    if 'pickup_lat' not in df.columns or 'pickup_lon' not in df.columns:
        logging.error("Missing pickup location data for clustering.")
        return

    coords = df[['pickup_lat', 'pickup_lon']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(coords)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pickup_lon', y='pickup_lat', hue='cluster', data=df, palette='viridis')
    plt.title('Clustering of Pickup Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title="Cluster")
    plt.show()

    logging.info("Clustering of pickup locations completed.")

def predict_fare(df):
    """
    Predicts fare using a Random Forest Regressor. If weather_condition is categorical,
    it is encoded. The model uses distance, hour, and weather as features.
    """
    required_cols = ['distance', 'hour', 'fare_amount', 'weather_condition']
    if not all(col in df.columns for col in required_cols):
        logging.error("Missing required columns for fare prediction.")
        return

    # Encode weather_condition as a numeric category
    df['weather_encoded'] = df['weather_condition'].astype('category').cat.codes

    features = ['distance', 'hour', 'weather_encoded']
    X = df[features]
    y = df['fare_amount']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5, color='teal')
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.title('Fare Prediction: Actual vs Predicted')
    plt.show()

    logging.info(f"Fare Prediction Model: MSE = {mse:.2f}")
    print(f"Fare Prediction Model: MSE = {mse:.2f}")

def predict_demand(df):
    """
    Predicts ride demand by hour and day using Random Forest. This example assumes a
    pre-aggregated column 'trip_count' exists or groups data accordingly.
    """
    if 'trip_count' not in df.columns:
        # Create a simple aggregated demand feature by grouping by hour and day_of_week
        demand = df.groupby(['day_of_week', 'hour']).size().reset_index(name='trip_count')
        logging.info("Aggregated trip_count by day_of_week and hour.")
    else:
        demand = df.copy()

    # Encode day_of_week
    demand['day_encoded'] = demand['day_of_week'].astype('category').cat.codes
    X = demand[['day_encoded', 'hour']]
    y = demand['trip_count']

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5, color='purple')
    plt.xlabel('Actual Trip Count')
    plt.ylabel('Predicted Trip Count')
    plt.title('Demand Prediction: Actual vs Predicted')
    plt.show()

    logging.info(f"Demand Prediction Model: MSE = {mse:.2f}")
    print(f"Demand Prediction Model: MSE = {mse:.2f}")

def estimate_surge_multiplier(df):
    """
    Estimates a surge multiplier metric based on fare efficiency.
    The assumption is that under normal conditions, the fare per unit distance 
    is within a certain range. Deviations above the median may indicate surge pricing.
    """
    if not all(col in df.columns for col in ['fare_amount', 'distance']):
        logging.error("Missing columns for surge multiplier estimation.")
        return

    # Compute fare per unit distance; avoid division by zero
    df['fare_per_distance'] = df.apply(lambda row: row['fare_amount'] / row['distance']
                                       if row['distance'] > 0 else np.nan, axis=1)
    # Drop rows with NaN values resulted from zero distance
    valid_fares = df['fare_per_distance'].dropna()
    median_fare_rate = valid_fares.median()
    
    # Define surge multiplier as the ratio of fare per distance to the median rate
    df['surge_multiplier'] = df['fare_per_distance'] / median_fare_rate

    plt.figure(figsize=(10, 6))
    sns.histplot(df['surge_multiplier'].dropna(), bins=30, kde=True, color='orange')
    plt.title('Surge Multiplier Distribution')
    plt.xlabel('Surge Multiplier')
    plt.ylabel('Frequency')
    plt.show()

    logging.info("Surge multiplier estimation complete.")
    print(f"Median Fare per Distance: {median_fare_rate:.2f}")
    print("Surge multiplier calculated for each trip.")

# =============================================================================
# Main Function to Execute the Analysis Pipeline
# =============================================================================
def main():
    filepath = "uber_trip_data.csv"  # Update this with your dataset path
    df = load_and_preprocess_data(filepath)
    if df is None:
        return

    # Perform Exploratory Data Analysis (EDA)
    perform_eda(df)

    # Generate Geospatial Map
    create_geospatial_map(df)

    # Advanced Analysis
    predict_trip_duration(df)
    cluster_pickup_locations(df, n_clusters=5)
    predict_fare(df)
    predict_demand(df)
    estimate_surge_multiplier(df)

    # Reporting: You may export results, save plots, or compile insights in a notebook
    logging.info("Analysis complete. Review the plots and logs for insights.")

if __name__ == '__main__':
    main()
