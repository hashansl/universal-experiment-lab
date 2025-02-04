
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry as geom
from scipy.ndimage import gaussian_filter

import libpysal as ps
from esda.moran import Moran

def generate_grid_dataframe(grid_side_length, autocorrelation="positive", random_seed=42):
    """
    Generates a GeoDataFrame based on the grid size of one side with different types of spatial autocorrelation.
    
    Parameters:
        grid_side_length (int): The number of squares along one side of the grid.
        autocorrelation (str): Type of spatial autocorrelation. Options:
                               "none" - No spatial correlation (pure random)
                               "positive" - Spatially smoothed using a Gaussian filter
                               "negative" - Checkerboard pattern (neighbors tend to be opposite)
        random_seed (int): Random seed for reproducibility.

    Returns:
        GeoDataFrame: A GeoDataFrame containing square geometries and values based on the selected correlation.
    """
    np.random.seed(random_seed)  # Set the random seed for reproducibility

    num_squares = grid_side_length ** 2
    df = pd.DataFrame({'Index': np.arange(num_squares)})

    # Define grid size and statistical parameters
    grid_size = (grid_side_length, grid_side_length)
    mean, std_dev = 0.5, 0.125

    # Generate initial random values
    # random_values = np.random.normal(mean, std_dev, grid_size)  # Random values with normal distribution

    lambda_vals = np.random.normal(mean, std_dev, grid_size)

    E = 30  # Expected value
    lambda_vals = lambda_vals * E

    random_values =  np.array([np.random.poisson(lam) for lam in lambda_vals])  # Poisson distribution

    if autocorrelation == "none":
        values = random_values  # No spatial correlation

    elif autocorrelation == "positive":
        values = gaussian_filter(random_values, sigma=1.5)  # Apply Gaussian smoothing for positive spatial correlation

    elif autocorrelation == "negative":
        # Create a checkerboard pattern to simulate negative spatial autocorrelation.
        # Cells where (x + y) is even get a higher value, while cells where (x + y) is odd get a lower value.
        x, y = np.indices(grid_size)
        # Option 1: Deterministic checkerboard values.
        # values = np.where((x + y) % 2 == 0, mean + std_dev, mean - std_dev)

        # Option 2: Checkerboard with a bit of noise (uncomment to use)
        noise = np.random.normal(0, std_dev/2, grid_size)
        base_values = np.where((x + y) % 2 == 0, mean + std_dev, mean - std_dev)
        values = base_values + noise

        # different valeus used

    else:
        raise ValueError("Invalid autocorrelation type. Choose from 'none', 'positive', or 'negative'.")

    # Assign values to the DataFrame
    df['Value'] = values.ravel()

    # Function to calculate square coordinates
    def calculate_square_coordinates(row):
        value = row['Index']
        x = value % grid_side_length
        y = value // grid_side_length
        return geom.Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])

    # Convert to GeoDataFrame with square geometries
    df['geometry'] = df.apply(calculate_square_coordinates, axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    return gdf


def calculate_moran_i(gdf, grid_side_length):
    """
    Calculates Moran's I for the given GeoDataFrame.

    Parameters:
        gdf (GeoDataFrame): A spatial dataframe with values and geometry.
        grid_side_length (int): The number of rows/columns in the square grid.

    Returns:
        float: Moran's I value indicating spatial autocorrelation.
    """
    # Create spatial weights matrix (rook contiguity)
    w = ps.weights.lat2W(grid_side_length, grid_side_length)
    w.transform = 'r'  # Row-standardized weights

    # Extract the 'Value' column for Moran's I computation
    values = gdf['Value'].values

    # Compute Moran’s I
    moran = Moran(values, w)

    return moran.I