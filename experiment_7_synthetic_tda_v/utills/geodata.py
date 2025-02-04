import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry as geom
from scipy.ndimage import gaussian_filter

import libpysal as ps
from esda.moran import Moran


def generate_grid_dataframes(grid_side_length, county_autocorrelation="positive", census_autocorrelation="positive" , random_seed=42):

    # PHASE 1: Generate county GeoDataFrame

    # Set the random seed for reproducibility
    np.random.seed(random_seed)  

    # Generate county GeoDataFrame

    num_squares = grid_side_length ** 2
    df = pd.DataFrame({'Index_county': np.arange(num_squares)})

    # Define grid size and statistical parameters
    grid_size = (grid_side_length, grid_side_length)
    mean, std_dev = 0.5, 0.5

    # Generate initial random values
    random_values = np.random.normal(mean, std_dev, grid_size) # Random values with normal distribution
    # random_values =  np.random.wald(mean=0.5, scale=1, size=grid_size)  # Inverse Gaussian

    if county_autocorrelation == "none":
        values = random_values  # No spatial correlation

    elif county_autocorrelation == "positive":
        values = gaussian_filter(random_values, sigma=1.5)  # Apply Gaussian smoothing for spatial correlation

    else:
        raise ValueError("Invalid autocorrelation type. Choose from 'none', 'positive', or 'negative'.")

    # Assign values to the DataFrame
    df['Rate_cou'] = values.ravel()

    # Multiply values by 30 (expected value)
    df['Rate_cou'] = df['Rate_cou'] * 30

    # Function to calculate square coordinates
    def calculate_square_coordinates(row):
        value = row['Index_county']
        x = value % grid_side_length
        y = value // grid_side_length
        return geom.Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])

    # Convert to GeoDataFrame with square geometries
    df['geometry'] = df.apply(calculate_square_coordinates, axis=1)
    gdf_county = gpd.GeoDataFrame(df, geometry='geometry')


    # PHASE 2: Generate census tract GeoDataFrame

    # The possible numbers of census tracts per county.
    census_count_values = [4, 9, 16, 25, 49, 64]

    # Create an empty list to store each county's census tract GeoDataFrame
    county_gdf_list = []

    for row in gdf_county.itertuples():

        parent_county_index = row[1]   # assuming first column is county id
        parent_county_rate = row[2]    # assuming second column is the rate value
        
        # Randomly choose the number of tracts for this county
        num_tracts = np.random.choice(census_count_values)
        # print(num_tracts)

        # Divide the parent's county rate into tract rates using a Dirichlet distribution
        tract_rates = np.random.dirichlet(np.ones(num_tracts)) * parent_county_rate

        # Create a DataFrame to hold census tract data
        df_census = pd.DataFrame({'Index_census': np.arange(num_tracts)})

        if census_autocorrelation == "positive":
            # Apply Gaussian smoothing to introduce some positive spatial autocorrelation
            values = gaussian_filter(tract_rates, sigma=1.5)
        else:
            values = tract_rates
            print("No spatial autocorrelation applied to census tracts.")
            # later we can add more options for autocorrelation

        df_census['Rate_cen'] = values.ravel()

        # Determine the side length for the square grid (assumes a perfect square)
        census_grid_side_length = int(np.sqrt(num_tracts))
        
        # Function to create a square polygon for each census tract
        def calculate_square_coordinates(row):
            idx = row['Index_census']
            x = idx % census_grid_side_length
            y = idx // census_grid_side_length
            return geom.Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])
        
        # Calculate the geometry for each tract and add it to the DataFrame
        df_census['geometry'] = df_census.apply(calculate_square_coordinates, axis=1)
        
        # Convert the DataFrame into a GeoDataFrame
        gdf_single_county = gpd.GeoDataFrame(df_census, geometry='geometry')
        
        # Add the parent county id to each tract
        gdf_single_county['Index_county'] = parent_county_index
        
        # Append this county's GeoDataFrame to our list
        county_gdf_list.append(gdf_single_county)

    # Combine all county census tract GeoDataFrames into one final GeoDataFrame
    gdf_census = pd.concat(county_gdf_list, ignore_index=True)


    return gdf_county, gdf_census
    









