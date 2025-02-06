import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry as geom
from scipy.ndimage import gaussian_filter

import libpysal as ps
from esda.moran import Moran


def generate_grid_dataframes(grid_side_length, county_autocorrelation="positive", census_autocorrelation="positive" , random_seed=42):

    # PHASE 1: Generate census GeoDataFrame

    # Set the random seed for reproducibility
    np.random.seed(random_seed)  

    # Generate census GeoDataFrame

    total_counties = grid_side_length ** 2

    # for each county, generate a random number of census tracts from [2,3,4,5,6,7,8]
    num_census_tracts = np.random.choice([2,3,4,5,6,7,8], total_counties)

    # loop through each county and generate the census tracts
    df = pd.DataFrame()

    for county_index in range(total_counties):

        print(f"Generating census tracts for county {county_index}")

        num_tracts = num_census_tracts[county_index]

        lambda_vals = np.random.normal(loc=0.5, scale=0.125, size=num_tracts)

        # Multiply by E = 20
        lambda_vals *= 20
        lambda_vals = np.clip(lambda_vals, a_min=0, a_max=None)

        # Generate Poisson counts for each lambda
        poisson_counts = np.array([np.random.poisson(lam) for lam in lambda_vals])

        # assign Index_county and  poissons_counts to the dataframe
        df_temp = pd.DataFrame({'Index_county': [county_index] * num_tracts, 'Rate_cen': poisson_counts})

        

        # Function to calculate square coordinates
        def calculate_square_coordinates(row):
            value = row['Index_county']
            x = value % grid_side_length
            y = value // grid_side_length
            return geom.Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])
        
        # Convert to GeoDataFrame with square geometries
        df['geometry'] = df_temp.apply(calculate_square_coordinates, axis=1)
        gdf_census = gpd.GeoDataFrame(df, geometry='geometry')

        # Append to the main dataframe
        df = df.append(gdf_census, ignore_index=True)
        

        # break



# run the function
generate_grid_dataframes(3)