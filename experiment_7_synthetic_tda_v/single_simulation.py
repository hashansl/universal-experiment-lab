
# import libraries
from utills.geodata import generate_grid_dataframes
from utills.adjacency_simplex import AdjacencySimplex  # Import the class
from utills.plot_utills import plot_simplicial_complex_gif
from utills.calculate_tda_summaries import compute_persistence

def run_single_simulation(grid_side_length, county_autocorrelation, census_autocorrelation, random_seed, variable_name):
    """
    Run a single simulation of the pipeline.
    """
    
    # Generate grid dataframes
    gdf_county, gdf_census = generate_grid_dataframes(grid_side_length, county_autocorrelation, census_autocorrelation, random_seed)
    
    for row in gdf_county.iterrows():

        country_index = row[1]['Index_county']

        # get the census tracts that belong to the county
        census_temp_df = gdf_census[gdf_census['Index_county'] == country_index]

        number_of_census_tracts = len(census_temp_df)

        for filter_meth in  ['up', 'down']:

            # Initialize the AdjacencySimplex class
            adj_simplex = AdjacencySimplex(census_temp_df, 'Rate_cen', threshold = None, filter_method = filter_meth)

            # Filter the GeoDataFrame
            filtered_df,gdf_id = adj_simplex.filter_sort_gdf()

            # Calculate the adjacent countries
            adj_simplex.calculate_adjacent_countries()

            # Form the simplicial complex
            simplex = adj_simplex.form_simplicial_complex()

            total_h0_points, tl, al, tml, aml, intervals_dim0 = compute_persistence(simplex,filtered_df, 'Rate_cen')

            if filter_meth == 'up':
                gdf_county.loc[country_index, 'up_AL'] = al
                gdf_county.loc[country_index, 'up_AML'] = aml

                gdf_county.loc[country_index, 'up_TL'] = tl
                gdf_county.loc[country_index, 'up_ML'] = tml

                # gdf_county.loc[country_index, 'intervals_dim0'] = intervals_dim0

            elif filter_meth == 'down':
                gdf_county.loc[country_index, 'down_AL'] = al
                gdf_county.loc[country_index, 'down_AML'] = aml

                gdf_county.loc[country_index, 'down_TL'] = tl
                gdf_county.loc[country_index, 'down_ML'] = tml

        # gdf_county.loc[country_index, 'intervals_dim0'] = intervals_dim0
        print(f'Country index: {country_index} has been processed.')
        print(f'Number of census tracts: {number_of_census_tracts}')
        print(f"Intervals_dim0: {intervals_dim0}")
                
        gdf_county.loc[country_index, 'cencus_count'] = number_of_census_tracts

    return gdf_county, gdf_census



# # Run a single simulation
# grid_side_length = 10
# county_autocorrelation = "positive"
# census_autocorrelation = "positive"
# random_seed = 42
# variable_name = 'Rate_cen'

# gdf_county, gdf_census = run_single_simulation(grid_side_length, county_autocorrelation, census_autocorrelation, random_seed, variable_name)

# print(gdf_county.head())