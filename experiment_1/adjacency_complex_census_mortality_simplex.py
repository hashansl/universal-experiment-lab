# Import libraries
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import gudhi
from tqdm import tqdm
from persim import PersistenceImager
import invr
import matplotlib as mpl

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Matplotlib default settings
mpl.rcParams.update(mpl.rcParamsDefault)

# Utility functions
def get_folders(location):
    """Get list of folders in a directory."""
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

def generate_adjacent_counties(dataframe, variable_name):
    """Generate adjacent counties based on given dataframe and variable."""
    filtered_df = dataframe
    adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')
    adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')
    adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()
    adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)
    adjacencies_list = adjacent_counties['adjacent'].tolist()
    county_list = adjacent_counties['county'].tolist()
    merged_df = pd.merge(adjacent_counties, dataframe, left_on='county', right_on='sortedID', how='left')
    merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
    return adjacencies_list, merged_df, county_list

def form_simplicial_complex(adjacent_county_list, county_list):
    """Form a simplicial complex based on adjacent counties."""
    max_dimension = 3
    V = invr.incremental_vr([], adjacent_county_list, max_dimension, county_list)
    return V

def create_variable_folders(base_path, variables):
    """Create folders for each variable."""
    for variable in variables:
        os.makedirs(os.path.join(base_path, variable), exist_ok=True)
    print('Done creating folders for each variable')

def process_state(state, selected_variables, selected_variables_with_censusinfo,df,mort):
    """Process data for a given state."""
    svi_od = mort[mort['ST_ABB'] == state]

        
    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)



    # Get the unique counties
    unique_county_stcnty = svi_od_filtered_state['STCNTY'].unique()

    print(f'length of unique counties: {len(unique_county_stcnty)}')

    # get the counties with more than 10 census data
    unique_county_stcnty = [county_stcnty for county_stcnty in unique_county_stcnty if len(svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]) > 10]

    print(f'length of unique counties with more than 10 census data: {len(unique_county_stcnty)}')

    print(f'svi_od shape: {svi_od_filtered_state.shape}')


    for county_stcnty in unique_county_stcnty:
        # Filter the dataframe to include only the current county
        county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]

        census_count = len(county_svi_df)
    
        for variable_name in selected_variables:
            df_one_variable = county_svi_df[['STCNTY', variable_name, 'geometry']]
            df_one_variable = df_one_variable.sort_values(by=variable_name)
            df_one_variable['sortedID'] = range(len(df_one_variable))
            df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
            df_one_variable.crs = "EPSG:3395"

            adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable_name)
            adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
            county_list = adjacent_counties_df['county'].tolist()
            simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

            st = gudhi.SimplexTree()
            st.set_dimension(2)

            for simplex in simplices:
                if len(simplex) == 1:
                    st.insert([simplex[0]], filtration=0.0)
            
            for simplex in simplices:
                if len(simplex) == 2:
                    last_simplex = simplex[-1]
                    filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                    st.insert(simplex, filtration=filtration_value)

            for simplex in simplices:
                if len(simplex) == 3:
                    last_simplex = simplex[-1]
                    filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                    st.insert(simplex, filtration=filtration_value)

            st.compute_persistence()
            persistence = st.persistence()

            print(f'County: {county_stcnty}, Variable: {variable_name}, Census count: {len(df_one_variable)}')


            intervals_dim0 = st.persistence_intervals_in_dimension(0)
            intervals_dim1 = st.persistence_intervals_in_dimension(1)

            print(f'Intervals dim0: {intervals_dim0}')


            
            # get the infinity values count for each dimension
            infinity_dim0 = np.sum(intervals_dim0[:, 1] == np.inf)
            infinity_dim1 = np.sum(intervals_dim1[:, 1] == np.inf)

            # print(f'Infinity count for dimension 0: {infinity_dim0}')
            # print(f'Infinity count for dimension 1: {infinity_dim1}')
            # print(intervals_dim1)

            dim0_count = len(intervals_dim0)
            dim1_count = len(intervals_dim1)


            # remove infinity values
            intervals_dim0_without_inf = intervals_dim0[intervals_dim0[:, 1] != np.inf]

            # CALCULATE total life span for H0
            TL = 0
            for interval in intervals_dim0_without_inf:
                TL += interval[1] - interval[0]

            TML = 0
            for interval in intervals_dim0_without_inf:
                TML += (interval[1] + interval[0])/2

            # df = df.append({'State': state, 'STCNTY': county_stcnty, 'Variable': variable_name, 'Census_count': len(df_one_variable), 'H0_count': dim0_count, 'H1_count': dim1_count, 'H0_inf_count': infinity_dim0, 'H1_inf_count': infinity_dim1}, ignore_index=True)
            new_row = pd.DataFrame([{
                'State': state, 
                'STCNTY': county_stcnty, 
                'Variable': variable_name, 
                'Census_count': len(df_one_variable), 
                'H0_count': dim0_count, 
                'H1_count': dim1_count, 
                'H0_inf_count': infinity_dim0, 
                'H1_inf_count': infinity_dim1,
                'H1_withou_inf_count': dim1_count - infinity_dim1,
                'H0_withou_inf_count': dim0_count - infinity_dim0,
                'census_count': census_count,
                'Total_life_span_H0': TL,
                'Total_mid_life_span_H0': TML
            }])

            df = pd.concat([df, new_row], ignore_index=True)
        break

    return df





# Define the main function

if __name__ == "__main__":
    # Main execution
    base_path = '/home/h6x/git_projects/universal-experiment-lab/experiment_1/outputs'
    # data_path = '/home/h6x/git_projects/ornl-svi-data-processing/processed_data/SVI/2020/SVI2020_MIN_MAX_SCALED_MISSING_REMOVED'

    mort_path = '/home/h6x/git_projects/universal-experiment-lab/experiment_1/data/shape/mortality.gdb'

    # Read the mortality data
    mort = gpd.read_file(mort_path)

    # drop all the columns that stats with EP
    mort = mort.drop(mort.filter(regex='EP').columns, axis=1)

    # print(mort.head(3))

    states = mort['ST_ABB'].unique().tolist()

    #drop 'DC' from the list
    states.remove('DC')

    # state = 'TN'

    selected_variables = ['MOR_14']

    selected_variables_with_censusinfo = ['STCNTY'] + selected_variables + ['geometry']

    # create empty df
    df = pd.DataFrame(columns=['State', 'STCNTY','Variable','Census_count', 'H0_count', 'H1_count', 'H0_inf_count', 'H1_inf_count', 'H1_withou_inf_count', 'H0_withou_inf_count', 'census_count', 'Total_life_span_H0', 'Total_mid_life_span_H0'])

    state = 'TN'

    df = process_state(state, selected_variables, selected_variables_with_censusinfo,df,mort)

    print(f'Sum Total Life Span H0: {df["Total_life_span_H0"].sum()}')



    # for state in tqdm(states, desc="Processing states"):

    #     df = process_state(state, selected_variables, selected_variables_with_censusinfo,df,mort)

    # # save the df to a csv file in base path
    # df.to_csv(f'{base_path}/census_complex_info_mortality_simplex.csv', index=False)
    # print('All states processed.')







    # states = get_folders(data_path)

    # selected_variables = [
    #      'EP_POV150','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
    #     'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    # ]

    # selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + selected_variables + ['geometry']

    # # state ='TN'

    # # create empty df
    # df = pd.DataFrame(columns=['State', 'STCNTY','Variable','Census_count', 'H0_count', 'H1_count', 'H0_inf_count', 'H1_inf_count'])

    # for state in tqdm(states, desc="Processing states"):

    #     df = process_state(state, selected_variables, selected_variables_with_censusinfo,df)


    # # save the df to a csv file in base path
    # df.to_csv(f'{base_path}/census_complex_info_mortality_simplex.csv', index=False)
    # print('All states processed.')
