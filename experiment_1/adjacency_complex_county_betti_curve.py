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

def process_state(state, selected_variable, selected_variables_with_censusinfo,df,mort_df):
    """Process data for a given state."""
    # svi_od_path = os.path.join(data_path, state, state + '.shp')
    # svi_od = gpd.read_file(svi_od_path)
    # # for variable in selected_variables:
    #     # svi_od = svi_od[svi_od[variable] != -999]

    mort_df = mort_df[mort_df['ST_ABB'] == state]

        
    mort_df_filtered_state = mort_df[selected_variables_with_censusinfo].reset_index(drop=True)
    
    variable_name = selected_variable[0]

    r_vals = np.linspace(0, 100, 100) 

    for r_val in r_vals:

        temp_df = mort_df_filtered_state[mort_df_filtered_state[variable_name] <= r_val]
        df_one_variable = temp_df[['STCNTY', variable_name, 'geometry']]
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

        intervals_dim0 = st.persistence_intervals_in_dimension(0)
        intervals_dim1 = st.persistence_intervals_in_dimension(1)
        
        # get the infinity values count for each dimension
        infinity_dim0 = np.sum(intervals_dim0[:, 1] == np.inf)
        infinity_dim1 = np.sum(intervals_dim1[:, 1] == np.inf)

        # remove infinity values
        intervals_dim0_without_inf = intervals_dim0[intervals_dim0[:, 1] != np.inf]
        intervals_dim1_without_inf = intervals_dim1[intervals_dim1[:, 1] != np.inf]


        # print('Length of intervals_dim0:', len(intervals_dim0))

        # CALCULATE total life span for H0
        TL = 0
        for interval in intervals_dim0_without_inf:
            TL += interval[1] - interval[0]


        
        
        # # print('Total life span for H0:', TL)

        TML = 0
        for interval in intervals_dim0_without_inf:
            TML += (interval[1] + interval[0])/2

        # AL = TL/len(intervals_dim0)
        # AML = TML/len(intervals_dim0)

        # break

        # # print(f'Infinity count for dimension 0: {infinity_dim0}')
        # # print(f'Infinity count for dimension 1: {infinity_dim1}')
        # # print(intervals_dim1)

        # dim0_count = len(intervals_dim0)
        # dim1_count = len(intervals_dim1)

        # # df = df.append({'State': state, 'STCNTY': county_stcnty, 'Variable': variable_name, 'Census_count': len(df_one_variable), 'H0_count': dim0_count, 'H1_count': dim1_count, 'H0_inf_count': infinity_dim0, 'H1_inf_count': infinity_dim1}, ignore_index=True)
        # new_row = pd.DataFrame([{
        #     'State': state, 
        #     # 'STCNTY': county_stcnty, 
        #     'Variable': variable_name, 
        #     'Census_count': len(df_one_variable), 
        #     'H0_count': dim0_count, 
        #     'H1_count': dim1_count, 
        #     'H0_inf_count': infinity_dim0, 
        #     'H1_inf_count': infinity_dim1,
        #     'H1_withou_inf_count': dim1_count - infinity_dim1,
        #     'H0_withou_inf_count': dim0_count - infinity_dim0,
        #     # 'census_count': census_count
        # }])

        new_row = pd.DataFrame([{
            'State': state, 
            'Variable': variable_name, 
            'Census_count': len(df_one_variable), 
            'H0_count': len(intervals_dim0), 
            'H1_count': len(intervals_dim1), 
            'H0_inf_count': infinity_dim0, 
            'H1_inf_count': infinity_dim1,
            'H1_withou_inf_count': len(intervals_dim1_without_inf),
            'H0_withou_inf_count': len(intervals_dim0_without_inf),
            'Total_life_span_H0': TL,
            'Total_mid_life_span_H0': TML,
            'r_val': r_val
        }])

        df = pd.concat([df, new_row], ignore_index=True)

    # return 
    return df





# Define the main function

if __name__ == "__main__":
    # Main execution
    base_path = '/home/h6x/git_projects/universal-experiment-lab/experiment_1/data'
    mor_path = '/home/h6x/git_projects/universal-experiment-lab/experiment_1/data/shape/mortality.gdb'

    # Read the csv file making sure the FIPS code is read as a string
    mortality_df = gpd.read_file(mor_path)

    # Get the unique states
    states = mortality_df['ST_ABB'].unique()
    # remove 'DC' from the states
    states = [state for state in states if state != 'DC']

    # selected_variables = ['MOR_14','MOR_15','MOR_16','MOR_17','MOR_18','MOR_19','MOR_20','PRIS_20']

    selected_variable = ['PRIS_20']

    selected_variables_with_censusinfo = ['STCNTY'] + selected_variable + ['geometry']


    state = 'TN'

    # create empty df
    # df = pd.DataFrame(columns=['State', 'STCNTY','Variable','Census_count', 'H0_count', 'H1_count', 'H0_inf_count', 'H1_inf_count', 'H1_withou_inf_count', 'H0_withou_inf_count'])

    df = pd.DataFrame(columns=['State', 'Variable', 'Census_count', 'H0_count', 'H1_count', 'H0_inf_count', 'H1_inf_count', 'H1_withou_inf_count', 'H0_withou_inf_count','Total_life_span_H0','Total_mid_life_span_H0', 'r_val'])

    df = process_state(state, selected_variable, selected_variables_with_censusinfo,df,mortality_df)

    print(df)

    # plot H0_withou_inf_count against r_val
    plt.plot(df['r_val'], df['H0_withou_inf_count'])
    # plt.plot(df['r_val'], df['Total_mid_life_span_H0'])
    plt.xlabel('r_val')
    plt.ylabel('H0_withou_inf_count')
    plt.title('H0_withou_inf_count vs r_val')
    plt.savefig('/home/h6x/git_projects/universal-experiment-lab/experiment_1/plots/H0_withou_inf_count_vs_r_val.png')



    # print the sum of MOR_14 column for state 'TN'
    # print(mortality_df[mortality_df['ST_ABB'] == 'TN']['MOR_14'].sum())

    # print(mortality_df[mortality_df['ST_ABB'] == 'TN']['MOR_14'].count())

    # print(TL)

    # create empty df

    # tl_df = pd.DataFrame(columns=['State', 'Total_life_span', 'MOR_20'])


    # for state in tqdm(states, desc="Processing states"):

    #     TL = process_state(state, selected_variable, selected_variables_with_censusinfo,df,mortality_df)
    #     sum_mor = mortality_df[mortality_df['ST_ABB'] == state]['MOR_18'].sum()

    #     # add the row to the df
    #     new_row = pd.DataFrame([{ 'State': state, 'Total_life_span': TL, 'Sum_of_MOR_20': sum_mor}])
    #     tl_df = pd.concat([tl_df, new_row], ignore_index=True)

    #     print(f'State: {state}, Total life span: {TL}, Sum of MOR_20: {sum_mor}')

    # print(tl_df)

    # # plot the scatter plot
    # plt.scatter(tl_df['Total_life_span'], tl_df['Sum_of_MOR_20'])
    # plt.xlabel('Total life span')
    # plt.ylabel('Sum of MOR_20')
    # plt.title('Total life span vs Sum of MOR_20')
    # plt.savefig('/home/h6x/git_projects/universal-experiment-lab/experiment_1/total_life_span_PRIS_20_vs_mort_20.png')

    
    # states = get_folders(data_path)

    # selected_variables = [
    #      'EP_POV','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
    #     'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    # ]

    # selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + selected_variables + ['geometry']

    # # state ='TN'

    # # create empty df
    # df = pd.DataFrame(columns=['State', 'STCNTY','Variable','Census_count', 'H0_count', 'H1_count', 'H0_inf_count', 'H1_inf_count'])

    # for state in tqdm(states, desc="Processing states"):

    #     df = process_state(state, selected_variables, selected_variables_with_censusinfo,df)


    # # save the df to a csv file in base path
    # df.to_csv(f'{base_path}/census_complex_info.csv', index=False)
    # print('All states processed.')
