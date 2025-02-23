# import libraries
import numpy as np
import pandas as pd
import geopandas as gpd
import spatial_tda as tda
import random



def one_sample_permutaion(geo_dataframe,variable, filter_method, n_permutation, random_seed=None):
    """
    This function performs one sample permutation test on the data.
    
    Parameters:
    data: a pandas dataframe
    n_permutation: an integer
    
    Returns:
    p_value: a float
    """

    # for the original - observed data

    adjacency_simplex = tda.AdjacencySimplex(geo_dataframe=geo_dataframe, variable = variable, threshold=None, filter_method=filter_method)
    adjacency_simplex.filter_sort_gdf()
    adjacency_simplex.calculate_adjacent_countries()
    adjacency_simplex.form_simplicial_complex()
    oberved = adjacency_simplex.compute_persistence()
    observed_tl = oberved['TL']

    oberved_values = geo_dataframe[variable].values
    permuted_tls = []

    if random_seed:
        print('random seed is set')
        random.seed(random_seed)

    # loop through the number of permutation
    for i in range(n_permutation):

        #shuffle the data
        geo_dataframe[variable] = np.random.shuffle(oberved_values)

        # for the permuted data
        adjacency_simplex = tda.AdjacencySimplex(geo_dataframe=geo_dataframe, variable = variable, threshold=None, filter_method=filter_method)
        adjacency_simplex.filter_sort_gdf()
        adjacency_simplex.calculate_adjacent_countries()
        adjacency_simplex.form_simplicial_complex()
        permuted = adjacency_simplex.compute_persistence()
        permuted_tl = permuted['TL']

        # append the permuted mean difference to the list
        permuted_tls.append(permuted_tl)


    # calculate the p-value

    # mean of the permuted tls
    permuted_mean_tl = np.mean(permuted_tls)

    # mean tl - observed tl
    mean_observed_diff = permuted_mean_tl - observed_tl

    # let's calculate the number of permuted tls that are greater than or equal to the observed tl
    count = 0
    for permuted_tl in permuted_tls:

        mean_permuted_diff = permuted_mean_tl - permuted_tl

        if mean_permuted_diff >= mean_observed_diff:
            count += 1
    
    p_value = (count+1) / (n_permutation+1)

    return p_value
