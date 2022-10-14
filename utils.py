import random

import numpy as np
import pandas as pd
from scipy.stats import truncnorm


def change_val(val):
    """ Function to add randomness to
        a singular predictor value.
    """
    rv = np.random.rand()
    if rv < 0.35:
        return val - 1
    elif rv > 0.65:
        return val + 1
    else:
        return val


def add_randomness(data):
    """ Function to add randomness to the 
        complete predictor dataset.
    """
    for i, row in data.iterrows():
        for col in row.keys():
            data.loc[i, col] = change_val(data.loc[i, col])
    return data


def compute_target(data, stats):
    """ Function to compute the target values
        based on the predictor values and user
        provided optimum stat values.
    """
    error = truncnorm.rvs(-100, 100, loc=0, scale=1, size=len(data))
    coeffs = np.array(list(stats.values()))
    target = data.values @ coeffs + error
    return 71.347 * target / np.max(target)


def generate_predictors(player_stats, n):
    """ Function to generate a random dataset 
        of predictors
    """
    data = {}
    for name in player_stats.keys():
        data[name] = np.random.chisquare(25, n) / 20
    data = pd.DataFrame.from_dict(data)
    data = data.div(data.sum(axis=1) / 50, axis=0).round()
    for i, row in data.iterrows():
        diff = 50 - sum(row.values)
        if diff != 0:
            index = random.randint(0, len(row.values) - 1)
            data.iloc[i, index] += diff
    return data


def create_dataset(player_stats, n=1000):
    """ Function to create the n length synthetic 
        dataset using the optimum player stats.
    """
    data = generate_predictors(player_stats, n)
    rating = compute_target(data, player_stats)
    data = add_randomness(data)
    data["Rating"] = rating
    return data
