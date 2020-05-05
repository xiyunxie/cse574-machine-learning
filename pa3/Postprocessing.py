
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: #
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
import numpy as np
import math
from utils import *
########################################################
#help functions
def in_percentile(perc,pp_rates):
    for i in range(len(pp_rates)):
        x = pp_rates[i]
        y = pp_rates[(i+1)%len(pp_rates)]
        diff = x-y
        if(not abs(diff)<= x*perc and abs(diff)<=y*perc):
            return False

    return True
def positive_prediction_rate(race,demographic_parity_data):
    num_positive_predictions = get_num_predicted_positives(demographic_parity_data[race])
    prob = num_positive_predictions / len(demographic_parity_data[race])
    return prob
########################################################
def enforce_demographic_parity(categorical_results, epsilon):

    demographic_parity_data = {}
    thresholds = {}
    race_length = []
    for race in categorical_results.keys():
        length_for_a_race = len(categorical_results[race])
        race_length.append([race,length_for_a_race])
        thresholds[race] = 0.5
    race_length = sorted(race_length, key=lambda item: item[1])
    all_predictions = {}
    for i in range(len(race_length)):
        race = race_length[0]
        data_for_a_race = categorical_results[race]
        arr = []
        for i in data_for_a_race:
            arr.append(i[0])
        all_predictions[race] = arr
        if(i< len(race_length)/2):
            thresholds[race] = np.percentile(arr, 75)
        else:
            thresholds[race] = np.percentile(arr, 25)


    for race in categorical_results.keys():
        data_for_a_race = categorical_results[race]
        print(len(data_for_a_race))
        arr = []
        for i in data_for_a_race:
            arr.append(i[0])

        print("25th percentile of arr : ",
              np.percentile(arr, 25))

        print("50th percentile of arr : ",
              np.percentile(arr, 50))

        print("75th percentile of arr : ",
              np.percentile(arr, 75))
        print("=======================")

        # Must complete this function!
    #return demographic_parity_data, thresholds

    return None, None


#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}

    # Must complete this function!
    #return equal_opportunity_data, thresholds

    return None, None

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}

    # Must complete this function!
    #return mp_data, thresholds

    return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    # Must complete this function!
    #return predictive_parity_data, thresholds

    return None, None

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    # Must complete this function!
    #return single_threshold_data, thresholds

    return None, None