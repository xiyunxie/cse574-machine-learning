
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
        if(not abs(diff)<= x*perc and not abs(diff)<=y*perc):
            return False
    return True

def positive_prediction_rate(race,demographic_parity_data):
    num_positive_predictions = get_num_predicted_positives(demographic_parity_data[race])
    prob = num_positive_predictions / len(demographic_parity_data[race])
    return prob

def positive_prediction_rate_with_threshold(race,threshold,demographic_parity_data):
    predicted_positives = 0
    for pair in demographic_parity_data[race]:
        prediction = pair[0]
        if prediction > threshold:
            predicted_positives += 1
    prob = predicted_positives / len(demographic_parity_data[race])
    return prob

def accuracy_with_threshold(race,threshold,demographic_parity_data):
    num_correct = 0
    for pair in demographic_parity_data[race]:
        prediction = int(pair[0])
        label = int(pair[1])
        if(prediction > threshold):
            prediction = 1
        else:
            prediction = 0
        if prediction == label:
            num_correct += 1
    prob = num_correct / len(demographic_parity_data[race])
    return prob

def get_fairness_by_threshold(thresholds,categorical_results):
    parity_data = {}
    for race in categorical_results.keys():
        race_parity_row = []
        threshold = thresholds[race]
        race_data = categorical_results[race_data]
        for data in race_data:
            if(data[0]>threshold):
                race_parity_row.append((1,data[1]))
            else:
                race_parity_row.append((0,data[1]))
        parity_data[race] = race_parity_row
    return parity_data
########################################################
def enforce_demographic_parity(categorical_results, epsilon):

    demographic_parity_data = {}
    thresholds = {}
    race_length = []
    for race in categorical_results.keys():
        length_for_a_race = len(categorical_results[race])
        race_length.append([race,length_for_a_race])
        # thresholds[race] = 0.5
    race_length = sorted(race_length, key=lambda item: item[1])
    all_predictions = {}
    min_race = {}
    max_race = {}
    mean_of_one = {}
    for i in range(len(race_length)):
        race = race_length[i][0]
        data_for_a_race = categorical_results[race]
        arr = []
        min=2
        max=-2

        total_prediction_of_one_prediction = 0
        one_prediction_count = 0
        for data in data_for_a_race:
            arr.append(data[0])
            if(data[0]<min):
                min = data[0]
            if(data[0]>max):
                max = data[0]
            if(data[1]==1):
                total_prediction_of_one_prediction += data[0]
                one_prediction_count += 1
        mean_of_one[race] = total_prediction_of_one_prediction / one_prediction_count
        min_race[race] = min
        max_race[race] = max
        all_predictions[race] = arr
        thresholds[race] = mean_of_one[race]

    print(thresholds)
    counter = 0
    while True:
        counter +=1
        if(counter>100):
            break
        race_pp_rate = []
        rate_map = {}
        for race in categorical_results.keys():
            print(race)
            pp_rate = positive_prediction_rate_with_threshold(race,thresholds[race],categorical_results)
            print(pp_rate)
            rate_map[race] = pp_rate
            race_pp_rate.append(pp_rate)
        if(in_percentile(epsilon,race_pp_rate)):
            break
        else:
            mean = np.mean(race_pp_rate)
            for race_of_rate in rate_map.keys():
                if(rate_map[race_of_rate]<mean):
                    if(not thresholds[race_of_rate]-0.005<min_race[race_of_rate]):
                        thresholds[race_of_rate] -= 0.005
                else:
                    if(not thresholds[race_of_rate]+0.005>max_race[race_of_rate]):
                        thresholds[race_of_rate] += 0.005

    #return demographic_parity_data, thresholds
    # while in_percentile
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