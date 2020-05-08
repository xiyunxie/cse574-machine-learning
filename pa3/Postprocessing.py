
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#   Accuracy
# Chosen Secondary Optimization Metric: #
#######################################################################################################################

import numpy as np
import math
from utils import *
import operator
########################################################
#help functions
def in_percentile(perc,rates):
    max = np.amax(rates)
    for i in range(len(rates)-1):
        for j in range(i+1,len(rates)):
            x = rates[i]
            y = rates[j]
            quotient = x - y
            if(not (quotient <= 1+perc and quotient >= 1-perc)):
                return False
    return True

def three_in_range(perc,race_rates):
    race_rates = sorted(race_rates.items(), key=operator.itemgetter(1))
    keys = []
    rate_list = []
    max = 0
    for k in range(len(race_rates)):
        if(race_rates[k][1]>max):
            max = race_rates[k][1]
        keys.append(race_rates[k][0])
        rate_list.append(race_rates[k][1])

    rate_index = []
    range_count = 0
    for i in range(len(keys)-2):
        num0 = rate_list[i]
        num1 = rate_list[i+1]
        num2 = rate_list[i+2]
        diff01 = abs(num0-num1)
        diff12 = abs(num1-num2)
        diff02 = abs(num0-num2)
        if(diff01 <= max * perc and diff12 <= max * perc and diff02 <= max * perc):
            mean_in_range = (num0+num1+num2)/3
            return True,[keys[i],keys[i+1],keys[i+2]],mean_in_range
    return False,[],0



def positive_prediction_rate_with_threshold(race,threshold,demographic_parity_data):
    predicted_positives = 0
    for pair in demographic_parity_data[race]:
        prediction = pair[0]
        if prediction >= threshold:
            predicted_positives += 1
    prob = predicted_positives / len(demographic_parity_data[race])
    return prob

def false_negative_rate_with_threshold(race,threshold,prediction_label_pairs):
    false_negatives = 0
    labelled_positives = 0

    for pair in prediction_label_pairs[race]:
        prediction = pair[0]
        label = pair[1]
        if label == 1:
            labelled_positives += 1
            if prediction < threshold:
                false_negatives += 1

    if labelled_positives != 0:
        return false_negatives / labelled_positives
    else:
        return 0

def true_positive_rate_with_threshold(race,threshold,prediction_label_pairs):
    return 1-false_negative_rate_with_threshold(race,threshold,prediction_label_pairs)


def positive_predictive_rate_with_threshold(race,threshold,prediction_label_pairs):
    true_positive = 0
    predicted_positive = 0

    for pair in prediction_label_pairs[race]:
        prediction = pair[0]
        label = pair[1]
        if prediction > threshold and label == 1:
            true_positive += 1
        if prediction > threshold :
            predicted_positive += 1


    if predicted_positive != 0:
        return true_positive / predicted_positive
    else:
        return 0


def accuracy_with_threshold(race,threshold,demographic_parity_data):
    num_correct = 0
    for pair in demographic_parity_data[race]:
        prediction = pair[0]
        label = pair[1]
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
        race_data = categorical_results[race]
        for data in race_data:
            if(data[0]>threshold):
                race_parity_row.append((1,data[1]))
            else:
                race_parity_row.append((0,data[1]))
        parity_data[race] = race_parity_row
    return parity_data
########################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
########################################################
def enforce_demographic_parity(categorical_results, epsilon):
    # return None,None
    thresholds = {}
    all_predictions = {}
    min_race = {}
    max_race = {}
    mean_of_one = {}
    for race in categorical_results.keys():
        data_for_a_race = categorical_results[race]
        arr = []
        min = 2
        max = -2
        total_prediction_of_one_prediction = 0
        one_prediction_count = 0
        for data in data_for_a_race:
            arr.append(data[0])
            if (data[0] < min):
                min = data[0]
            if (data[0] > max):
                max = data[0]
            if (data[1] == 1):
                total_prediction_of_one_prediction += data[0]
                one_prediction_count += 1
        mean_of_one[race] = total_prediction_of_one_prediction / one_prediction_count
        min_race[race] = min
        max_race[race] = max
        all_predictions[race] = arr
        thresholds[race] = mean_of_one[race]

    counter = 0
    while True:
        counter +=1
        if(counter>10000):
            break
        race_pp_rate = []
        rate_map = {}
        for race in categorical_results.keys():
            pp_rate = positive_prediction_rate_with_threshold(race,thresholds[race],categorical_results)
            rate_map[race] = pp_rate
            race_pp_rate.append(pp_rate)
        if(in_percentile(epsilon,race_pp_rate)):
            break
        else:
            # r_map = {'African-American': 0.29560153709725095, 'Caucasian': 0.29571625978811605, 'Hispanic': 0.29936619718309857, 'Other': 0.24404761904761904}
            has_three_in_range,in_range_races,mean_in_range = three_in_range(epsilon,rate_map)
            if(has_three_in_range):
                for race_of_rate in rate_map.keys():
                    if(not race_of_rate in in_range_races):
                        if(rate_map[race_of_rate]>mean_in_range):
                            thresholds[race_of_rate] += 0.0005
                        else:
                            thresholds[race_of_rate] -= 0.0005

            else:

                acc_map = {}
                for race in categorical_results.keys():
                    acc_of_race = accuracy_with_threshold(race,thresholds[race],categorical_results)
                    acc_map[race] = acc_of_race
                sort_map = sorted(acc_map.items(), key=lambda x: x[1])
                sort_key = list(sort_map)
                median_race = sort_key[len(sort_key) // 2][0]
                median = rate_map[median_race]
                # median = rate_map[sort_key[len(sort_key)//2][0]]
                # mean = np.mean(race_pp_rate)
                for race_of_rate in rate_map.keys():
                    if(not race_of_rate==median_race):
                        if(rate_map[race_of_rate]<median):
                            thresholds[race_of_rate] -= 0.005

                        else:
                            thresholds[race_of_rate] += 0.005

    demographic_parity_data = get_fairness_by_threshold(thresholds,categorical_results)
    #return demographic_parity_data, thresholds
    return demographic_parity_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    # return None,None
    thresholds = {}
    equal_opportunity_data = {}
    min_race = {}
    max_race = {}
    mean_of_one = {}
    for race in categorical_results.keys():
        data_for_a_race = categorical_results[race]
        arr = []
        min = 2
        max = -2
        total_prediction_of_one_prediction = 0
        one_prediction_count = 0
        for data in data_for_a_race:
            arr.append(data[0])
            if (data[0] < min):
                min = data[0]
            if (data[0] > max):
                max = data[0]
            if (data[1] == 1):
                total_prediction_of_one_prediction += data[0]
                one_prediction_count += 1
        mean_of_one[race] = total_prediction_of_one_prediction / one_prediction_count
        min_race[race] = min
        max_race[race] = max
        thresholds[race] = mean_of_one[race]

    counter = 0
    while True:
        counter += 1
        if (counter > 10000):
            break
        race_tp_rate = []
        rate_map = {}
        for race in categorical_results.keys():
            tp_rate = true_positive_rate_with_threshold(race, thresholds[race], categorical_results)
            rate_map[race] = tp_rate
            race_tp_rate.append(tp_rate)

        if (in_percentile(epsilon, race_tp_rate)):
            break
        else:
        # r_map = {'African-American': 0.29560153709725095, 'Caucasian': 0.29571625978811605, 'Hispanic': 0.29936619718309857, 'Other': 0.24404761904761904}
            has_three_in_range, in_range_races, mean_in_range = three_in_range(epsilon, rate_map)
            if (has_three_in_range):
                for race_of_rate in rate_map.keys():
                    if (not race_of_rate in in_range_races):
                        if (rate_map[race_of_rate] > mean_in_range):
                            thresholds[race_of_rate] += 0.0005
                        else:
                            thresholds[race_of_rate] -= 0.0005

            else:
                acc_map = {}
                for race in categorical_results.keys():
                    acc_of_race = accuracy_with_threshold(race, thresholds[race], categorical_results)
                    acc_map[race] = acc_of_race
                sort_map = sorted(acc_map.items(), key=lambda x: x[1])
                sort_key = list(sort_map)
                median_race = sort_key[len(sort_key) // 2][0]
                median = rate_map[median_race]
                # median = rate_map[sort_key[len(sort_key) // 2][0]]
                # mean = np.mean(race_pp_rate)
                for race_of_rate in rate_map.keys():
                    if (not race_of_rate == median_race):
                        if (rate_map[race_of_rate] < median):
                            thresholds[race_of_rate] -= 0.005

                        else:
                            thresholds[race_of_rate] += 0.005

    # Must complete this function!
    #return equal_opportunity_data, thresholds
    equal_opportunity_data = get_fairness_by_threshold(thresholds,categorical_results)
    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    # return None,None
    mp_data = {}
    thresholds = {}
    max_race_accuracy = {}
    for race in categorical_results:
        threshold = 0
        max_accuracy = 0
        max_threshold = 0
        race_data = categorical_results[race]

        while(threshold<1):
            accuracy = accuracy_with_threshold(race,threshold,categorical_results)
            if(accuracy>max_accuracy):
                max_accuracy = accuracy
                max_threshold = threshold
            threshold += 0.0001
        max_race_accuracy[race] = max_accuracy
        thresholds[race] = max_threshold
    # Must complete this function!
    #return mp_data, thresholds
    mp_data = get_fairness_by_threshold(thresholds, categorical_results)
    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    # return None,None
    all_predictions = {}
    min_race = {}
    max_race = {}
    mean_of_one = {}
    for race in categorical_results.keys():
        data_for_a_race = categorical_results[race]
        arr = []
        min = 2
        max = -2
        total_prediction_of_one_prediction = 0
        one_prediction_count = 0
        for data in data_for_a_race:
            arr.append(data[0])
            if (data[0] < min):
                min = data[0]
            if (data[0] > max):
                max = data[0]
            if (data[1] == 1):
                total_prediction_of_one_prediction += data[0]
                one_prediction_count += 1
        mean_of_one[race] = total_prediction_of_one_prediction / one_prediction_count
        min_race[race] = min
        max_race[race] = max
        all_predictions[race] = arr
        thresholds[race] = mean_of_one[race]

    counter = 0
    while True:
        counter += 1
        if (counter > 10000):
            break
        race_pp_rate = []
        rate_map = {}
        for race in categorical_results.keys():
            pp_rate = positive_predictive_rate_with_threshold(race, thresholds[race], categorical_results)
            rate_map[race] = pp_rate
            race_pp_rate.append(pp_rate)
        if (in_percentile(epsilon, race_pp_rate)):
            break
        else:
            # r_map = {'African-American': 0.29560153709725095, 'Caucasian': 0.29571625978811605, 'Hispanic': 0.29936619718309857, 'Other': 0.24404761904761904}
            has_three_in_range, in_range_races, mean_in_range = three_in_range(epsilon, rate_map)
            if (has_three_in_range):
                a = 1+1
                for race_of_rate in rate_map.keys():
                    if (not race_of_rate in in_range_races):
                        if (rate_map[race_of_rate] > mean_in_range):
                            thresholds[race_of_rate] -= 0.0002
                        else:
                            thresholds[race_of_rate] += 0.0002

            else:
                acc_map = {}
                for race in categorical_results.keys():
                    acc_of_race = accuracy_with_threshold(race, thresholds[race], categorical_results)
                    acc_map[race] = acc_of_race
                sort_map = sorted(acc_map.items(), key=lambda x: x[1])
                sort_key = list(sort_map)
                median_race = sort_key[len(sort_key) // 2][0]
                median = rate_map[median_race]
                # mean = np.mean(race_pp_rate)
                for race_of_rate in rate_map.keys():
                    if (not race_of_rate == median_race):
                        if (rate_map[race_of_rate] < median):
                            thresholds[race_of_rate] += 0.0005

                        else:
                            thresholds[race_of_rate] -= 0.0005
    predictive_parity_data = get_fairness_by_threshold(thresholds, categorical_results)
    # Must complete this function!
    #return predictive_parity_data, thresholds

    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}
    max_threshold = 0
    threshold = 0
    accuracy_sum_max = 0
    while (threshold < 1):
        accuracy_sum = 0
        for race in categorical_results.keys():
            accuracy = accuracy_with_threshold(race, threshold, categorical_results)
            accuracy_sum += accuracy
        if (accuracy_sum > accuracy_sum_max):
            accuracy_sum_max = accuracy_sum
            max_threshold = threshold
        threshold += 0.0001
    for race in categorical_results.keys():
        thresholds[race] = max_threshold
    # Must complete this function!
    #return single_threshold_data, thresholds
    single_threshold_data = get_fairness_by_threshold(thresholds, categorical_results)
    return single_threshold_data, thresholds