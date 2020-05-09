from sklearn.naive_bayes import MultinomialNB
from Preprocessing import preprocess
from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

data = np.concatenate((training_data, test_data))
labels = np.concatenate((training_labels, test_labels))
total_data_class_prediction = NBC.predict_proba(data)
total_prediction = []

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(labels)):
    total_prediction.append(total_data_class_prediction[i][1])

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

total_race_cases = get_cases_by_metric(data, categories, "race", mappings, total_prediction, labels)
training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, 0.02)
print("")
ROC_curves = []
for group in test_race_cases.keys():
    ROC_data = get_ROC_data(training_race_cases[group],group)
    ROC_curves.append(ROC_data)

plot_ROC_data(ROC_curves)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])



# report_results(test_race_cases)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

for group in total_race_cases.keys():
    total_race_cases[group] = apply_threshold(total_race_cases[group], thresholds[group])

print("")
for group in test_race_cases.keys():
    accuracy = get_num_correct(test_race_cases[group]) / len(test_race_cases[group])
    print("Accuracy for " + group + ": " + str(accuracy))

print("")
for group in test_race_cases.keys():
    num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
    prob = num_positive_predictions / len(test_race_cases[group])
    print("Probability of positive prediction for " + str(group) + ": " + str(prob))

print("")
for group in test_race_cases.keys():
    PPV = get_positive_predictive_value(test_race_cases[group])
    print("PPV for " + group + ": " + str(PPV))

print("")
for group in test_race_cases.keys():
    FPR = get_false_positive_rate(test_race_cases[group])
    print("FPR for " + group + ": " + str(FPR))

print("")
for group in test_race_cases.keys():
    FNR = get_false_negative_rate(test_race_cases[group])
    print("FNR for " + group + ": " + str(FNR))

print("")
for group in test_race_cases.keys():
    TPR = get_true_positive_rate(test_race_cases[group])
    print("TPR for " + group + ": " + str(TPR))

print("")
for group in test_race_cases.keys():
    TNR = get_true_negative_rate(test_race_cases[group])
    print("TNR for " + group + ": " + str(TNR))

print("")
for group in test_race_cases.keys():
    print("Threshold for " + group + ": " + str(thresholds[group]))

print("")
for group in test_race_cases.keys():
    score = calculate_Fscore(test_race_cases[group])
    print("Score for " + group + ": " + str(score))



print("")
print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("")
print("Accuracy on test data:")
print(get_total_accuracy(test_race_cases))

print("")
print("Accuracy on whole data:")
print(get_total_accuracy(total_race_cases))

print("")
print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("")
print("Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")

print("")
print("Cost on all data:")
print('${:,.0f}'.format(apply_financials(total_race_cases)))
print("")
