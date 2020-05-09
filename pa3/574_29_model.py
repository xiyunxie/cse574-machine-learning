from numpy.random import seed
seed(4940)
from tensorflow import set_random_seed
set_random_seed(80)

import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from Preprocessing import preprocess
from Report_Results import report_results
from utils import *
from Postprocessing import *

metrics = ["sex", "age_cat", "race", 'c_charge_degree', 'priors_count']

training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

activation = "relu"
model = Sequential()
model.add(Dense(len(metrics)*2, activation=activation, kernel_regularizer=regularizers.l2(0.1), input_shape = (len(metrics), )))
model.add(Dense(30, activation=activation, kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss="binary_crossentropy")
model.fit(training_data, training_labels, epochs=30, batch_size=300, validation_data=(test_data, test_labels), verbose=0)

data = np.concatenate((training_data, test_data))
labels = np.concatenate((training_labels, test_labels))

predictions = model.predict(data)
predictions = np.squeeze(predictions, axis=1)

training_predictions = model.predict(training_data)
training_predictions = np.squeeze(training_predictions, axis=1)

test_predictions = model.predict(test_data)
test_predictions = np.squeeze(test_predictions, axis=1)


total_race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)
training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

print("")
# ROC_curves = []
# for group in test_race_cases.keys():
#     ROC_data = get_ROC_data(total_race_cases[group],group)
#     ROC_curves.append(ROC_data)
#
# plot_ROC_data(ROC_curves)

training_race_cases, thresholds = enforce_maximum_profit(training_race_cases)

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
