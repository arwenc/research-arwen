import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pandas
from pandas import read_csv
from pandas import DataFrame as df
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from matplotlib import cm
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier


### plotting method ####
def plot_coefficients(classifier, feature_names, top_features=3):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(11, 10))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()


def f_importances(coef, names):
    imp = coef.flatten()
    index = 0
    for i in range(len(imp)):
        temp = imp[i]
        if imp[i] < 0:
            imp[i] = abs(temp)

    print("Weights: ")
    print(imp)
    print('\n')
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()



# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#### Loading the training dataset from csv  #########
filenameX = "training6x.csv"
headers = ["age-group", "tsize_mm", "grade_recode", "chemo", "RS_SEER"]
training_X = read_csv(filenameX, names=headers)
filenameY = "training6y.csv"
header = ["breast_cancer_death"]
training_Y = read_csv(filenameY, names=header)
### Flattening the mortality dataset #####
numpy_trainingY = df.to_numpy(training_Y)
flat_trainingY = np.ndarray.flatten(numpy_trainingY)
### Normalizing Training Data ####
scaler = Normalizer().fit(training_X)
#scaler = StandardScaler().fit(training_X)
normalized_xtrains = scaler.transform(training_X)



#### Loading the validation dataset from csv  #########
filenameX = "validation6x.csv"
headers = ["age-group", "tsize_mm", "grade_recode", "chemo", "RS_SEER"]
validation_X = read_csv(filenameX, names=headers)
filenameY = "validation6y.csv"
header = ["breast_cancer_death"]
validation_Y = read_csv(filenameY, names=header)
### Flattening the mortality dataset #####
numpy_validationY = df.to_numpy(validation_Y)
flat_validationY = np.ndarray.flatten(numpy_validationY)
### Normalizing Training Data ####
#scaler = Normalizer().fit(validation_X)
#scaler = StandardScaler().fit(validation_X)
normalized_xvalidation = scaler.transform(validation_X)


#### Loading the prediction dataset from csv  #########
filenameX = "prediction6x.csv"
headers = ["age-group", "tsize_mm", "grade_recode", "chemo", "RS_SEER"]
prediction_X = read_csv(filenameX, names=headers)
filenameY = "prediction6y.csv"
header = ["breast_cancer_death"]
prediction_Y = read_csv(filenameY, names=header)
### Flattening the mortality dataset #####
numpy_predictionY = df.to_numpy(prediction_Y)
flat_predictionY = np.ndarray.flatten(numpy_predictionY)
### Normalizing Training Data ####
scaler = Normalizer().fit(prediction_X)
#scaler = StandardScaler().fit(prediction_X)
normalized_xprediction = scaler.transform(prediction_X)



framesX = [training_X, validation_X, prediction_X]
combine_X = pandas.concat(framesX)
all_scaler =  Normalizer().fit(combine_X)
X_All = all_scaler.transform(combine_X)
framesy = [training_Y, validation_Y, prediction_Y]
combine_y = pandas.concat(framesy)
# y_All =  np.ndarray.flatten(df.to_numpy(combine_y))

print(X_All)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =  train_test_split(X_All, combine_y, test_size=0.33)

from imblearn.over_sampling import SMOTE

y_train = np.ndarray.flatten(df.to_numpy(y_train))
y_test = np.ndarray.flatten(df.to_numpy(y_test))

print(X_train.shape)

#X_train, y_train = SMOTE().fit_sample(X_train, y_train)
smote_nc = SMOTENC(categorical_features=[2, 3], random_state=0, sampling_strategy='minority')
#X_test, y_test = smote_nc.fit_resample(X_test, y_test)
X_train, y_train = smote_nc.fit_resample(X_train, y_train)


print(X_train.shape)

features_names = headers

bagging = BaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)
balanced_bagging = BalancedBaggingClassifier(n_estimators=50, random_state=0, n_jobs=-1)

bagging.fit(X_train, y_train)
balanced_bagging.fit(X_train, y_train)

y_pred_bc = bagging.predict(X_test)
y_pred_bbc = balanced_bagging.predict(X_test)


# linear_model2 = SVC(kernel='linear', C=100, probability=True)
# # linear_model2.fit(X_train, y_train)

#f_importances(linear_model2.coef_, features_names)


metrics = list()
metrics.append('normalized_mutual_info_score')
metrics.append('accuracy')
metrics.append('roc_auc')


params_dict = {'kernel': ['rbf'], 'C': [1, 10, 150, 1000]}
#svm = SVC(kernel="linear")


scoring = {'AUC': 'roc_auc', "Accuracy": 'accuracy'}

#Gaussian?????????????????????????????
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)



grid_search = GridSearchCV(estimator=svclassifier, param_grid=params_dict, cv=3, scoring=scoring, refit='AUC')
grid_search.fit(X_train, y_train)
#
print("Best parameter values:", grid_search.best_params_)
print("CV Score with best parameter values:", grid_search.best_score_)
#
dframe = pandas.DataFrame(grid_search.cv_results_)

print("\n")

print("Cross validation method 2: Exhaustive Grid-Search")
for key in grid_search.cv_results_:
    print(key + ": ", end=" ")
    print(grid_search.cv_results_[key])



################Finalize####################333
# Capture and fit the best estimator from across the grid search
best_svm = grid_search.best_estimator_
print("\n")
print(best_svm)
#predicted = linear_model2.predict(X_test)
predicted = best_svm.predict(X_test)
final_score = best_svm.score(X_test, y_test)
print("Final AUC Score: " + str(final_score))

from sklearn import metrics

rocauc = metrics.roc_auc_score(y_test, predicted)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
print( "\nGaussian AUC: " +  str(metrics.auc(fpr, tpr)))
print("Gaussian roc_auc_score: " + str(rocauc))
target_names = ["Lived", "Died"]
report = classification_report(y_test, predicted, target_names=target_names, output_dict=True)
for label in report:
    print(label + ": ", end=" ")
    print(report[label])

#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

# cm1 = confusion_matrix(y_test, predicted)
# print('Linear Confusion Matrix : \n', cm1)
#
# total1=sum(sum(cm1))
# #####from confusion matrix calculate accuracy
# accuracy1=(cm1[0,0]+cm1[1,1])/total1
# print ('Accuracy : ', accuracy1)
#
# sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
# print('Sensitivity : ', sensitivity1 )
#
# specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
# print('Specificity : ', specificity1)


print("Gaussian Kernel")
# svclassifier = SVC(kernel='rbf')
# svclassifier.fit(X_train, y_train)
#y_pred = svclassifier.predict(X_test)

rocauc = metrics.roc_auc_score(y_test, predicted)
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
print( "\nGaussian AUC: " +  str(metrics.auc(fpr, tpr)))
print("Gaussian roc_auc_score: " + str(rocauc))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))


#plotting the thing##
plt.title('Receiver Operating Characteristic Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rocauc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test, predicted)
print('Gaussian Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)
