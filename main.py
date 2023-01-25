# from sklearn.linear_model import LogisticRegression # import the model LogisticRegression from group linear_model
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # importing function to split dataset
from sklearn.metrics import precision_score, confusion_matrix, f1_score, roc_curve, auc # calculates precision score, importing functions
import pandas as pd # importing pandas library

dataframe = pd.read_csv("data/spambase.csv") # load the dataset, dataframe is like a table

print(dataframe.head()) # print first 5 rows of dataset
X = dataframe
y = X.pop("class") # split dataset into features and labels, pop removes class column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # training and testing features and labals, calling function to split

classifier = RandomForestClassifier() # creating a model
classifier.fit(X_train, y_train) # training model with training dateset due to split
y_pred = classifier.predict(X_test) # getting predictions for test dataset

confusion = confusion_matrix(y_test, y_pred) # creating the confustion matrix
precision = precision_score(y_test, y_pred) # comparing labels from the dataset vs the model
f1score = f1_score(y_test, y_pred) # calculating recall f1 score
fpr, tpr, _ = roc_curve(y_test, y_pred) #creates roc curve
aucscore = auc(fpr, tpr)
print(f"confusion_matrix: {confusion}") # f for string formatting
print(f"precision_score: {precision}")
print(f"f1_score: {f1score}")
print(f"auc: {aucscore}")

# for index in range(len(X_train.columns)): # for each feature
#     print(X_train.columns[index].ljust(28), classifier.coef_[0][index]) # print feature name and coefficient value

# print(classifier.coef_[0][0])
