import os, sys
import csv
import numpy as np
import pandas as pd
import pickle as pk
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


posDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/pos"
negDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/neg"
testDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/test"

newfile = 'strongFeatures.pk'
with open(newfile, 'rb') as fi:
    strongFeatures = pk.load(fi)
wordsInDF = strongFeatures

newfile = 'df.pk'
with open(newfile, 'rb') as fi:
    df = pk.load(fi)

#--------------------------------------------------------------------------------------------------------------#
# Support Vector Machines Classifier
print("We are training our SVM model with our training set...")

posReviews = os.listdir(posDirectory)
negReviews = os.listdir(negDirectory)                           # Files in the directory neg
numPosReviewsTrainingData = len(os.listdir(posDirectory))
numNegReviewsTrainingData = len(os.listdir(negDirectory))
totalNumReviewsInTrainingData = (numNegReviewsTrainingData + numPosReviewsTrainingData)

trainingDataSVM = np.ones((totalNumReviewsInTrainingData, len(df)))       # A numpy matrix to store our result to fit our SVM model
                                                                    # Columns: 1) Number of positive adjectives a review has, 2) Number of negative adjectives a review has, 3) Number of swear words a review has
posReviewsDirectoryPath = os.path.normpath(posDirectory)
negReviewsDirectoryPath = os.path.normpath(negDirectory)

for file in posReviews:                                 # Loop through each file in the pos directory to organize data for our model
    filepath = os.path.join(posReviewsDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    wordList = (content.lower()).split()
    try:
        underScoreIndex = file.index('_')           # Take the part of the string before the under score symbol, we will have the file's order
        fileOrder = int(file[:underScoreIndex])
        for w in wordList:
            if w in wordsInDF:
                wIndex = wordsInDF.index(w)         # Getting where the word should be in the matrix
                trainingDataSVM[fileOrder][wIndex] += 1 + (np.log(25000) / (1 + df[w]))
    except ValueError:                              # There is a .DS_store file we have catch in order to prevent our program from crashing
        continue

for file in negReviews:
    filepath = os.path.join(negReviewsDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    wordList = (content.lower()).split()
    try:
        underScoreIndex = file.index('_')
        fileOrder = int(file[:underScoreIndex]) + int(len(posReviews))  # File number is obtained by taking the string the under score.
        for w in wordList:
            if w in wordsInDF:
                wIndex = wordsInDF.index(w) # Getting where the word should be in the matrix
                trainingDataSVM[fileOrder][wIndex] += 1 + (np.log(25000) / (1 + df[w]))
    except ValueError:
        continue

print("This is the input data for our Logistics Regression model: ")
print(trainingDataSVM)
print("#--------------------------------------------------------------------------------------------------------------#")
trainTarget = [0] * totalNumReviewsInTrainingData
for i in range(len(posReviews)):
    trainTarget[i] = 1
print("And this is the target parameter for our SVM model: ")
print (trainTarget)
print("#--------------------------------------------------------------------------------------------------------------#")

X_train, X_test, y_train, y_test = train_test_split(trainingDataSVM, trainTarget, test_size=0.33)

print("X train : ", len(X_train))
print("Y train : ",len(y_train))
print("X test : ", len(X_test))
print("Y test : ", len(y_test))

model = svm.SVC()
model.fit(X_train,y_train)

testSetPrediction = model.predict(X_test)

pCounter = 0
nCounter = 0
for i in range(0,len(y_test)):
    if testSetPrediction[i] == 1 and y_test[i] == 1 or testSetPrediction[i] == 0 and y_test[i] == 0:
        pCounter += 1
    else:
        nCounter += 1

posPredictedReviews = (pCounter / len(y_test)) * 100
negPredictedReviews = (nCounter / len(y_test)) * 100
print("In our test data set, ", posPredictedReviews, "% are predicted right.")
print("And ", negPredictedReviews, "% are predicted wrong.")

#--------------------------------------------------------------------------------------------------------------#
print("#--------------------------------------------------------------------------------------------------------------#")
# K-Fold Cross Validation
print("Running K-Fold cross validation...")

'''
kf = KFold(n_splits=5)          # Use 5 folds on the dataset
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):       # An example to show how K-fold works
    print(train_index, test_index)
'''

folds = StratifiedKFold(n_splits=5)

def get_score (model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

