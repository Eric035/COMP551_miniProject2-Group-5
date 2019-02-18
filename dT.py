#from pathlib import Path
import io
import os, sys
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as pp
import csv
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


posDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/pos"
negDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/neg"
testDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/test"

posAdjString = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,best,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,enjoyable,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,phenomenal,always,also,both,hit,fun,zesty"
negAdjString = "absurd,arrogant,boring,bad,intolerant,crazy,miserly,patronizing,vulgar,crude,obnoxious,offensive,violent,cryptic,failure,fail,cringy,atrocious,awful,cheap,crummy,dreadful,lousy,noisy,poor,poorly,unacceptable,garbage,gross,horrible,inaccurate,inferior,obnoxious,synthetic,careless,cheesy,crappy,abominable,faulty,godawful,substandard,despicable,horrendous,terrible,attempt,upsetting,not,no,vile,abominable,appalling,cruel,disgusting,dreadful,eerie,grim,hideous,disastrous,disaster,horrid,horrendous,any,can't,because,better,anything,unpleasant,defective,miserable,failed,unsatisfied"

#--------------------------------------------------------------------------------------------------------------#
# Decision Tree classifier
print("We are training our Decision Tree model with our training set...")
def adjStringToDict (adjString):            # Covert a string of adjectives into a dictionary
    adjList = adjString.split(',')
    adjDict = {}
    for adjective in adjList:
        adjDict[adjective] = 0
    return adjDict

posAdjDict = adjStringToDict(posAdjString)                # A dictionary object that contains all the positive adjectives
negAdjDict = adjStringToDict(negAdjString)

posReviews = os.listdir(posDirectory)
negReviews = os.listdir(negDirectory)                       # Files in the directory neg
numPosReviewsTrainingData = len(os.listdir(posDirectory))
numNegReviewsTrainingData = len(os.listdir(negDirectory))
totalNumReviewsInTrainingData = (numNegReviewsTrainingData + numPosReviewsTrainingData)

trainingDataDT = np.zeros((totalNumReviewsInTrainingData, 2))       # A numpy matrix to store our result to fit our DT model
                                                                    # Columns: 1) Number of positive adjectives a review has, 2) Number of negative adjectives a review has, 3) Number of swear words a review has

posReviewsDirectoryPath = os.path.normpath(posDirectory)
negReviewsDirectoryPath = os.path.normpath(negDirectory)

for file in posReviews:                                 # Loop through each file in the pos directory to organize data for our DT model
    filepath = os.path.join(posReviewsDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    reviewLen = len(content)            # Count the length of that particular review
    wordList = (content.lower()).split()
    try:
        underScoreIndex = file.index('_')           # Take the part of the string before the under score symbol, we will have the file's order
        fileOrder = int(file[:underScoreIndex])
        for w in wordList:
            if w in posAdjDict:
                trainingDataDT[fileOrder][0] += 1
            if w in negAdjDict:
                trainingDataDT[fileOrder][1] += 1
    except ValueError:                              # There is a .DS_store file we have catch in order to prevent our program from crashing
        continue

for file in negReviews:
    filepath = os.path.join(negReviewsDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    reviewLen = len(content)            # Count the length of that particular review
    wordList = (content.lower()).split()
    try:
        underScoreIndex = file.index('_')
        fileOrder = int(file[:underScoreIndex]) + int(len(posReviews))  # File number is obtained by taking the string the under score.
        for w in wordList:
            if w in posAdjDict:
                trainingDataDT[fileOrder][0] += 1
            if w in negAdjDict:
                trainingDataDT[fileOrder][1] += 1
    except ValueError:
        continue

print("This is the input data for our DT model: ")
print(trainingDataDT)
print("#--------------------------------------------------------------------------------------------------------------#")
trainTarget = [0] * totalNumReviewsInTrainingData
for i in range(len(posReviews)):
    trainTarget[i] = 1
print("And this is the target parameter for our DT model: ")
print (trainTarget)
print("#--------------------------------------------------------------------------------------------------------------#")
model = tree.DecisionTreeClassifier()
model.fit(trainingDataDT, trainTarget)
# DF = pd.DataFrame(trainingDataDT, columns=['Positive Adj', 'Negative Adj'])
# DF.to_csv('DT_training.csv', encoding='utf-8', index=False)

#--------------------------------------------------------------------------------------------------------------#
# For our test data
print("Now we are going to extract the features from our test data set, and prepare it for our DT model: ")
testSetDirectoryPath = os.path.normpath(testDirectory)
filesInTest = os.listdir(testDirectory)

numReviewsTestData = len(os.listdir(testDirectory))
testDataDT = np.zeros((numReviewsTestData, 2))

for file in filesInTest:
    filepath = os.path.join(testSetDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    reviewLen = len(content)  # Count the length of that particular review, for extra feature
    wordList = (content.lower()).split()
    index = int(file[:-4])              # Take away the '.txt' from file (string), we will have our index for our matrix
    for w in wordList:
        if w in posAdjDict:
            testDataDT[index][0] += 1
        if w in negAdjDict:
            testDataDT[index][1] += 1

print("This is the input data after pre-processing for our test set: ")
print(testDataDT)
print("#--------------------------------------------------------------------------------------------------------------#")
print("Predicting...")
print('')
print("Result: ")
testSetPrediction= model.predict(testDataDT)
print(testSetPrediction)

pCounter = 0
nCounter = 0
csv_testSetPrediction = np.zeros((numReviewsTestData, 2))   # A matrix that stores our predicted values for our csv data frame.

for i in range(numReviewsTestData):
    if testSetPrediction[i] == 1:
        pCounter += 1
        csv_testSetPrediction[i][0] = int(i)
        csv_testSetPrediction[i][1] = int(1)
    else:
        nCounter += 1
        csv_testSetPrediction[i][0] = int(i)
        csv_testSetPrediction[i][1] = int(0)

posReviewPercentage = (pCounter / numReviewsTestData) * 100
negReviewPercentage = (nCounter / numReviewsTestData) * 100
print("In our test data set, ", posReviewPercentage, "% are predicted as positive reviews.")
print("And ", negReviewPercentage, "% are predicted as negative reviews.")

print("--------------------------------------------------------------------------------------------------------------")
print("CSV Data frame: ")
dataFrame = pd.DataFrame(csv_testSetPrediction, columns=["Id", "Category"])
dataFrame.Id = dataFrame.Id.astype(int)
dataFrame.Category = dataFrame.Category.astype(int)
print(dataFrame)


# print("Random prediction: ", model.predict(([[1000, 5]])))
dataFrame.to_csv('dTPredictedValues.csv', encoding='utf-8', index=False)
#--------------------------------------------------------------------------------------------------------------#