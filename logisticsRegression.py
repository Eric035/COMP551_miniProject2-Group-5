from sklearn.linear_model import LogisticRegression
import os, sys
import csv
import numpy as np
import pandas as pd
import pickle as pk

posDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/pos"
negDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/neg"
testDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/test"

# Files loaded

newfile = 'strongFeatures.pk'
with open(newfile, 'rb') as fi:
    strongFeatures = pk.load(fi)

newfile = 'df.pk'
with open(newfile, 'rb') as fi:
    df = pk.load(fi)

newfile = 'posdf.pk'
with open(newfile, 'rb') as fi:
    posdf = pk.load(fi)

newfile = 'negdf.pk'
with open(newfile, 'rb') as fi:
    negdf = pk.load(fi)

#--------------------------------------------------------------------------------------------------------------#
# Logistic Regression Classifier
print("We are training our Logistics Regression model with our training set...")
def adjStringToDict (adjString):            # Covert a string of adjectives into a dictionary
    adjList = adjString.split(',')
    adjDict = {}
    for adjective in adjList:
        adjDict[adjective] = 0
    return adjDict

posReviews = os.listdir(posDirectory)
negReviews = os.listdir(negDirectory)                           # Files in the directory neg
numPosReviewsTrainingData = len(os.listdir(posDirectory))
numNegReviewsTrainingData = len(os.listdir(negDirectory))
totalNumReviewsInTrainingData = (numNegReviewsTrainingData + numPosReviewsTrainingData)

trainingDataLog = np.zeros((totalNumReviewsInTrainingData, 2))       # A numpy matrix to store our result to fit our DT model
                                                                    # Columns: 1) Number of positive adjectives a review has, 2) Number of negative adjectives a review has, 3) Number of swear words a review has
posReviewsDirectoryPath = os.path.normpath(posDirectory)
negReviewsDirectoryPath = os.path.normpath(negDirectory)

for file in posReviews:                                 # Loop through each file in the pos directory to organize data for our DT model
    filepath = os.path.join(posReviewsDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    wordList = (content.lower()).split()
    wordlistwithfreq = {}
    temp = set(wordList)
    for word in temp:
        wordlistwithfreq[word] = 0

    for word in wordList:
        wordlistwithfreq[word] += 1

    try:
        underScoreIndex = file.index('_')           # Take the part of the string before the under score symbol, we will have the file's order
        fileOrder = int(file[:underScoreIndex])
        for w in wordList:
            if w in strongFeatures:
                if posdf[w] > negdf[w]:
                    trainingDataLog[fileOrder][0] += wordlistwithfreq[word] * (np.log(25000) / (1 + df[w]))
                elif posdf[w] < negdf[w]:
                    trainingDataLog[fileOrder][1] += wordlistwithfreq[word] * (np.log(25000) / (1 + df[w]))
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
            if w in strongFeatures:
                if posdf[w] > negdf[w]:
                    trainingDataLog[fileOrder][0] += wordlistwithfreq[word] * (np.log(25000) / (1 + df[w]))
                elif posdf[w] < negdf[w]:
                    trainingDataLog[fileOrder][1] += wordlistwithfreq[word] * (np.log(25000) / (1 + df[w]))
    except ValueError:
        continue

print("This is the input data for our Logistics Regression model: ")
print(trainingDataLog)
print("#--------------------------------------------------------------------------------------------------------------#")
trainTarget = [0] * totalNumReviewsInTrainingData
for i in range(len(posReviews)):
    trainTarget[i] = 1
print("And this is the target parameter for our Logistics Regression model: ")
print (trainTarget)
print("#--------------------------------------------------------------------------------------------------------------#")

model = LogisticRegression()
model.fit(trainingDataLog, trainTarget)

#--------------------------------------------------------------------------------------------------------------#
# For our test data
print("Now we are going to extract the features from our test data set, and prepare it for our Logistic Regression model: ")
testSetDirectoryPath = os.path.normpath(testDirectory)
filesInTest = os.listdir(testDirectory)

numReviewsTestData = len(os.listdir(testDirectory))
testDataLog = np.zeros((numReviewsTestData, 2))

for file in filesInTest:
    filepath = os.path.join(testSetDirectoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    wordList = (content.lower()).split()
    index = int(file[:-4])              # Take away the '.txt' from file (string)
    # , we will have our index for our matrix
    for w in wordList:
        if w in strongFeatures:
            if posdf[w] > negdf[w]:
                testDataLog[index][0] += wordlistwithfreq[word] * (np.log(25000) / (1 + df[w]))
            elif posdf[w] < negdf[w]:
                testDataLog[index][1] += wordlistwithfreq[word] * (np.log(25000) / (1 + df[w]))

print("This is the input data after pre-processing for our test set: ")
print(testDataLog)
print("#--------------------------------------------------------------------------------------------------------------#")
print("Predicting...")
print('')
print("Result: ")
testSetPrediction= model.predict(testDataLog)
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

# print("Random prediction: ", model.predict(([[1000, 10010]])))
dataFrame.to_csv('logisticRegPredictions.csv', encoding='utf-8', index=False)
#--------------------------------------------------------------------------------------------------------------#