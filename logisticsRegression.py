from sklearn.linear_model import LogisticRegression
import os, sys
import csv
import numpy as np
import pandas as pd

posDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/pos"
negDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/neg"
testDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/test"

posAdjString = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,best,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,enjoyable,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,imaginative,incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,phenomenal,pleasant,always,also,both,hit,fun,zesty"
negAdjString = "absurd,arrogant,boring,bad,intolerant,crazy,miserly,patronizing,vulgar,crude,obnoxious,offensive,violent,cryptic,failure,fail,cringy,atrocious,awful,cheap,crummy,dreadful,lousy,noisy,poor,poorly,unacceptable,garbage,gross,horrible,inaccurate,inferior,obnoxious,synthetic,careless,cheesy,crappy,abominable,faulty,godawful,substandard,despicable,horrendous,terrible,attempt,upsetting,not,no,vile,abominable,appalling,cruel,disgusting,dreadful,eerie,grim,hideous,disastrous,disaster,horrid,horrendous,any,can't,because,better,anything,unpleasant,defective,miserable,failed,unsatisfied"
stopwordsList = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

#--------------------------------------------------------------------------------------------------------------#
# Logistic Regression Classifier
print("We are training our Logistics Regression model with our training set...")
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

print("This is the input data for our Logistics Regression model: ")
print(trainingDataDT)
print("#--------------------------------------------------------------------------------------------------------------#")
trainTarget = [0] * totalNumReviewsInTrainingData
for i in range(len(posReviews)):
    trainTarget[i] = 1
print("And this is the target parameter for our Logistics Regression model: ")
print (trainTarget)
print("#--------------------------------------------------------------------------------------------------------------#")
model = LogisticRegression()
model.fit(trainingDataDT, trainTarget)

#--------------------------------------------------------------------------------------------------------------#
# For our test data
print("Now we are going to extract the features from our test data set, and prepare it for our Logistic Regression model: ")
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


# print("Random prediction: ", model.predict(([[1000, 10010]])))
dataFrame.to_csv('logisticRegPredictions.csv', encoding='utf-8', index=False)
#--------------------------------------------------------------------------------------------------------------#