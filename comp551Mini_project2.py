# COMP551 PROJECT 2: Group number: 5
# Student Name: Cheuk Hang Leung (Eric), Donovan Chiazzese, Mohamed Maoui
# Student ID: 260720788, (Put ur ID's here)
#----------------------------------------------------------------------------------------------------------#
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

def lowerCaseAndSplit (words):
    wList = (words.lower()).split(",")
    return wList

stopwordsList = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
posAdjString = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,Incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,phenomenal,always,also,both,hit,fun,zesty"
negAdjString = "absurd,arrogant,boring,bad,intolerant,crazy,miserly,patronizing,vulgar,crude,obnoxious,offensive,violent,cryptic,failure,fail,cringy,atrocious,awful,cheap,crummy,dreadful,lousy,noisy,poor,poorly,unacceptable,garbage,gross,horrible,inaccurate,inferior,obnoxious,synthetic,careless,cheesy,crappy,abominable,faulty,godawful,substandard,despicable,horrendous,terrible,attempt,upsetting,not,no,vile,abominable,appalling,cruel,disgusting,dreadful,eerie,grim,hideous,disastrous,disaster,horrid,horrendous,any,can't,because,better,anything,unpleasant,defective,miserable,failed,unsatisfied"

AdjString = posAdjString + negAdjString

AdjList = lowerCaseAndSplit(AdjString)          # A list of adjectives

predicted_value = np.zeros([12501])
observed_value = np.zeros([12501])

# def top150Words(directoryString):
#     wordFreqDict = {}  # A dictionary to store the frequency of every word from all comments
#     directory = os.listdir(directoryString)
#     directoryPath = os.path.normpath(directoryString)
#     for file in directory:
#         filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
#         content = open(filepath, 'r', encoding='latin-1')
#         content = content.read()
#         wordList = (content.lower()).split()
#
#
#         for z in wordList:  # Loop through each comment
#             if z in wordFreqDict:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
#                 wordFreqDict[z] += 1
#             else:  # Else case: We have encountered a new word, therefore we just simply add the word into our dictionary and set its value to 1
#                 wordFreqDict[z] = 1
#
#     wordListTuple = sorted(wordFreqDict.items(), key=lambda x: x[1],reverse=True)  # Sort our dictionary by value and store it into a list of tuples
#     wordListTuple = wordListTuple[0:500]  # We only need the top 160 values(frequencies) from our list
#
#     x = sorted(wordListTuple)
#     print(x)
#
# #top150Words(posDirectory)
#top150Words(negDirectory)

def checkAdjFreq (directoryString):   # A function that takes in a review (.txt file), count the frequencies of each adjective that are in our AdjList appear in that particular review.
    freqAdjDict = {}
    for w in AdjList:                       # Loop through the review
        freqAdjDict[w] = 0
    directory = os.listdir(directoryString)
    directoryPath = os.path.normpath(directoryString)
    for file in directory:
        filepath = os.path.join(directoryPath, os.path.normpath(file))       # Filepath = directoryPath + filename
        content = open(filepath, 'r', encoding='latin-1')
        content = content.read()
        wordList = (content.lower()).split()

        for word in wordList:
            if word in freqAdjDict:
                freqAdjDict[word] += 1

    # print (freqAdjDict)
    return freqAdjDict

posReviewsFreqDict = checkAdjFreq(posDirectory)
negReviewsFreqDict = checkAdjFreq(negDirectory)

# The two dictionaries with the word count frequencies for every feature has all been written onto 2 files to avoid unnecessarily recalculating

# posReviewFreqDict is the word count of each feature for positive reviews
# negReviewFreqDict is the word count of each feature for negative reviews


# This was the code to write the files
with open('store_posReviewsFreqDict.pk', 'wb') as f:  # Python 3: open(..., 'wb')
    pk.dump(posReviewsFreqDict, f)


with open('store_negReviewsFreqDict.pk', 'wb') as f:  # Python 3: open(..., 'wb')
    pk.dump(negReviewsFreqDict, f)

# Files loaded

newfile = 'store_posReviewsFreqDict.pk'

with open(newfile, 'rb') as fi:
    posReviewsFreqDict = pk.load(fi)

newfile2 = 'store_negReviewsFreqDict.pk'

with open(newfile2, 'rb') as pi:
    negReviewsFreqDict = pk.load(pi)

print(posReviewsFreqDict)
print(negReviewsFreqDict)
print('')

# Naive Bayes Algorithm

training_predicted_values = np.zeros([12501])
test_predicted_values = np.zeros([25001])
def noName(directoryString, predicted_values):
    wordIsPresent = {}
    directory = os.listdir(directoryString)
    directoryPath = os.path.normpath(directoryString)
    review = 0
    for file in directory:
        filepath = os.path.join(directoryPath, os.path.normpath(file))       # Filepath = directoryPath + filename
        content = open(filepath, 'r', encoding='latin-1')
        content = content.read()
        wordList = (content.lower()).split()

        for word in wordList:
            if word in posReviewsFreqDict:
                wordIsPresent[word] = 1

        sum_Class_1 = 0
        sum_Class_0 = 0

        for word in wordIsPresent:
            sum_Class_1 += posReviewsFreqDict[word] / (posReviewsFreqDict[word] + negReviewsFreqDict[word] + 1)
            sum_Class_0 += negReviewsFreqDict[word] / (posReviewsFreqDict[word] + negReviewsFreqDict[word] + 1)

        if sum_Class_1 > sum_Class_0:
            predicted_values[review] = 1
        else:
            predicted_values[review] = 0
        review += 1

    numOfPos = float(0)
    numOfNeg = float(0)
    for i in range(0,predicted_values.size):
        if predicted_values[i] == 1:
            numOfPos += 1
        else:
            numOfNeg += 1
    print('number of positive reviews = ', numOfPos)
    print('number of negative reviews = ', numOfNeg)

    if numOfPos > numOfNeg:
        accuracy = float(numOfPos/float(12501)) * 100
        print('That gives us an accuracy of: ', accuracy, '%')
    else:
        accuracy = float(numOfNeg/float(12501)) * 100
        print('That gives us an accuracy of: ', accuracy, '%')

    return predicted_values


print('Training Set')
print('We will now test the algorithm with all 12501 positive reviews:')
print(noName(posDirectory, training_predicted_values))
print('')
print('We will now test the algorithm with all 12501 negative reviews:')
print(noName(negDirectory, training_predicted_values))
print('')


def noName2(directoryString, predicted_values):
    wordIsPresent = {}
    directory = os.listdir(directoryString)
    directoryPath = os.path.normpath(directoryString)
    review = 0
    for file in directory:
        filepath = os.path.join(directoryPath, os.path.normpath(file))       # Filepath = directoryPath + filename
        content = open(filepath, 'r', encoding='latin-1')
        content = content.read()
        wordList = (content.lower()).split()

        for word in wordList:
            if word in posReviewsFreqDict:
                wordIsPresent[word] = 1

        sum_Class_1 = 0
        sum_Class_0 = 0

        for word in wordIsPresent:
            sum_Class_1 += posReviewsFreqDict[word] / (posReviewsFreqDict[word] + negReviewsFreqDict[word] + 1)
            sum_Class_0 += negReviewsFreqDict[word] / (posReviewsFreqDict[word] + negReviewsFreqDict[word] + 1)

        if sum_Class_1 > sum_Class_0:
            predicted_values[review] = 1
        else:
            predicted_values[review] = 0
        review += 1

    numOfPos = float(0)
    numOfNeg = float(0)
    for i in range(0,predicted_values.size):
        if predicted_values[i] == 1:
            numOfPos += 1
        else:
            numOfNeg += 1
    print('number of positive reviews = ', numOfPos)
    print('number of negative reviews = ', numOfNeg)

    return predicted_values

csv_predictedValues = np.zeros((25000,2))               # A numpy matrix to store our result, convert to CSV data frame.

print("")
print('Test Set')
predicted_values = noName2(testDirectory, test_predicted_values)

for i in range(len(csv_predictedValues)):
    csv_predictedValues[i][0] = int(i)
    csv_predictedValues[i][1] = int(predicted_values[i])

dataFrame = pd.DataFrame(csv_predictedValues, columns=["Id", "Category"])
dataFrame.Id = dataFrame.Id.astype(int)
dataFrame.Category = dataFrame.Category.astype(int)
# print(dataFrame)

dataFrame.to_csv('PredictedValues.csv', encoding='utf-8', index=False)


#--------------------------------------------------------------------------------------------------------------#
# Decision Tree classifier






#--------------------------------------------------------------------------------------------------------------#




