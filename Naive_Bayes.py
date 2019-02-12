# COMP551 PROJECT 2: Group number: 5
# Student Name: Cheuk Hang Leung (Eric), Donovan Chiazzese, Mohamed Maoui
# Student ID: 260720788, (Put ur ID's here)
#----------------------------------------------------------------------------------------------------------#
#from pathlib import Path

import os, sys
import numpy as np
import pickle as pk
import matplotlib.pyplot as pp



posDirectory = "/Users/Donovan/Desktop/proj2_materials/Train_Set/pos"
negDirectory = "/Users/Donovan/Desktop/proj2_materials/Train_Set/neg"
testDirectory = "/Users/Donovan/Desktop/proj2_materials/Test_Set"

def lowerCaseAndSplit (words):
    wList = (words.lower()).split(",")
    return wList

posAdjString = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,Incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,fun,zesty,!,?,"
negAdjString = "absurd,arrogant,boring,bad,intolerant,crazy,miserly,patronizing,vulgar,crude,obnoxious,offensive,violent,cryptic,failure,fail,cringy,atrocious,awful,cheap,crummy,dreadful,lousy,noisy,poor,poorly,unacceptable,garbage,gross,horrible,inaccurate,inferior,obnoxious,synthetic,careless,cheesy,crappy,abominable,faulty,godawful,substandard,despicable,horrendous,terrible,attempt,upsetting,not,no,vile,abominable,appalling,cruel,disgusting,dreadful,eerie,grim,hideous,disastrous,disaster,horrid,horrendous,unpleasant,defective,unsatisfied"
AdjString = posAdjString + negAdjString

AdjList = lowerCaseAndSplit(AdjString)          # A list of adjectives

predicted_value = np.zeros([12500])
observed_value = np.zeros([12500])

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

    #print (freqAdjDict)
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

predicted_values = np.zeros([12501])

def noName(directoryString):
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
print('We will now test the algorithm with all 12500 positive reviews:')
print(noName(posDirectory))
print('')
print('We will now test the algorithm with all 12500 negative reviews:')
print(noName(negDirectory))
print('')




print('Test Set')
print(noName(testDirectory))

#print('It seems our algorithm is biased towards positive reviews, this is because we have more positive adjective features.\nTo solve this we will remove some positive adjective features to match the number of negative adjective features.\nWe will select which ones to remove by removing the adjectives with the least difference frequencies between positive and negative reviews')
#print('Seemed to work well')
