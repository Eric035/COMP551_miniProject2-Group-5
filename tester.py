# COMP551 PROJECT 2: Group number: 5
# Student Name: Cheuk Hang Leung (Eric), Donovan Chiazzese, Mohamed Maoui
# Student ID: 260720788, (Put ur ID's here)
#----------------------------------------------------------------------------------------------------------#
#from pathlib import Path

import pickle as pk

##### TO DO #####

#Fix ordering of file reader

newfile = 'strongFeatures.pk'
with open(newfile, 'rb') as fi:
    strongFeatures = pk.load(fi)

print(strongFeatures)

newfile = 'df.pk'
with open(newfile, 'rb') as fi:
    df = pk.load(fi)

print(df)


#
#
# import os, sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as pp
# import csv
# import os
#
# predicted_values = np.zeros([12501])
# observed_values = np.zeros([12501])
#
# posDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/pos"
# negDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/neg"
# testDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/test"
#
#
# def lowerCaseAndSplit (words):
#     wList = (words.lower()).split(",")
#     return wList
#
# def preprocess(content):
#
#     content = content.replace(".", "").replace(";", "").replace(",", "").replace("!", "").replace("?", "").replace("^", "")\
#         .replace( ":", "").replace("(", "").replace(")", "").replace("<", " ").replace(">", "").replace(">>", "") \
#         .replace(">", "").replace("/", "")
#
#     content = content.replace("the", "").replace("of", "").replace("", "")
#
#     return content
#
#
# wordFreqDict = {}  # A dictionary to store the frequency of every word from all comments
#
# directory = os.listdir(posDirectory)
# directoryPath = os.path.normpath(posDirectory)
#
# for file in directory:
#
#     filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
#     content = open(filepath, 'r', encoding='latin-1')
#     content = content.read()
#     content = preprocess(content)
#     wordList = (content.lower()).split()
#
#
#     for z in wordList:  # Loop through each comment
#         if z in wordFreqDict:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
#             wordFreqDict[z] += 1
#         else:  # Else case: We have encountered a new word, therefore we just simply add the word into our dictionary and set its value to 1
#             wordFreqDict[z] = 1
#
#
# directory = os.listdir(negDirectory)
# directoryPath = os.path.normpath(negDirectory)
#
# for file in directory:
#
#     filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
#     content = open(filepath, 'r', encoding='latin-1')
#     content = content.read()
#     content = preprocess(content)
#     wordList = (content.lower()).split()
#
#     for z in wordList:  # Loop through each comment
#         if z in wordFreqDict:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
#             wordFreqDict[z] += 1
#         else:  # Else case: We have encountered a new word, therefore we just simply add the word into our dictionary and set its value to 1
#             wordFreqDict[z] = 1
#
# wordListTuple = sorted(wordFreqDict.items(), key=lambda x: x[1],reverse=True)  # Sort our dictionary by value and store it into a list of tuples
# wordListTuple = wordListTuple[0:5000]  # We only need the top 2000 values(frequencies) from our list
#
# topWords = list()
#
# for i in range(len(wordListTuple)):
#     temp = wordListTuple[i][0]
#     topWords.append(temp)
#
# adj = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,Incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,phenomenal,always,also,both,hit,fun,zesty,!,?,absurd,arrogant,boring,bad,intolerant,crazy,miserly,patronizing,vulgar,crude,obnoxious,offensive,violent,cryptic,failure,fail,cringy,atrocious,awful,cheap,crummy,dreadful,lousy,noisy,poor,poorly,unacceptable,garbage,gross,horrible,inaccurate,inferior,obnoxious,synthetic,careless,cheesy,crappy,abominable,faulty,godawful,substandard,despicable,horrendous,terrible,attempt,upsetting,not,no,vile,abominable,appalling,cruel,disgusting,dreadful,eerie,grim,hideous,disastrous,disaster,horrid,horrendous,any,can't,because,better,anything,unpleasant,defective,miserable,failed,unsatisfied"
# adj = lowerCaseAndSplit(adj)
#
# for word in adj:
#     if word in topWords:
#         continue
#     else:
#         topWords.append(word)
#
# features = topWords
# print(features)
#
#
#
# directory = os.listdir(posDirectory)
# directoryPath = os.path.normpath(posDirectory)
#
# df = {}  # A dictionary to store the frequency of every word from all comments
#
# for word in features:
#     df[word] = 0
#
# for file in directory:
#
#     filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
#     content = open(filepath, 'r', encoding='latin-1')
#     content = content.read()
#     content = preprocess(content)
#     wordList = (content.lower()).split()
#
#     wordList = set(wordList)
#     for z in wordList:  # Loop through each comment
#         if z in features:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
#             df[z] += 1
#
# directory = os.listdir(negDirectory)
# directoryPath = os.path.normpath(negDirectory)
#
# for word in features:
#     df[word] = 0
#
# for file in directory:
#
#     filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
#     content = open(filepath, 'r', encoding='latin-1')
#     content = content.read()
#     content = preprocess(content)
#     wordList = (content.lower()).split()
#
#     wordList = set(wordList)
#     for z in wordList:  # Loop through each comment
#         if z in features:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
#             df[z] += 1
#
#
# with open('df.pk', 'wb') as f:  # Python 3: eopen(..., 'wb')
#     pk.dump(df, f)