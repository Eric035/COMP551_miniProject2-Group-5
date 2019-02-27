 COMP551 PROJECT 2: Group number: 5
# Student Name: Cheuk Hang Leung (Eric), Donovan Chiazzese, Mohamed Maoui
# Student ID: 260720788, (Put ur ID's here)
#----------------------------------------------------------------------------------------------------------#
#from pathlib import Path



##### TO DO #####

#Fix ordering of file reader




import os, sys
import numpy as np
import pickle as pk
import pandas as pd
import matplotlib.pyplot as pp
import csv
import os

from nltk import word_tokenize
from nltk.util import ngrams

predicted_values = np.zeros([12501])
observed_values = np.zeros([12501])

posDirectory = "/Users/Donovan/Desktop/proj2_materials/Train_Set/pos"
negDirectory = "/Users/Donovan/Desktop/proj2_materials/Train_Set/neg"
testDirectory = "/Users/Donovan/Desktop/proj2_materials/Test_Set"


def lowerCaseAndSplit (words):
    wList = (words.lower()).split(",")
    return wList

def preprocess(content):

    content = content.replace(".", "").replace(";", "").replace(",", "").replace("!", "").replace("?", "").replace("^", "")\
        .replace( ":", "").replace("(", "").replace(")", "").replace("<", " ").replace(">", "").replace(">>", "") \
        .replace(">", "").replace("/", "")

    return content




## BI-GRAMS

wordFreqDict = {}  # A dictionary to store the frequency of every word from all comments

directory = os.listdir(posDirectory)
directoryPath = os.path.normpath(posDirectory)

for file in directory:

    filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    content = preprocess(content)
    wordList = (content.lower()).split()


    for z in wordList:  # Loop through each comment
        if z in wordFreqDict:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
            wordFreqDict[z] += 1
        else:  # Else case: We have encountered a new word, therefore we just simply add the word into our dictionary and set its value to 1
            wordFreqDict[z] = 1


directory = os.listdir(negDirectory)
directoryPath = os.path.normpath(negDirectory)

for file in directory:

    filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    content = preprocess(content)
    wordList = (content.lower()).split()

    for z in wordList:  # Loop through each comment
        if z in wordFreqDict:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
            wordFreqDict[z] += 1
        else:  # Else case: We have encountered a new word, therefore we just simply add the word into our dictionary and set its value to 1
            wordFreqDict[z] = 1

wordListTuple = sorted(wordFreqDict.items(), key=lambda x: x[1],reverse=True)  # Sort our dictionary by value and store it into a list of tuples
wordListTuple = wordListTuple[0:50000]  # We only need the top 2000 values(frequencies) from our list

topWords = list()

for i in range(len(wordListTuple)):
    temp = wordListTuple[i][0]
    topWords.append(temp)

adj = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,Incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,phenomenal,always,also,both,hit,fun,zesty,!,?,absurd,arrogant,boring,bad,intolerant,crazy,miserly,patronizing,vulgar,crude,obnoxious,offensive,violent,cryptic,failure,fail,cringy,atrocious,awful,cheap,crummy,dreadful,lousy,noisy,poor,poorly,unacceptable,garbage,gross,horrible,inaccurate,inferior,obnoxious,synthetic,careless,cheesy,crappy,abominable,faulty,godawful,substandard,despicable,horrendous,terrible,attempt,upsetting,not,no,vile,abominable,appalling,cruel,disgusting,dreadful,eerie,grim,hideous,disastrous,disaster,horrid,horrendous,any,can't,because,better,anything,unpleasant,defective,miserable,failed,unsatisfiedact,actor,actress,adaptation,ambiance,angle,antagonist,protagonist,anti-climax,anti-hero,archetype,atmosphere,audience,audition,audio,author,backdrop,background,balance,blockbuster,box-office,cameo,camera,caption,caricature,cast,casting,censor,cgi,character,cinema,cinematic,cinerama,cliffhanger,climax,comic,commentary,compose,compostion,contrast,convention,credits,credit,critic,critique,crisis,depth,dialogue,director,directing,documentary,document,dynamic,edit,editing,ensemble,epilogue,exposition,extra,feature,film,movie,featurette,flick,focus,footage,foreshadow,foreshadowing,format,frame,framing,genre,hero,homage,image,interlude,lens,light,lighting,mark,master,metaphor,simile,methods,monitor,monologue,montage,musical,romance,narrator,narrate,narration,narrative,nostalgia,nostalgic,novel,overture,tone,pace,parody,persona,plot,premiere,prequel,preview,design,prologue,rate,rating,reaction,shot,remake,resolution,vision,satire,scene,scenery,score,screen,write,writer,writing,script,sequence,setting,sound,quality,soundtrack,effects,spoof,studio,subplot,suspense,symbolism,talent,theme,producer,production,relief,perform,performance"
adj = lowerCaseAndSplit(adj)

for word in adj:
    if word in topWords:
        continue
    else:
        topWords.append(word)

features = topWords
print(features)



directory = os.listdir(posDirectory)
directoryPath = os.path.normpath(posDirectory)

df = {}
negdf = {}  # A dictionary to store the frequency of every word from all comments

for word in features:
    negdf[word] = 0
    df[word] = 0
for file in directory:

    filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    content = preprocess(content)
    wordList = (content.lower()).split()

    wordList = set(wordList)
    for z in wordList:  # Loop through each comment
        if z in features:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
            negdf[z] += 1
            df[z] += 1

directory = os.listdir(negDirectory)
directoryPath = os.path.normpath(negDirectory)

posdf = {}

for word in features:
    posdf[word] = 0
    df[word] = 0
for file in directory:

    filepath = os.path.join(directoryPath, os.path.normpath(file))  # Filepath = directoryPath + filename
    content = open(filepath, 'r', encoding='latin-1')
    content = content.read()
    content = preprocess(content)
    wordList = (content.lower()).split()

    wordList = set(wordList)
    for z in wordList:  # Loop through each comment
        if z in features:  # If case: The word is already in our word frequency dictionary, so we just increment its frequency by 1
            posdf[z] += 1
            df[z] += 1

strongFeatures = []

for word in posdf:
    if word in posdf:
        if word in negdf:
            posratio = (posdf[word]+1)/(negdf[word]+1)
            negratio = (negdf[word] + 1) / (posdf[word] + 1)
            if posratio > 1.2 or negratio > 1.2:
                strongFeatures.append(word)

print(strongFeatures)
print(len(strongFeatures))

# This was the code to write the files
with open('strongFeatures.pk', 'wb') as f:  # Python 3: open(..., 'wb')
    pk.dump(strongFeatures, f)

with open('df.pk', 'wb') as f:  # Python 3: open(..., 'wb')
    pk.dump(df, f)


