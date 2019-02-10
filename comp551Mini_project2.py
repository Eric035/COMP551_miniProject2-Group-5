# COMP551 PROJECT 2: Group number: 5
# Student Name: Cheuk Hang Leung (Eric), Donovan Chiazzese, Mohamed Maoui
# Student ID: 260720788, (Put ur ID's here)
#----------------------------------------------------------------------------------------------------------#
#from pathlib import Path

import os, sys
import numpy as np
posDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/pos"
negDirectory = "/Users/ericleung/Desktop/Comp/Comp551/Comp551_Project2/project2Reviews/train/neg"

def lowerCaseAndSplit (words):
    wList = (words.lower()).split(",")
    return wList

posAdjString = "amazing,alluring,adventurous,amusing,awesome,ambitious,beautiful,bold,brainy,breathtaking,blazing,brazen,cool,cheerful,charming,creative,clever,daring,delightful,dazzling,energetic,elegant,excellent,exceptional,emotional,exuberant,fascinating,fantastic,funny,genius,glorious,great,genuine,happy,honest,helpful,heavenly,hilarious,humorous,hearty,Incredible,inspirational,inspiring,impeccable,ingenious,impressive,innovative,insightful,intense,impartial,imaginative,independent,intuitive,inventive,intellectual,intelligent,Jolly,joyful,jubilant,jovial,joyous,Keen,kind,kindhearted,lively,lovable,lovely,marvellous,majestic,modest,nice,positive,passionate,perfect,phenomenal,quirky,ravishing,reserved,romantic,sweet,stunning,smart,sensational,thrilling,tenacious,talented,upbeat,uplifting,vigorous,valiant,versatile,witty,wise,wonderful,welcoming,warmhearted,inventive,witty,funzesty"
negAdjString = "Absurd,Arrogant,Boring,bad,intolerant,crazy,Miserly,Patronizing,Vulgar,Crude,Obnoxious,Offensive,Violent,Cryptic"
AdjString = posAdjString + negAdjString

AdjList = lowerCaseAndSplit(AdjString)          # A list of adjectives

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

    print (freqAdjDict)
    return freqAdjDict

posReviewsFreqDict = checkAdjFreq(posDirectory)
negReviewsFreqDict = checkAdjFreq(negDirectory)





