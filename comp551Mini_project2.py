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
            predicted_values_train[review] = 1
        else:
            predicted_values_train[review] = 0
        review += 1

    numOfPos = float(0)
    numOfNeg = float(0)
    for i in range(0,predicted_values_train.size):
        if predicted_values_train[i] == 1:
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

    return predicted_values_train
