import numpy as np
import random
import perceptron as per
import input_gen as igen

def safeInput(*args):
    try:
        return input(*args)
    except Exception as e:
        print(e)
        return ""

def safeIntInput(*args):
    try:
        return int(input(*args))
    except Exception as e:
        print(e)
        return -1

def safeFloatInput(*args):
    try:
        return float(input(*args))
    except Exception as e:
        print(e)
        return -1.0

def askYesNoQuestion(question):
    answer = ""
    while answer != "y" and answer != "n":
        answer = safeInput("\n" + question + " (y or n)? ").lower()

    if answer == "y":
        return True
    else:
        return False

def getMatrixSize():
    size = 0
    while size < 3:
        size = safeIntInput("\nEnter size of input matrix (minimum 3): ")
    return size

def getLearningRate():
    rate = -1.0
    while rate <= 0:
        rate = safeFloatInput("\nEnter learning rate (EX: 0.1): ")
    return rate

def getMaxIterations():
    maxIter = 0
    while maxIter < 100:
        maxIter = safeIntInput("\nEnter maximum number of training iterations (minimum 100): ")
    return maxIter

def getAmountNoise():
    amntNoise = -1
    while amntNoise < 0:
        amntNoise = safeIntInput("\nEnter maximum amount of added noise (minimum 0): ")
    return amntNoise

def initSamples(p, size, amntNoise):
    # p = per.Perceptron(size * size, 0.1)
    gen = igen.InputGenerator(size)

    inputStr = askYesNoQuestion("Would you like to customize the shape of the 'I' and 'L' letters")

    I_samples = None
    L_samples = None

    print()
    print("Note: Generation may take up to", gen.getMaxTime() * 2, "seconds to complete.")
    
    if not inputStr:
        print("Generating basic samples ...")

        I_samples = gen.generate_I_samples(3, 3, amntNoise)
        print("Generated", len(I_samples), "'I' samples.")
        L_samples = gen.generate_L_samples(3, 3, 2, 2, amntNoise)
        print("Generated", len(L_samples), "'L' samples.")  
    else:
        I_minheight = 0
        I_maxHeight = 0
        L_minHeight = 0
        L_maxHeight = 0
        L_minWidth = 0
        L_maxWidth = 0
        
        while I_minheight < 2 or I_minheight > size:
            I_minheight = safeIntInput("Enter minimum height for 'I' character (2 - " + str(size) + "): ")

        while I_maxHeight < I_minheight or I_maxHeight > size:
            I_maxHeight = safeIntInput("Enter maximum height for 'I' character (" + str(I_minheight) + " - " + str(size) + "): ")

        # --------
        while L_minHeight < 2 or L_minHeight > size:
            L_minHeight = safeIntInput("Enter minimum height for 'L' character (2 - " + str(size) + "): ")

        while L_maxHeight < L_minHeight or L_maxHeight > size:
            L_maxHeight = safeIntInput("Enter maximum height for 'L' character (" + str(L_minHeight) + " - " + str(size) + "): ")

        # --------
        while L_minWidth < 2 or L_minWidth > size:
            L_minWidth = safeIntInput("Enter minimum width for 'L' character (2 - " + str(size) + "): ")

        while L_maxWidth < L_minWidth or L_maxWidth > size:
            L_maxWidth = safeIntInput("Enter maximum width for 'L' character (" + str(L_minWidth) + " - " + str(size) + "): ")

        print("Generating custom samples ...")

        I_samples = gen.generate_I_samples(I_minheight, I_maxHeight, amntNoise)
        print("Generated", len(I_samples), "'I' samples.")

        L_samples = gen.generate_L_samples(L_minHeight, L_maxHeight, L_minWidth, L_maxWidth, amntNoise)
        print("Generated", len(L_samples), "'L' samples.")  

    print("\nExample 'I' sample:")
    print(I_samples[0])

    print("\nExample 'L' sample:")
    print(L_samples[0])

    totalSamples = len(I_samples) + len(L_samples)
    if totalSamples < 10:
        print("\nNote: Total number of samples is less than 10.")
        print("Training and Testing sets will reuse the same samples.")
        
        for s in I_samples:
            p.addToTrainingSet(s, -1)
            p.addToTestingSet(s, -1)

        for s in L_samples:
            p.addToTrainingSet(s, 1)
            p.addToTestingSet(s, 1)
    else:
        ratio = safeFloatInput("\nEnter ratio of samples used for training (EX: 0.8): ")
        if ratio < 0.0 or ratio > 1.0:
            print("Error: Ratio must be a value between 0.0 and 1.0")
            exit(1)

        combinedSamples = []

        for s in I_samples:
            combinedSamples.append((s, -1))

        for s in L_samples:
            combinedSamples.append((s, 1))

        trainingSize = int(len(combinedSamples) * ratio)
        random.shuffle(combinedSamples)

        if trainingSize == len(combinedSamples):
            print("Note: Since all samples are being used in training, all will be reused in testing.")
            for i in range(len(combinedSamples)):
                p.addToTrainingSet(combinedSamples[i][0], combinedSamples[i][1])
                p.addToTestingSet(combinedSamples[i][0], combinedSamples[i][1])
        else:
            for i in range(trainingSize):
                p.addToTrainingSet(combinedSamples[i][0], combinedSamples[i][1])

            for i in range(trainingSize, len(combinedSamples)):
                p.addToTestingSet(combinedSamples[i][0], combinedSamples[i][1])

            print("Using", trainingSize, "samples for training and", len(combinedSamples) - trainingSize, "samples for testing.")
