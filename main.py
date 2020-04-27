import numpy as np
import random
import perceptron as per
import input_gen as igen

size = int(input("Enter size of input matrix (minimum 3): "))
if size < 3:
    print("Error: Size cannot be less than 3")
    exit(1)

p = per.Perceptron(size * size, 0.1)
gen = igen.InputGenerator(size)

inputStr = ""

while inputStr != "y" and inputStr != "n":
    inputStr = input("Would you like to customize the shape of the 'I' and 'L' letters (y or n)? ").lower()

I_samples = None
L_samples = None

if inputStr == "n":
    print("Generating basic samples ...")

    I_samples = gen.generate_I_samples(3, 3)
    L_samples = gen.generate_L_samples(3, 3, 2, 2)
else:
    I_minheight = 0
    I_maxHeight = 0
    L_minHeight = 0
    L_maxHeight = 0
    L_minWidth = 0
    L_maxWidth = 0

    while I_minheight < 2 or I_minheight > size:
        I_minheight = int(input("Enter minimum height for 'I' character (2 - " + str(size) + "): "))

    while I_maxHeight < I_minheight or I_maxHeight > size:
        I_maxHeight = int(input("Enter maximum height for 'I' character (" + str(I_minheight) + " - " + str(size) + "): "))

    # --------
    while L_minHeight < 2 or L_minHeight > size:
        L_minHeight = int(input("Enter minimum height for 'L' character (2 - " + str(size) + "): "))

    while L_maxHeight < L_minHeight or L_maxHeight > size:
        L_maxHeight = int(input("Enter maximum height for 'L' character (" + str(L_minHeight) + " - " + str(size) + "): "))

    # --------
    while L_minWidth < 2 or L_minWidth > size:
        L_minWidth = int(input("Enter minimum width for 'L' character (2 - " + str(size) + "): "))

    while L_maxWidth < L_minWidth or L_maxWidth > size:
        L_maxWidth = int(input("Enter maximum width for 'L' character (" + str(L_minWidth) + " - " + str(size) + "): "))

    print("Generating custom samples ...")

    I_samples = gen.generate_I_samples(I_minheight, I_maxHeight)
    L_samples = gen.generate_L_samples(L_minHeight, L_maxHeight, L_minWidth, L_maxWidth)

print("Generated", len(I_samples), "'I' samples.")
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
    ratio = float(input("\nEnter ratio of samples used for training (EX: 0.8): "))
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

print("\nInitial weight vector:", p.weightVector)
print()

p.train()
p.test()
