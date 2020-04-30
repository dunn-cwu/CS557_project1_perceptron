# Assignment: Project 1
# Course: CS557 Computational Intelligence and Machine Learning
# Instructor: Dr. Razvan Andonie
# Student: Andrew Dunn
# Date: 4/30/2020
# File: perceptron.py
# Desc: Contains the Perceptron class, which is a simple single neuron
#       machine learning model that can classify an input into two different
#       classes.

import numpy as np
import time

class Perceptron:
    def __init__(self, inputMatrixSize, learningRate, maxIterations):
        self.inputSize = inputMatrixSize + 1
        self.learningRate = learningRate
        self.maxIterations = maxIterations
        self.trainingSet = []
        self.trainingLabels = []
        self.testingSet = []
        self.testingLabels = []
        self.initializeRandomWeights()

    def initializeRandomWeights(self):
        self.weightVector = np.random.uniform(low=-1, high=1, size=(self.inputSize,))

    def vectorizeMatrix(self, m):
        return np.append(np.ndarray.flatten(m), -1)

    def addToTrainingSet(self, matrix, label):
        v = self.vectorizeMatrix(matrix)
        if np.shape(v)[0] != self.inputSize:
            raise Exception("Error: input training matrix is incorrect size")
        if label != -1 and label != 1:
            raise Exception("Error: Label must be -1 or 1")
        self.trainingSet.append(v)
        self.trainingLabels.append(label)

    def addToTestingSet(self, matrix, label):
        v = self.vectorizeMatrix(matrix)
        if np.shape(v)[0] != self.inputSize:
            raise Exception("Error: input test matrix is incorrect size")
        if label != -1 and label != 1:
            raise Exception("Error: Label must be -1 or 1")
        self.testingSet.append(v)
        self.testingLabels.append(label)

    def sigmoidFunc(self, resultVector):
        vSum = np.sum(resultVector)
        if vSum <= 0:
            return -1
        else:
            return 1

    def adjustWeights(self, inputVector, expectedResult, actualResult):
        newWeights = (self.learningRate * (expectedResult - actualResult)) * inputVector
        newWeights += self.weightVector
        self.weightVector = newWeights

    def train(self):
        if len(self.trainingSet) <= 0:
            raise Exception("Error: Training set is empty")

        iteration = 0
        trainingSetSize = len(self.trainingSet)
        trainingSuccess = False

        print("Training using", trainingSetSize, "samples ...")

        start_time = time.time()

        while True:
            iteration += 1
            numCorrect = 0

            for i in range(trainingSetSize):
                resultVect = self.trainingSet[i] * self.weightVector
                resultVal = self.sigmoidFunc(resultVect)

                if resultVal == self.trainingLabels[i]:
                    numCorrect += 1
                else:
                    self.adjustWeights(self.trainingSet[i], self.trainingLabels[i], resultVal)

            if numCorrect == trainingSetSize:
                trainingSuccess = True
                break
            elif iteration >= self.maxIterations:
                break
        
        end_time = time.time()

        print("\n=======================================")
        if trainingSuccess:
            print("Training successfully completed in", iteration, 
                "iterations (" + str(round(end_time - start_time, 3)) + " seconds)")
        else:
            print("Training failed to complete after", iteration, 
                "iterations (" + str(round(end_time - start_time, 3)) + " seconds)")
        print("Final weight vector:", self.weightVector)
        print("=======================================\n")

        return trainingSuccess

    def test(self):
        if len(self.testingSet) <= 0:
            raise Exception("Error: Testing set is empty")

        testingSetSize = len(self.testingSet)
        numCorrect = 0

        print("Testing using", testingSetSize, "samples ...")

        start_time = time.time()

        for i in range(testingSetSize):
            resultVect = self.testingSet[i] * self.weightVector
            resultVal = self.sigmoidFunc(resultVect)

            if resultVal == self.testingLabels[i]:
                numCorrect += 1

        end_time = time.time()

        accuracy = round((float(numCorrect) / float(testingSetSize)) * 100, 2)

        print("\n=======================================")
        print("Testing completed in", round(end_time - start_time, 3), "seconds.")
        print(numCorrect, "out of", testingSetSize, 
            "inputs were correctly classified (" + str(accuracy) + "%).")
        print("=======================================\n")
        return accuracy

# ================================
# End Of File
# ================================
