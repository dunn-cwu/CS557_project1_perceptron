import numpy as np

I_MATRIX = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
L_MATRIX = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])

class Perceptron:
    def __init__(self, inputMatrixSize, learningRate):
        self.inputSize = inputMatrixSize + 1
        self.learningRate = learningRate
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
                break
        
        print("\n=======================================")
        print("Training completed in", iteration, "iterations")
        print("Final weight vector:", self.weightVector)

print("I matrix:")
print(I_MATRIX)

print("\nL matrix:")
print(L_MATRIX)

(rows, cols) = np.shape(I_MATRIX)
p = Perceptron(rows * cols, 1.0)
p.addToTrainingSet(I_MATRIX, -1)
p.addToTrainingSet(L_MATRIX, 1)

print("\nInital weight vector:", p.weightVector)

p.train()
