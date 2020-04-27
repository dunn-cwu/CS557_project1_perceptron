import numpy as np

MAX_SEQUENTIAL_FAILURES = 50

class InputGenerator:
    def __init__(self, matrixSize):
        self.matrixSize = matrixSize
    def isUniqueSample(self, sampleList, testSample):
        for s in sampleList:
            if np.array_equal(s, testSample):
                return False
            
        return True
    def generate_I(self, minHeight, maxHeight):
        if maxHeight > self.matrixSize:
            raise Exception("Error: Max height cannot be greater than matrix size")
        height = np.random.randint(minHeight, maxHeight + 1)
        x_pos = np.random.randint(0, self.matrixSize)
        y_pos = np.random.randint(0, self.matrixSize - height + 1)
        sample = np.zeros((self.matrixSize, self.matrixSize))

        for i in range(y_pos, y_pos + height):
            sample[i][x_pos] = 1

        return sample
    def generate_L(self, minHeight, maxHeight, minWidth, maxWidth):
        if maxHeight > self.matrixSize:
            raise Exception("Error: Max height cannot be greater than matrix size")
        if maxWidth > self.matrixSize:
            raise Exception("Error: Max width cannot be greater than matrix size")
        
        width = np.random.randint(minWidth, maxWidth + 1)
        height = np.random.randint(minHeight, maxHeight + 1)
        x_pos = np.random.randint(0, self.matrixSize - width + 1)
        y_pos = np.random.randint(0, self.matrixSize - height + 1)
        sample = np.zeros((self.matrixSize, self.matrixSize))

        for i in range(x_pos + 1, x_pos + width):
            sample[y_pos + height - 1][i] = 1

        for i in range(y_pos, y_pos + height):
            sample[i][x_pos] = 1

        return sample
    def generate_I_samples(self, minHeight, maxHeight, maxSamples=0xFFFF):
        samples = []
        seqFailures = 0

        while len(samples) < maxSamples and seqFailures < MAX_SEQUENTIAL_FAILURES:
            sample = self.generate_I(minHeight, maxHeight)
            if self.isUniqueSample(samples, sample):
                seqFailures = 0
                samples.append(sample)
            else:
                seqFailures += 1
        return samples
    def generate_L_samples(self, minHeight, maxHeight, minWidth, maxWidth, maxSamples=0xFFFF):
        samples = []
        seqFailures = 0

        while len(samples) < maxSamples and seqFailures < MAX_SEQUENTIAL_FAILURES:
            sample = self.generate_L(minHeight, maxHeight, minWidth, maxWidth)
            if self.isUniqueSample(samples, sample):
                seqFailures = 0
                samples.append(sample)
            else:
                seqFailures += 1
        return samples
        
#g = InputGenerator(3)
#print(g.generate_L_samples(3, 3, 2, 2)[0])
