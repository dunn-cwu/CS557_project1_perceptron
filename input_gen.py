import numpy as np
import time

MAX_SEQUENTIAL_FAILURES = 50
MAX_TIME_SECONDS = 240

class InputGenerator:
    def __init__(self, matrixSize):
        self.matrixSize = matrixSize

    def getMaxTime(self):
        return MAX_TIME_SECONDS

    def isUniqueSample(self, sampleList, testSample):
        for s in sampleList:
            if np.array_equal(s, testSample):
                return False
            
        return True
    def generate_I(self, minHeight, maxHeight, amntNoise = 0):
        if maxHeight > self.matrixSize:
            raise Exception("Error: Max height cannot be greater than matrix size")
        height = np.random.randint(minHeight, maxHeight + 1)
        x_pos = np.random.randint(0, self.matrixSize)
        y_pos = np.random.randint(0, self.matrixSize - height + 1)
        sample = np.zeros((self.matrixSize, self.matrixSize))

        for i in range(y_pos, y_pos + height):
            sample[i][x_pos] = 1

        if amntNoise > 0:
            randNoise = np.random.randint(0, amntNoise + 1)
            for i in range(randNoise):
                sample[np.random.randint(0, self.matrixSize)][np.random.randint(0, self.matrixSize)] = 1

        return sample
    def generate_L(self, minHeight, maxHeight, minWidth, maxWidth, amntNoise = 0):
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

        if amntNoise > 0:
            for i in range(amntNoise):
                sample[np.random.randint(0, self.matrixSize)][np.random.randint(0, self.matrixSize)] = 1

        return sample
    def generate_I_samples(self, minHeight, maxHeight, amntNoise = 0, maxSamples=0xFFFF):
        samples = []
        seqFailures = 0
        start_time = time.time()

        while len(samples) < maxSamples and seqFailures < MAX_SEQUENTIAL_FAILURES:
            sample = self.generate_I(minHeight, maxHeight, amntNoise)
            if self.isUniqueSample(samples, sample):
                seqFailures = 0
                samples.append(sample)
            else:
                seqFailures += 1

            end_time = time.time()
            if end_time - start_time > MAX_TIME_SECONDS:
                break
        return samples
    def generate_L_samples(self, minHeight, maxHeight, minWidth, maxWidth, amntNoise = 0, maxSamples=0xFFFF):
        samples = []
        seqFailures = 0
        start_time = time.time()

        while len(samples) < maxSamples and seqFailures < MAX_SEQUENTIAL_FAILURES:
            sample = self.generate_L(minHeight, maxHeight, minWidth, maxWidth, amntNoise)
            if self.isUniqueSample(samples, sample):
                seqFailures = 0
                samples.append(sample)
            else:
                seqFailures += 1
            
            end_time = time.time()
            if end_time - start_time > MAX_TIME_SECONDS:
                break

        return samples
        
#g = InputGenerator(3)
#print(g.generate_L_samples(3, 3, 2, 2)[0])
