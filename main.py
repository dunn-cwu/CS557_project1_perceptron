import perceptron as per
import experiment as exper

def main():
    createPerceptron = False
    size = None
    learningRate = None
    maxIterations = None
    p = None
    
    while True:
        createPerceptron = False

        if size == None or exper.askYesNoQuestion("Would you like to enter a new matrix size?"):
            print("Note: A new training and sample set will have to be created.")
            size = exper.getMatrixSize()
            createPerceptron = True
        else:
            print("Reusing matrix size of", size, "x", size)
        
        if learningRate == None or exper.askYesNoQuestion("Would you like to enter a new learning rate?"):
            learningRate = exper.getLearningRate()
        else:
            print("Reusing learning rate of", learningRate)

        if maxIterations == None or exper.askYesNoQuestion("Would you like to enter a new value for max iterations?"):
            maxIterations = exper.getMaxIterations()
        else:
            print("Reusing maximum training iterations of", maxIterations)

        if p == None or createPerceptron or exper.askYesNoQuestion("Would you like to generate a new set of samples?"):
            p = per.Perceptron(size * size, learningRate, maxIterations)
            exper.initSamples(p, size)
        else:
            p.initializeRandomWeights()
            p.learningRate = learningRate
            p.maxIterations = maxIterations
            print("\nTraining and Testing samples are being reused.")
            print("Neuron weights were randomized to new values.")

        input("\nPress the Enter key to begin training.")
        print("\nInitial weight vector:", p.weightVector)
        print()
        
        p.train()
        p.test()

        if not exper.askYesNoQuestion("Would you like to run another experiment"):
            break
        
if __name__ == "__main__":
    main()
