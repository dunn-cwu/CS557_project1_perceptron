import numpy as np
import perceptron as per

I_MATRIX_1 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
I_MATRIX_2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
I_MATRIX_3 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
L_MATRIX_1 = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]])
L_MATRIX_2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])

(rows, cols) = np.shape(I_MATRIX_1)
p = per.Perceptron(rows * cols, 0.1)

p.addToTrainingSet(I_MATRIX_1, -1)
p.addToTrainingSet(I_MATRIX_2, -1)
p.addToTrainingSet(I_MATRIX_3, -1)
p.addToTrainingSet(L_MATRIX_1, 1)
p.addToTrainingSet(L_MATRIX_2, 1)

p.addToTestingSet(I_MATRIX_1, -1)
p.addToTestingSet(I_MATRIX_2, -1)
p.addToTestingSet(I_MATRIX_3, -1)
p.addToTestingSet(L_MATRIX_1, 1)
p.addToTestingSet(L_MATRIX_2, 1)

print("\nInitial weight vector:", p.weightVector)

p.train()
p.test()
