import numpy as np
import perceptron as per
import input_gen as igen

# I_MATRIX_1 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
# I_MATRIX_2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
# I_MATRIX_3 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
# L_MATRIX_1 = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]])
# L_MATRIX_2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]])

# (rows, cols) = np.shape(I_MATRIX_1)
# p = per.Perceptron(rows * cols, 0.1)

size = int(input("Input size of matrix (minimum 3): "))
if size < 3:
    print("Error: Size cannot be less than 3")
    exit(1)

p = per.Perceptron(size * size, 0.1)
gen = igen.InputGenerator(size)

print("Generating samples ...")

I_samples = gen.generate_I_samples(3, 3)
L_samples = gen.generate_L_samples(3, 3, 2, 2)

print("Generated", len(I_samples), " 'I' samples.")
print("Generated", len(L_samples), " 'L' samples.")

print("Example 'I' sample:")
print(I_samples[0])

print("\nExample 'L' sample:")
print(L_samples[0])

for s in I_samples:
    p.addToTrainingSet(s, -1)
    p.addToTestingSet(s, -1)

for s in L_samples:
    p.addToTrainingSet(s, 1)
    p.addToTestingSet(s, 1)

print("\nInitial weight vector:", p.weightVector)
print()

p.train()
p.test()
