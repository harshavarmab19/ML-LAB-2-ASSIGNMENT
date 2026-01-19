import numpy as np

def compute_dot_product(vectorA, vectorB):
    total = 0
    for i in range(len(vectorA)):
        total += vectorA[i] * vectorB[i]
    return total

def compute_euclidean_length(vector):
    squared_sum = 0
    for value in vector:
        squared_sum += value ** 2
    return squared_sum ** 0.5

def main():
    vectorA = np.array([2, 5, 7, 9])
    vectorB = np.array([1, 3, 4, 8])

    dot_manual = compute_dot_product(vectorA, vectorB)
    lengthA_manual = compute_euclidean_length(vectorA)
    lengthB_manual = compute_euclidean_length(vectorB)

    dot_numpy = np.dot(vectorA, vectorB)
    lengthA_numpy = np.linalg.norm(vectorA)
    lengthB_numpy = np.linalg.norm(vectorB)

    print("Manual Dot Product:", dot_manual)
    print("NumPy Dot Product:", dot_numpy)
    print("Manual Euclidean Length of A:", lengthA_manual)
    print("NumPy Euclidean Length of A:", lengthA_numpy)
    print("Manual Euclidean Length of B:", lengthB_manual)
    print("NumPy Euclidean Length of B:", lengthB_numpy)

main()
