import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import matplotlib.pyplot as plt
import time

class SparseMatrix:
    def __init__(self, matrix, tol=1e-08):
        # read the wikipedia article to understand what ind & start_ind are
        self.val, self.ind, self.start_ind = self.convert_matrix_to_csr(matrix, tol)
        self.normalize_indices()
        self.intern_represent = 'CSR'
        self.number_of_nonzero = len(self.val)
        self.tol = tol
        # To not lose dim while changing representation and check matching dim operations.
        self.shape = matrix.shape

    def convert_matrix_to_csr(self, matrix, tol):
        val = [] # list to store non-zero values
        col_ind = [] # stores the column index of its corresponding non-zero value
        row_start_ind = [0] # stores the index of val/col_ind where each row starts & ends
        for row in matrix:
            val.extend(row[abs(row) > tol].tolist()) # add non_zero values
            col_ind.extend(np.nonzero(abs(row) > tol)[0].tolist()) # add corresponding col index
            row_start_ind.append(len(val)) # store where the row ends
        return val, col_ind, row_start_ind
    
    def normalize_indices(self):
        # Ensure all indices are plain integers, avoid np.int64
        self.ind = [int(i) for i in self.ind]

    def change_element(self, i, j, a_ij):  # a_ij is the value to be inserted, i is row, j is col
        # Raise an error if (i, j) is outside of dim.
        if i+1 > self.shape[0] or j+1 > self.shape[1]:
            raise IndexError("Trying to insert value outside of the matrix dimension.")
        if self.intern_represent == 'CSR':
            # CSR: start/end i is row index, target_index is column index
            start_i = self.start_ind[i]  # start of row i in val and ind
            end_i = self.start_ind[i + 1]  # end of row i in val and ind
            indices = self.ind[start_i:end_i] # extract the relevant column indices for row i
            target_index = j  # column index in CSR
        else:
            # CSC: start/end i is column index, target_index is row index
            start_i = self.start_ind[j]  # start of column j in val and ind
            end_i = self.start_ind[j + 1]  # end of column j in val and ind
            indices = self.ind[start_i:end_i] # extract the relevant row indices for column j
            target_index = i  # row index in CSC

        if target_index in indices:
            index = indices.index(target_index) # The position in the relevant row (CSC) or columc (CSR)
            if a_ij < abs(self.tol):
                # Remove the element
                self.val.pop(start_i + index) # start_i + the position in the relevant row/column 
                self.ind.pop(start_i + index) # is the absolute position in the val & ind arrays
                self.number_of_nonzero -= 1
                # Update start_ind for subsequent rows/columns
                self.start_ind[(i + 1):] = [x - 1 for x in self.start_ind[(i + 1):]]
            else:
                # Update the value
                self.val[start_i + index] = a_ij
        else:
            if a_ij > abs(self.tol):
                # Insert the new element
                insert_position = start_i + np.searchsorted(indices, target_index)
                self.val.insert(insert_position, a_ij)
                self.ind.insert(insert_position, target_index)
                self.number_of_nonzero += 1
                # Update start_ind for subsequent rows/columns
                self.start_ind[(i + 1):] = [x + 1 for x in self.start_ind[(i + 1):]]


    def change_representation(self):
        val, ind, start_ind = [], [], [0]
        n_col = max(self.ind) + 1
        for col in range(n_col):
            col_val_ind = np.where(np.array(self.ind) == col)[0].tolist()
            if col_val_ind:
                val.extend([self.val[ind] for ind in col_val_ind])
                start_ind.append(len(val))
                for c_ind in col_val_ind:
                    ind.append(np.searchsorted(self.start_ind, c_ind, side='right')-1)
            else:
                continue
        if self.intern_represent == 'CSR':
            if len(start_ind) != self.shape[1] + 1:
                start_ind = start_ind+[start_ind[-1]]*(self.shape[1]-len(start_ind) + 1)
            self.intern_represent = 'CSC'
        else:
            if len(start_ind) != self.shape[0] + 1:
                start_ind = start_ind+[start_ind[-1]]*(self.shape[0]-len(start_ind) + 1)
            self.intern_represent = 'CSR'
        self.val, self.ind, self.start_ind = val, ind, start_ind
        self.normalize_indices()

    def __eq__(self, other):
        # False if dim does not match.
        if self.shape != other.shape:
            _bool = False
        else:
            changed = False
            if self.intern_represent != other.intern_represent:
                other.change_representation()
                changed = True
            _bool = (self.start_ind == other.start_ind) & (self.ind == other.ind) \
                            & (self.val == other.val)
            if (changed):
                other.change_representation()
        return _bool

    def __add__(self, other):
        # Raise an error if the dimensions do not match
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same dimensions.")

        changed = False
        if self.intern_represent != other.intern_represent:
            other.change_representation()
            changed = True
        if self.intern_represent == 'CSR':
            n_rows_self = len(self.start_ind) - 1
            n_cols_self = max(self.ind) + 1 if self.ind else 0
        else:  # CSC
            n_cols_self = len(self.start_ind) - 1
            n_rows_self = max(self.ind) + 1 if self.ind else 0
        val = []  # List to store the values of the result matrix
        ind = []  # List to store the indices of the result matrix
        start_ind = [0]  # List to store the start indices for the result matrix

        if self.intern_represent == 'CSR':
            # Iterate over each row
            for i in range(n_rows_self):
                row_dict = {}
                # Add elements from the self matrix to the row_dict
                for j in range(self.start_ind[i], self.start_ind[i + 1]):
                    col = self.ind[j]
                    row_dict[col] = self.val[j]
                # Add elements from the other matrix to the row_dict
                for j in range(other.start_ind[i], other.start_ind[i + 1]):
                    col = other.ind[j]
                    if col in row_dict:
                        row_dict[col] += other.val[j]
                    else:
                        row_dict[col] = other.val[j]
                # Append the combined row to the val and ind lists
                for col in sorted(row_dict.keys()):
                    if abs(row_dict[col]) > self.tol: # if added values are close to 0 make them 0
                        val.append(row_dict[col])
                        ind.append(col)
                start_ind.append(len(val))
        else:  # CSC
            # Iterate over each column
            for i in range(n_cols_self):
                col_dict = {}
                # Add elements from the self matrix to the col_dict
                for j in range(self.start_ind[i], self.start_ind[i + 1]):
                    row = self.ind[j]
                    col_dict[row] = self.val[j]
                # Add elements from the other matrix to the col_dict
                for j in range(other.start_ind[i], other.start_ind[i + 1]):
                    row = other.ind[j]
                    if row in col_dict:
                        col_dict[row] += other.val[j]
                    else:
                        col_dict[row] = other.val[j]
                # Append the combined column to the val and ind lists
                for row in sorted(col_dict.keys()):
                    if abs(col_dict[row]) > 1e-08:  # if added values are close to 0 make them 0
                        val.append(col_dict[row])
                        ind.append(row)
                start_ind.append(len(val))

        # Create a new SparseMatrix object with the combined data
        result = SparseMatrix.__new__(SparseMatrix)
        result.val = val
        result.ind = ind
        result.start_ind = start_ind
        result.intern_represent = self.intern_represent
        result.shape = self.shape
        result.number_of_nonzero = len(val)

        # Restore the original representation of the other matrix if it was changed
        if changed:
            other.change_representation()

        return result

    # Task 8
    def __mul__(self, vector):
        if self.shape[1] != len(vector):
            raise ValueError("The length of the column is different from the vector")
        result = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            row_start = self.start_ind[i] 
            row_end = self.start_ind[i + 1]
            for j in range(row_start,row_end):
               result[i] += self.val[j] * vector[self.ind[j]]
        return result

    @classmethod
    def toeplitz(cls, n):
        # First row inserted.
        val = [2, -1]  # list to store non-zero values,
        ind = [0, 1]  # stores the column index of its corresponding non-zero value.
        start_ind = [0, 2]  # stores the index of val/col_ind where each row starts & ends.
        for i in range(n-2):
            val.extend([-1, 2, -1])
            ind.extend([i, i+1, i+2])
            start_ind.append(start_ind[-1]+3)
        val.extend([-1, 2]) # Last row inserted
        ind.extend([n-2, n-1])
        start_ind.append(3*n-2)

        # Create the new SparseMatrix object.
        result = SparseMatrix.__new__(SparseMatrix)
        result.val = val
        result.ind = ind
        result.start_ind = start_ind
        result.intern_represent = 'CSR'
        result.shape = (n, n)
        result.number_of_nonzero = len(val)
        result.tol = 1e-08 # redundant
        result.normalize_indices()
        return result

    def print_dense(self):
        n_rows, n_cols = self.shape

        # init with all zeros
        dense_matrix = np.zeros((n_rows, n_cols))

        if self.intern_represent == 'CSR':
            for i in range(n_rows):
                # Iterate over the range of indices for the current row
                for j in range(self.start_ind[i], self.start_ind[i + 1]):
                    col = self.ind[j] # Column index for the non-zero element
                    dense_matrix[i, col] = self.val[j]
        else:
            for i in range(n_cols):
                # Iterate over the range of indices for the current column
                for j in range(self.start_ind[i], self.start_ind[i + 1]):
                    row = self.ind[j] # Row index for the non-zero element
                    dense_matrix[row, i] = self.val[j]
        print(dense_matrix)

    def print_internal_arrays(self):
        print(f"val: {self.val}")
        print(f"ind: {self.ind}")
        print(f"start_ind: {self.start_ind}")



# Task 10
def task10():
    # Test with small matrices
    matrixSmall1 = np.array([[10, 20, 0, 0, 0, 0, 0],
                             [0, 30, 0, 40, 0, 0, 0],
                             [0, 0, 50, 60, 70, 0, 0],
                             [0, 0, 0, 0, 0, 80, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

    matrixSmall2 = np.array([[10, 20, 0, 0, 0, 0, 0],
                             [0, 30, 0, 40, 0, 0, 0],
                             [0, 0, 50, 60, 70, 0, 0],
                             [0, 0, 0, 0, 0, 80, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
    
    vector = [1,2,3,4,5,6,7]

    matrixSparse1 = SparseMatrix(matrixSmall1)
    matrixSparse2 = SparseMatrix(matrixSmall2)

    # Prints dense matrix
    print("Print the two small matrices in dense format\n")
    matrixSparse1.print_dense()
    print("\n")
    matrixSparse2.print_dense()

    # Prints internal arrays
    print("\n\n\nPrints internal arrays\n")
    matrixSparse1.print_internal_arrays()

    # Test whether these two sparce matrices are the same
    print("\n\n\nTest whether these two sparce matrices are the same\n")
    print(matrixSparse1 == matrixSparse2)

    # Change element and print the new dense matrix
    print("\n\n\nChange element and print the new dense matrix\n")
    matrixSparse1.change_element(2, 6, 88)
    matrixSparse1.print_dense()

    # Test whether these two sparce matrices still are the same
    print("\n\n\nTest whether these two sparce matrices still are the same\n")
    print(matrixSparse1 == matrixSparse2)

    # Test representation, change it and then test again
    print("\n\n\nTest representation, change it and then test again\n")
    print(matrixSparse1.intern_represent)
    matrixSparse1.print_internal_arrays()
    print("\n\n\nNew representation\n")
    matrixSparse1.change_representation()
    print(matrixSparse1.intern_represent)
    matrixSparse1.print_internal_arrays()

    # Add two sparse matrices together
    print("\n\n\nAdd two sparse matrices together\n")
    matrixSum = matrixSparse1 + matrixSparse2
    matrixSum.print_dense()

    # Multiply a sparse matrix with a vector
    print("\n\n\nMultiply a sparse matrix with a vector\n")
    product = matrixSparse2 * vector
    print(product)

    # Test tolerance init-method, one matrix with a value above and one below
    print("\n\n\nTest tolerance init-method, one matrix with a value above and one below\n")
    matrixSmall4 = np.array([[1e-10, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]])
    matrixSparse4 = SparseMatrix(matrixSmall4)
    matrixSparse4.print_dense()
    print("\n")
    matrixSparse4.change_element(0, 0, 1e-7)
    matrixSparse4.print_dense()


    ###########################
    
    # Test with toeplitz matrixes    

    toeplitz10 = SparseMatrix.toeplitz(10)
    toeplitz100 = SparseMatrix.toeplitz(100)
    toeplitz10000 = SparseMatrix.toeplitz(10000)

    
    # Prints dense matrix
    print("\n\n\nPrints the sparse toeplitz in dense format\n\ntoeplitz10:")
    toeplitz10.print_dense()
    print("\n\ntoeplitz100:")
    toeplitz100.print_dense()

    # Prints internal arrays
    print("\n\n\nPrints internal arrays\n")
    toeplitz10.print_internal_arrays()

    # Test whether these two sparce matrices are the same
    print("\n\n\nTest whether toeplitz10000 is equal to itself\n")
    print(toeplitz10000 == toeplitz10000)
    print("\n\n\nTest whether toeplitz10000 is equal to toeplitz100\n")
    print(toeplitz10000 == toeplitz100)
    
    # add toeplitz
    print("\n\n\nTesting adding two toeplitz10 matrices together\n")
    toeplitzSum = toeplitz10 + toeplitz10
    toeplitzSum.print_dense()

    # multiply toeplitz
    print("\n\n\nTesting multiplying toeplitz10 and 100 with random vectors of their respective length\n")
    toeplitzProduct10 = toeplitz10 * np.random.randint(0,10,10)
    toeplitzProduct100 = toeplitz100 * np.random.randint(0,10,100)
    print(f"{toeplitzProduct10}\n\n\n{toeplitzProduct100}\n\n\n")

    # Change element and print the new dense matrix
    print("Change the element in row 3, col 7 to 88 and print the new matrix in dense format\n")
    toeplitz10.change_element(2, 6, 88)
    toeplitz10.print_dense()
    print("\n\n\n")


# Task 11
def task11():
    def testAddElement(var, i, j, e):
        if isinstance(var, SparseMatrix) == True:
            start = time.time()
            var.change_element(i, j, e)
            stop = time.time()
        else:
            start = time.time()
            var[i, j] = e
            stop = time.time()
        return stop - start

    def testSum(var1, var2):
        start = time.time()
        var1 + var2
        stop = time.time()
        return stop - start
    
    def testMul(var,vector):      
        start = time.time()
        var * vector
        stop = time.time()
        return stop-start

    # Test with small matrices
    testMatrix1 = np.array([[20, 40, 0, 0, 5, 0],
                            [0, 60, 0, 80, 0, 0],
                            [0, 0, 100, 12, 140, 0],
                            [0, 234, 0, 0, 0, 16],
                            [0, 0, 0, 0, 0, 0],
                            [7, 0, 0, 3, 0, 0]])

    testMatrix2 = np.array([[2, 5, 2, 0, 77, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 14, 0],
                            [0, 234, 0, 0, 0, 6],
                            [0, 0, 0, 0, 5, 0],
                            [2, 0, 0, 378, 0, 3]])
    
    testVector = [1,2,3,4,5,6]

    sparseTest1 = SparseMatrix(testMatrix1)
    sparseTest2 = SparseMatrix(testMatrix2)
    scipyTest1 = csr_matrix(testMatrix1)
    scipyTest2 = csr_matrix(testMatrix2)

    # Insert element
    sparseElement = testAddElement(sparseTest1, 1, 5, 100)
    scipyElement = testAddElement(scipyTest1, 1, 5, 100)

    # Sum of matrices
    sparseSum = testSum(sparseTest1, sparseTest2)
    scipySum = testSum(scipyTest1, scipyTest2)

    # Multiply matrix with vector
    sparseMul = testMul(sparseTest1, testVector)
    scipyMul = testMul(scipyTest1, testVector)

    print(f"Insert element time\nSparseMatrix: {sparseElement}\nScipy: {scipyElement}\n\nSum of matrices time\nSparseMatrix: {sparseSum}\nScipy: {scipySum}\n\nMultiplication with vector\nSparseMatrix: {sparseMul}\nScipy: {scipyMul}")


    # Matplotlib
    resultsSparseNew = []
    resultsScipyNew = []
    resultsSparseSum = []
    resultsScipySum = []
    resultsSparseMul = []
    resultsScipyMul = []
    xAxis = list(range(2,100))

    for n in range(2, 100):
        toeplitzSparse = SparseMatrix.toeplitz(n) # Creates a toeplitzmatrix with SparseMatrix
        
        diagonal = [2 * np.ones(n), -1 * np.ones(n - 1), -1 * np.ones(n - 1)]
        offsets = [0, -1, 1]
        toeplitzScipy = diags(diagonal, offsets, shape=(n, n), format='csr')  # Creates a toeplitzmatrix with scipy
        
        vector = np.random.randint(0,100,n) # Creates a vector of length n

        # Add a new element
        timeSparseNew = testAddElement(toeplitzSparse,0,0,100)
        resultsSparseNew.append(timeSparseNew)
        timeScipyNew = testAddElement(toeplitzScipy,0,0,100)
        resultsScipyNew.append(timeScipyNew)

        # Add two matrices
        timeSparseSum = testSum(toeplitzSparse, toeplitzSparse)
        resultsSparseSum.append(timeSparseSum)
        timeScipySum = testSum(toeplitzScipy, toeplitzScipy)
        resultsScipySum.append(timeScipySum)

        # Multiply with vector
        timeSparseMul = testMul(toeplitzSparse, vector)
        resultsSparseMul.append(timeSparseMul)
        timeScipyMul = testMul(toeplitzScipy, vector)
        resultsScipyMul.append(timeScipyMul)


    plt.plot(xAxis, resultsScipyNew)
    plt.plot(xAxis,resultsSparseNew)
    plt.legend(["ScipyNew","SparseNew"])
    plt.xlabel("Matrix size")
    plt.ylabel("Time")
    plt.show()

    plt.plot(xAxis, resultsScipySum)
    plt.plot(xAxis, resultsSparseSum)
    plt.legend(["ScipySum","SparseSum"])
    plt.xlabel("Matrix size")
    plt.ylabel("Time")
    plt.show()

    plt.plot(xAxis, resultsScipyMul)
    plt.plot(xAxis, resultsSparseMul)
    plt.legend(["ScipyMul","SparseMul"])
    plt.xlabel("Matrix size")
    plt.ylabel("Time")
    plt.show()



