import numpy as np
#from scipy.sparse import csr_matrix

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
            # CSR: i is row index, j is column index
            start_i = self.start_ind[i]  # start of row i
            end_i = self.start_ind[i + 1]  # end of row i
            indices = self.ind[start_i:end_i]
            target_index = j  # column index in CSR
        else:
            # CSC: i is column index, j is row index
            start_i = self.start_ind[j]  # start of column j
            end_i = self.start_ind[j + 1]  # end of column j
            indices = self.ind[start_i:end_i]
            target_index = i  # row index in CSC

        if target_index in indices:
            index = indices.index(target_index)
            if a_ij == 0:
                # Remove the element
                self.val.pop(start_i + index)
                self.ind.pop(start_i + index)
                self.number_of_nonzero -= 1
                # Update start_ind for subsequent rows/columns
                self.start_ind[(i + 1):] = [x - 1 for x in self.start_ind[(i + 1):]]
            else:
                # Update the value
                self.val[start_i + index] = a_ij
        else:
            if a_ij != 0:
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
                    if abs(row_dict[col]) > self.tol:  # if added values are close to 0 make them 0
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
    
    def __str__(self):
        # bad implementation
        print(self.intern_represent)
        self.print_internal_arrays()
        self.print_dense()
        return ""



################################

# test area.

# Test change elements.
matrix1 = np.array([[10, 20, 0, 0, 0, 0, 0],
                    [0, 30, 0, 40, 0, 0, 0],
                    [0, 0, 50, 60, 70, 0, 0],
                    [0, 0, 0, 0, 0, 80, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]

test1 = SparseMatrix(matrix1)

test2 = SparseMatrix(matrix1)

test2.change_element(1, 1, 10)

print(test1.intern_represent)

test2.change_representation()

print(test2 + test1)

