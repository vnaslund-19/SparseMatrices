import numpy as np

class SparseMatrix:
    def __init__(self, matrix, tol=1e-08):
        # read the wikipedia article to understand what ind & start_ind are
        self.val, self.ind, self.start_ind = self.convert_matrix_to_csr(matrix, tol)
        self.normalize_indices()
        self.intern_represent = 'CSR'
        self.number_of_nonzero = len(self.val)

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
        if self.intern_represent == 'CSR':
            self.intern_represent = 'CSC'
        else:
            self.intern_represent = 'CSR'

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
        self.val, self.ind, self.start_ind = val, ind, start_ind
        self.normalize_indices()

    def __eq__(self, other):
        changed = False
        if self.intern_represent != other.intern_represent:
            other.change_representation()
            changed = True
        _bool = (self.start_ind == other.start_ind) & (self.ind == other.ind) \
                        & (self.val == other.val)
        if (changed):
            other.change_representation()
        return _bool

    def print_dense(self):
        if self.intern_represent == 'CSR':
            n_rows = len(self.start_ind) - 1
            n_cols = max(self.ind) + 1 if self.ind else 0
        else:
            n_cols = len(self.start_ind) - 1
            n_rows = max(self.ind) + 1 if self.ind else 0

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
matrix1 = np.array([[10, 20, 0, 0, 0, 0], 
                    [0, 30, 0, 40, 0, 0], 
                    [0, 0, 50, 60, 70, 0],
                    [0, 0, 0, 0, 0, 80]])

test1 = SparseMatrix(matrix1)

test2 = SparseMatrix(matrix1)

test2.change_representation()

print(test1.intern_represent)

test1.change_representation()
test2.change_representation()

print(test2 == test1)

print(test1.intern_represent)


