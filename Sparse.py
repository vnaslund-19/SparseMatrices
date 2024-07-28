import numpy as np

class SparseMatrix:
    def __init__(self, matrix, tol=1e-08):
        self.val, self.ind, self.ptr = self.convert_matrix_to_csr(matrix, tol)
        self.intern_represent = 'CSR'
        self.number_of_nonzero = len(self.val)

    def convert_matrix_to_csr(self, matrix, tol):
        val = []
        col_ind = []
        row_ptr = [0]
        for row in matrix:
            val.extend(row[abs(row) > tol].tolist())
            col_ind.extend(np.nonzero(abs(row) > tol)[0].tolist())
            row_ptr.append(len(val))
        return val, col_ind, row_ptr

    def change_element(self, i, j, a_ij):
        start_i = self.ptr[i]
        end_i = self.ptr[i + 1]
        if j in self.ind[start_i:end_i]:
            index = self.ind[start_i:end_i].index(j)
            if a_ij == 0:
                self.val.pop(start_i + index)
                self.ind.pop(start_i + index)
                self.number_of_nonzero -= 1
                self.ptr[(i + 1):] = [x - 1 for x in self.ptr[(i + 1):]]
            else:
                self.val[start_i + index] = a_ij
        else:
            if a_ij != 0:
                index = sorted(self.ind[start_i:end_i] + [j]).index(j)
                self.val.insert((start_i + index), a_ij)
                self.ind.insert((start_i + index), j)
                self.number_of_nonzero += 1
                self.ptr[(i + 1):] = [x + 1 for x in self.ptr[(i + 1):]]

    def change_representation(self):
        if self.intern_represent == 'CSR':
            self.intern_represent = 'CSC'
        else:
            self.intern_represent = 'CSR'
        val, ind, ptr = [], [], [0]
        n_col = max(self.ind) + 1
        for col in range(n_col):
            col_val_ind = np.where(np.array(self.ind) == col)[0].tolist()
            if col_val_ind:
                val.extend([self.val[ind] for ind in col_val_ind])
                ptr.append(len(val))
                for c_ind in col_val_ind:
                    ind.append(np.searchsorted(test.ptr, c_ind, side='right')-1)
            else:
                continue
        self.val, self.ind, self.ptr = val, ind, ptr

    def __eq__(self, other):
        if self.intern_represent == 'CSR':
            if other.intern_represent != 'CSR':
                other.change_representation()
        else:
            if other.intern_represent != 'CSC':
                other.change_representation()
        _bool = (self.ptr == other.ptr) & (self.ind == other.ind) \
                        & (self.val == other.val)
        return _bool





################################

# test area.

# Test change elements.
matrix = np.array([[10, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0],
                   [0, 0, 0, 0, 0, 80]])
test = SparseMatrix(matrix)
test.change_element(0, 0, 5)
test.change_element(3, 0, 12)

#
matrix = np.array([[10, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0],
                   [0, 0, 0, 0, 0, 80]])
test = SparseMatrix(matrix)
test.change_representation()

#
matrix = np.array([[10, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0],
                   [0, 0, 0, 0, 0, 80]])
test = SparseMatrix(matrix)
test1 = SparseMatrix(matrix)
print(test == test2)

matrix1 = np.array([[11, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0],
                   [0, 0, 0, 0, 0, 80]])
test2 = SparseMatrix(matrix1)
test == test2
