import numpy as np

class SparseMatrix:
    def __init__(self, matrix, tol=1e-08):
        self.values, self.ind, self.ptr = self.convert_matrix_to_csr(matrix, tol)
        self.intern_represent = 'CSR'
        self.number_of_nonzero = len(self.values)

    def convert_matrix_to_csr(self, matrix, tol):
        values = []
        col_ind = []
        row_ptr = [0]
        for row in matrix:
            values.extend(row[abs(row) > tol].tolist())
            col_ind.extend(np.nonzero(abs(row) > tol)[0].tolist())
            row_ptr.append(len(values))
        return values, col_ind, row_ptr

    def change_element(self, i, j, a_ij):
        start_i = self.ptr[i]
        end_i = self.ptr[i + 1]
        if j in self.ind[start_i:end_i]:
            index = self.ind[start_i:end_i].index(j)
            if a_ij == 0:
                self.values.pop(start_i + index)
                self.ind.pop(start_i + index)
                self.number_of_nonzero -= 1
                self.ptr[(i + 1):] = [x - 1 for x in self.ptr[(i + 1):]]
            else:
                self.values[start_i + index] = a_ij
        else:
            if a_ij != 0:
                index = sorted(self.ind[start_i:end_i] + [j]).index(j)
                self.values.insert((start_i + index), a_ij)
                self.ind.insert((start_i + index), j)
                self.number_of_nonzero += 1
                self.ptr[(i + 1):] = [x + 1 for x in self.ptr[(i + 1):]]

    def CSR_to_CSC(self):
        self.intern_represent = 'CSC'
        csc_val, csc_ind, csc_ptr = [], [], [0]
        n_col = max(self.ind) + 1
        for col in range(n_col):
            col_val_ind = np.where(np.array(self.ind) == col)[0].tolist()
            if col_val_ind:
                csc_val.extend([self.values[ind] for ind in col_val_ind])
                csc_ptr.append(len(csc_val))
                for ind in col_val_ind:
                    csc_ind.append(np.searchsorted(test.ptr, ind, side='right')-1)
            else:
                continue
        self.values, self.ind, self.ptr = csc_val, csc_ind, csc_ptr



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
test2 = test.CSR_to_CSC()
