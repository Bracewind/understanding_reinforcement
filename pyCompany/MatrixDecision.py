import numpy as np
class MatrixDecision(object):
    def __init__(self, nbProducts):
        self.matrix_np = np.ones((nbProducts, 2))
        self.matrix_np[0][0] = 1
        self.matrix_np[0][1] = 20000
        self.matrix_np[1][0] = 0
        self.matrix_np[1][1] = 1
        self.matrix_np[2][0] = 0
        self.matrix_np[2][1] = 10
        self.matrix_np[3][0] = 0
        self.matrix_np[3][1] = 100


