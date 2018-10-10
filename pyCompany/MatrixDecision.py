import numpy as np
class MatrixDecision(object):
    def __init__(self, nbProducts, vector):
        ### first part of the vector represents the first action
        vector.resize([ nbProducts, 2])
        self.matrix_np = vector



