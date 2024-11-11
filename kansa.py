import numpy as np
import pygeodesic.geodesic as geodesic
import math


class Kansa():
    def __init__(self, vertices, faces, L, F, N_i, kernel = "multiquadric", epsilon = 1, coef = 1):
        self.vertices = vertices
        self.faces = faces
        self.N = len(vertices)
        self.N_i = N_i
        self.L = L
        self.F = F
        self.kernel = kernel
        self.epsilon = epsilon
        self.is_singular = False
        self.weights = np.zeros(self.N)
        self.coefMat = np.zeros((self.N, self.N))
        self.coef = coef
        
        self.geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)

        Kansa.__BuildRBF(self)

    def __BuildRBF(self):
        
        tradMat = np.zeros((self.N, self.N))

        for i in range(self.N):
            for j in range(self.N):
                tradMat[i][j] = Kansa.__Kernel(self, Kansa.__Metric(self, i, j))
        
        upper = np.matmul(self.L, tradMat)

        for i in range(self.N_i):
            for j in range(self.N):
                self.coefMat[i][j] = self.coef*upper[i][j]
        
        """for i in range(self.N_i):
            row = np.array([Kansa.__Kernel(self, Kansa.__Metric(self, i, j)) for j in range(self.N)])
            row = np.matmul(self.L, row)

            for j in range(self.N):
                self.coefMat[i][j] = self.coef*row[j]"""
        
        for i in range(self.N_i, self.N):
            for j in range(self.N):
                self.coefMat[i][j] = Kansa.__Kernel(self, Kansa.__Metric(self, i, j))

        try:
            self.weights = np.linalg.solve(self.coefMat, self.F)
            print("solved system of equations")
        except:
            print("ill-conditioned matrix (coefficient matrix not full rank)")
            self.is_singular = True
            return


    def Interpolate(self, point):
        out = 0
        for i in range(self.N):
            d = Kansa.__Metric(self, point, i)
            out += Kansa.__Kernel(self, d)*self.weights[i]
            
        return out

    def __Metric(self, i, j):
        return self.geoalg.geodesicDistance(i, j)[0]
    
    def __Kernel(self, r):
        if self.kernel == "linear":
            return self.epsilon*r
        elif self.kernel == "cubic":
            return self.epsilon*r**3
        elif self.kernel == "thin_plate_spline":
            if r == 0:
                return 0
            else:
                return self.epsilon*r**2 * np.log(r)
        elif self.kernel == "quintic":
            return self.epsilon*r**5
        elif self.kernel == "gaussian":
            return np.exp(-1.0*(self.epsilon*r)**2)
        elif self.kernel == "multiquadric":
            return -1.0*np.sqrt(1 + (self.epsilon*r)**2)
        elif self.kernel == "inverse_quadric":
            return 1/(1 + (self.epsilon*r)**2)

    
        
            



