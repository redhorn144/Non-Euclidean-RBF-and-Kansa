import numpy as np
import pygeodesic.geodesic as geodesic
import math

class MRBF():
    def __init__(self, vertices, faces, fs, vertexIndsToInclude, kernel = 'linear', epsilon = 1):
        self.inds = vertexIndsToInclude
        self.vertices = vertices
        self.faces = faces
        self.fs = fs
        self.n = len(self.inds)
        self.fsReduced = MRBF.__MakeFsReduced(self)
        print(f"interpolating with {self.n} points")
        self.kernel = kernel
        self.epsilon = epsilon

        self.geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        self.weights = np.zeros(self.n)
        self.coefMat = np.zeros((self.n, self.n))
        self.is_singular = False
        MRBF.__BuildRBF(self)

    def __MakeFsReduced(self):
        fsReduced = np.zeros(self.n)
        for i in range(self.n):
            fsReduced[i] = self.fs[self.inds[i]]
        return fsReduced

    def __BuildRBF(self):

        for i in range(self.n):
            for j in range(i):
                d, p =  MRBF.metric(self, self.inds[i], self.inds[j])
                phi = MRBF.__kernel(self, d)
                self.coefMat[i][j] = phi
                if i != j:
                    self.coefMat[j][i] = phi
        print(f"built interpolation matrix of size {self.coefMat.shape}, solving system. . . ")
        try:
            self.weights = np.linalg.solve(self.coefMat, self.fsReduced)
            print("solved system of equations")
        except:
            print("ill-conditioned matrix (coefficient matrix not full rank)")
            self.is_singular = True
            return

    def Interpolate(self, point):
        out = 0
        for i in range(self.n):
            d, p = MRBF.metric(self, point, self.inds[i])
            out += MRBF.__kernel(self, d)*self.weights[i]
            
        return out

    #metric between vertex i and j
    def metric(self, i, j):
        return self.geoalg.geodesicDistance(i, j)

    def __kernel(self, r):
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