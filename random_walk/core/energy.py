import numpy as np
from abc import ABC, abstractmethod

class EnergyModel(ABC):
    def _check_x(self, x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=1:
            raise ValueError("x must be a 1D position vector")
        return x
    @abstractmethod
    def energy(self, x):
        pass
    @abstractmethod
    def gradient(self,x):
        pass
    def hessian(self, x):
        x = self._check_x(x)
        n = len(x)
        h = 1e-5
        H = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pp[i] += h
                x_pp[j] += h
                x_pm = x.copy()
                x_pm[i] += h
                x_pm[j] -= h
                x_mp = x.copy()
                x_mp[i] -= h
                x_mp[j] += h
                x_mm = x.copy()
                x_mm[i] -= h
                x_mm[j] -= h
                H[i,j]=(self.energy(x_pp)-self.energy(x_pm) - self.energy(x_mp)+self.energy(x_mm))/(4 * h**2)
        return H


class Lennard(EnergyModel):
    def __init__(self, epsilon=1.0, sigma=1.0, r_min=1e-12):
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.r_min = float(r_min)
    def energy(self, x):
         x = self._check_x(x)
         r = np.linalg.norm(x)
         if r<self.r_min:
             return np.inf
         
         s_by_r=self.sigma/r
         s_by_r6=s_by_r**6

         return 4*self.epsilon*(s_by_r6**2 -s_by_r6)
    

    def gradient(self, x):
        x = self._check_x(x)
        r = np.linalg.norm(x)
        if r < self.r_min:
            return np.zeros_like(x)
        s_by_r = self.sigma / r
        s_by_r6 = s_by_r ** 6
        dE_dr=24*self.epsilon/r*(2*s_by_r6**2-s_by_r6)
        return dE_dr*(x/r)
    

class harmonic(EnergyModel):
    def __init__(self, k=1.0):
        if k<=0:
            raise ValueError("k must be positive")
        self.k=float(k)
    
    def energy(self, x):
        x = self._check_x(x)
        return 0.5 * self.k * np.sum(x**2)
    
    def gradient(self, x):
        x = self._check_x(x)
        return self.k * x
    def hessian(self, x):
        x = self._check_x(x)
        n = len(x)
        return self.k * np.eye(n)
    


class DoubleWellEnergy(EnergyModel):
    def __init__(self, a=1.0, b=1.0):
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive")
        self.a = float(a)
        self.b = float(b)
    
    def energy(self, x):
        x = self._check_x(x)
        r2 = np.sum(x**2)
        return self.a * (r2 - self.b**2) ** 2
    
    def gradient(self, x):
        x = self._check_x(x)
        r2 = np.sum(x**2)
        return 4 * self.a * (r2 - self.b**2) * x
    
class QuarticEnergy(EnergyModel):
    def energy(self, x):
        x = self._check_x(x)
        return np.sum(x**4)
    def gradient(self, x):
        x = self._check_x(x)
        return 4 * x**3
    

class RosenbrockEnergy(EnergyModel):
    def __init__(self, a=1.0, b=100.0):
        self.a = float(a)
        self.b = float(b)
    
    def energy(self, x):
        x = self._check_x(x)
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2
    
    def gradient(self, x):
        x = self._check_x(x)
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        grad = np.zeros(2)
        grad[0] = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0]**2)
        grad[1] = 2 * self.b * (x[1] - x[0]**2)
        return grad