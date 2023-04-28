# Goudi Dwaish
# COT4500-23Spring 0001
# Assignment 3
# Professor Juna Parra

import numpy as np

# Euler Method
def euler(f, t0, y0, h, N):
    for i in range(N):
        y1 = y0 + h * f(t0, y0)
        t1 = t0 + h
        print("t = {:.2f}, y = {:.10f}".format(t1, y1))
        t0 = t1
        y0 = y1

# Runge-Kutta
def runge_kutta(f, t0, y0, h, N):
    for i in range(N):
        k1 = h * f(t0, y0)
        k2 = h * f(t0 + h/2, y0 + k1/2)
        k3 = h * f(t0 + h/2, y0 + k2/2)
        k4 = h * f(t0 + h, y0 + k3)
        y1 = y0 + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        t1 = t0 + h
        print("t = {:.2f}, y = {:.10f}".format(t1, y1))
        t0 = t1
        y0 = y1

# Gaussian elimination and backward substitution
def gauss_elimination(A):
    n = A.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            if A[i,i] == 0:
                raise ValueError("Pivot cannot be zero")
            factor = A[j,i] / A[i,i]
            A[j,:] -= factor * A[i,:]
    return A

def backward_substitution(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if U[i,i] == 0:
            raise ValueError("Matrix is singular")
        x[i] = (b[i] - U[i,i+1:] @ x[i+1:]) / U[i,i]
    return x

# LU Factorization
def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n-1):
        for j in range(i+1, n):
            if U[i,i] == 0:
                raise ValueError("Pivot cannot be zero")
            factor = U[j,i] / U[i,i]
            L[j,i] = factor
            U[j,:] -= factor * U[i,:]
    return L, U

# Test Euler method
f = lambda t, y: t - y**2
t0 = 0
y0 = 1
h = 0.2
N = 10
euler(f, t0, y0, h, N)

# Test Runge-Kutta method
f = lambda t, y: t - y**2
t0 = 0
y0 = 1
h = 0.2
N = 10
runge_kutta(f, t0, y0, h, N)

# Test Gaussian elimination and backward substitution
A = np.array([[2,-1,1,6], [1,3,1,0], [-1,5,4,-3]])
b = A[:, -1]
A = A[:, :-1]
A = gauss_elimination(A)
x = backward_substitution(A, b)
print(x)

# Test LU Factorization
A = np.array([[1,1,0
