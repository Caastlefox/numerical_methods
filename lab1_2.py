import numpy as np
np.set_printoptions(threshold=np.inf)

from numba import njit
import time
import matplotlib.pyplot as plt
H    = 5
dz   = 0.01 #dla 0.008 następuje overflow dla 0.009 błąd numeryczny minimum 0.01 nie ma górnej granicy
dt   = 60#overflow dla 100, błąd numeryczny conajmniej od 85 do 80 brak błędu od dołu nie ma granicy
N    = int(H/dz + 1) 
iter_num = 1000
Tm = 8
Talpha = 5
alpha = 6*10**(-7)#m^2/s
P = 24*3600#s
D = np.sqrt(alpha*P/np.pi)#m

@njit
def T0(t):
    return Tm + Talpha * np.sin(2*np.pi*t/P)

@njit
def zad1():
    r = alpha*dt/(dz**2)
    siatka = np.zeros((iter_num,N))
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    for i in range(0,N):
        siatka[0][i] = 8

    for t in range(1,iter_num):
        siatka[t][0] = T0(t*dt)
        siatka[t][N-1] = 8
        A[0][0] = 1
        A[N-1][N-1] = 1  
        B[0] = 1
        B[N-1] = 1     
        for i in range(1,N-1):
            A[i][i-1] = -r/2
            A[i][i]   =  1+r
            A[i][i+1] = -r/2
            B[i] =  (r/2)*siatka[t][i-1]+(1-r)*siatka[t][i]+(r/2)*siatka[t][i+1]
        tmparr = np.linalg.solve(A,B)
        #check array indexing not to go line by line for efficiency     
        for i in range(1,N-1):
            siatka[t][i] = tmparr[i]
              
    return siatka

begin = time.time()
sol = zad1()

tarr = np.linspace(0,iter_num-1,iter_num)/60
xarr = np.linspace(0,N-1,N)*dz*-1
X,Y = np.meshgrid(tarr,xarr)

plt.clf()
plt.pcolormesh(X,Y,np.transpose(sol))
plt.colorbar()
name = "T(x,t)"
plt.xlabel("t")
plt.ylabel("z")
plt.title(name)
plt.show()

end = time.time()
print(end-begin)