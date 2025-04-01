import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt

Lin = 100 #m
Lout = 250 #m
L = 2000
A = 1.5 #m2
U = 0.1 #m/s
D = 1 #m2/s
m = 10 #kg
dx = 10
N  = int(L/dx + 1) 
dt = 0.5
iter_num = 10001
Ca = U*dt/dx
Cd = D*dt/(dx**2)
Cin = m/(A*dx)
def zad1():
    siatka = np.zeros((iter_num,N))
    siatka[0][int(Lin/dx)] = Cin #poczatkowy
    for t in range(0,iter_num-1):
        #brzegowe
        siatka[t][N-1] = 0
        siatka[t][0] = 0
        siatka[t][1] = 0
        for x in range(2,N-1):
            siatka[t+1][x] += siatka[t][x]
            siatka[t+1][x] += -(Cd*(2 - 3*Ca) -(Ca/2)*(Ca**2-2*Ca - 1))*siatka[t][x]
            siatka[t+1][x] += (Cd*(1-Ca) - (Ca/6)*(Ca**2 - 3*Ca + 2))*siatka[t][x+1] 
            siatka[t+1][x] += (Cd*(1 - 3*Ca) - (Ca/2)*(Ca**2 - Ca - 2))*siatka[t][x-1]
            siatka[t+1][x] += (Cd*Ca+(Ca/6)*(Ca**2-1))*siatka[t][x-2] 
    return siatka

def sum_test(siatka):
    return np.sum(siatka)

begin = time.time()
sol = zad1()
end = time.time()
print(end-begin)
print(A*dx*sum_test(sol[iter_num-1]))#can find a trigger begin or end
print(A*dx*Cin)
xarr = np.linspace(0,N-1,N)*dx
plt.clf()
plt.scatter(xarr,sol[10000])
name = "zanieczyszczenie po 10000 iteracjach"
plt.xlabel("dlugosc[m]")
plt.ylabel("zanieczyszczenie[g/m^3]")
plt.title(name)
plt.show()
plt.clf()
plt.scatter(xarr,sol[1000])
name = "zanieczyszczenie po 10000 iteracjach"
plt.xlabel("dlugosc[m]")
plt.ylabel("zanieczyszczenie[g/m^3]")
plt.title(name)
plt.show()

#dobrac
listt = [70,74,78,82,86,90,94,98,102,106,110,114,118,122,126,130,134,138,142,146,150]
listc = [0,0.100375,0.290868,0.439789,0.442709,0.410586,0.355104,0.306039,0.224063,0.186628,0.14765,0.115656,0.087498,0.056595,0.047782,0.035593,0.017682,0.01648,0.008527,0.005379,0.004922]
# x = 2000
# m = 10 Br