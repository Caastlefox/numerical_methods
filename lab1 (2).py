
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
hbar = 1
m = 0.067 
L = 100/0.05292
csfont = {'fontname':'Comic Sans MS'}
@njit
def fpsi(E,V,dx):
    psi = np.zeros(N+1)
    psi[0] = 0
    psi[1] = 1
    for i in range(2,N+1):
        psi[i] = -2*m/(hbar**2)*(E-V)*(dx**2)*psi[i-1]-psi[i-2]+2*psi[i-1]
    return psi
    
@njit
def Csqrt(psi,dx):
    return np.sqrt(dx*np.sum(psi))
@njit
def fast_zeros(Earr,psi):
    miejsca_zerowe = []
    for i in range(1,len(psi)):
        if psi[i-1]*psi[i] > 0:
            continue
        miejsca_zerowe.append([psi[i-1],psi[i],Earr[i-1],Earr[i]])   
    return np.array(miejsca_zerowe)
    
@njit
def bisekcja(Earr,psiN,dx,dokl):
    mzero = fast_zeros(Earr,psiN)
    dok_miejsca_zerowe = []
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    xtmp  = 0 
    ytmp = 0
    for i in range(0,len(mzero)):
        y1 = mzero[i][0]
        y2 = mzero[i][1]
        x1 = mzero[i][2]
        x2 = mzero[i][3]
        while True:
            xtmp = (x2+x1)/2
            ytmp = fpsi(xtmp,V,dx)[N]
            if abs(ytmp) <= dokl:
                dok_miejsca_zerowe.append(xtmp)  
                break
            if ytmp*y1 > 0:
                x1 = xtmp
                y1 = ytmp
            elif ytmp*y2 > 0:
                x2 = xtmp
                y2 = ytmp
                
    return np.array(dok_miejsca_zerowe)

N = 100
dx = L/N
V = 0   

Earr = np.linspace(0,35,701)/27211.6
psiEarr = np.zeros(len(Earr))
index = 0
for E in Earr:
    psitmp = fpsi(E,V,dx)
    psiEarr[index] = psitmp[N]
    index += 1


#plt.clf()
#plt.xlabel("E",**csfont)
#plt.ylabel("psi",**csfont)
#plt.title("psi(E)",**csfont)
#plt.scatter(Earr*27211.6,psiEarr,marker=".",linewidths=0.1)
#plt.scatter(Earr*27211.6,np.zeros(len(Earr)),marker=".",linewidths=0.1)
#plt.show()

##wybrano 5.05 metodą "na oko"
#psiEzero = fpsi(5.05/27211.6,V,dx)
#psiEzerogora = fpsi(5.3/27211.6,V,dx)
#psiEzerodol = fpsi(4.8/27211.6,V,dx)
#Csrodek = Csqrt(psiEzero,dx)
#Cgora   = Csqrt(psiEzerogora,dx)
#Cdol    = Csqrt(psiEzerodol,dx)
#psiEzero /= Csrodek
#psiEzerogora /= Cgora
#psiEzerodol /= Cdol

#xarr = np.linspace(0,N,N+1)
#plt.clf()
#plt.xlabel("x",**csfont)
#plt.ylabel("psi",**csfont)
#plt.title("psi(x)",**csfont)
#plt.scatter(xarr,psiEzero,marker=".",linewidths=0.1,label="5.05 meV")
#plt.scatter(xarr,psiEzerogora,marker=".",linewidths=0.1,label="5.3 meV")
#plt.scatter(xarr,psiEzerodol,marker=".",linewidths=0.1,label="4.8 meV")
#plt.legend()
#plt.show()


Eteor = (np.linspace(1,7,7)**2)*(3.14**2)/(2*m*L**2)
dokl = 0.000001
mzero = bisekcja(Earr,psiEarr,dx,dokl)
print(mzero)
print(Eteor)
