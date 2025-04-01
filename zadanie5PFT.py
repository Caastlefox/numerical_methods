#parametry wejściowe
N  = 30
M = 150
drho = 0.1
dz   =0.1
iter_num = 5000
j1 = 60
j2 = 90
V0 = 10
#importowanie bibliotek
from scipy.optimize import curve_fit
import time
begin = time.time()
from numba import njit
from numba import jit
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import numpy as np
import warnings
warnings.filterwarnings('error')
csfont = {'fontname':'Comic Sans MS'}
 

def func1(x):
    return 0.747*x**2- 11.13*x+ 45.544
def func2(x):
    return -0.5426*x**2+0.1578*x+4.0551
def narysuj(X,Y,name,labels,flag):
    
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1,label=name)
    if flag ==3:
        plt.plot(X,func1(X),'r-',label='fit')
        plt.ylim(0, 10)

        plt.legend()
    if flag ==4:
        plt.plot(X,func2(X),'r-',label='fit')
        plt.legend()
    plt.show()
    #plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    #print("figure ",name," saved"),
    sys.stdout.flush()

@njit
def zad1(siatka,iter_num,z,rho):
    V = np.zeros((iter_num,N,M))
    #print(a)
    for k in range(0,iter_num):
        #print(siatka)
        for i in range(1,N-1):
            for j in range(1,M-1):               
                siatka[i][j]= (1/(2/drho**2+2/dz**2))*((siatka[i+1][j]+siatka[i-1][j])/drho**2+(siatka[i+1][j]-siatka[i-1][j])/(drho*2*i*drho)+(siatka[i][j-1]+siatka[i][j+1])/dz**2)
        #Brzegowe
        #1
        for j in range(0,j1):
            siatka[N-1][j] = V0
        #2  
        for j in range(j1,j2):
            siatka[N-1][j] = 0
        #3
        for j in range(j2,M):
            siatka[N-1][j] = V0
        #4  
        for i in range(0,N-1):
            siatka[i][M-1] = siatka[i][M-2]
        #5
        for i in range(0,N-1):
            siatka[i][0] = siatka[i][1]
        #6
        for j in range(0,M-1):
            siatka[0][j] = siatka[1][j]
        V[k] = siatka
    
    #print(a)
    return V

siatka = np.zeros((N,M))
rho = np.zeros((N,M))
z = np.zeros((N,M))

arr_iter_num = np.linspace(0,iter_num-1,iter_num)
#print(arr_iter_num)6

a = zad1(siatka,iter_num,z,rho)
#rysowanie

z = np.linspace(0,M-1,M)*dz
rho = np.linspace(0,N-1,N)*drho
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(rho,z,np.transpose(siatka))
plt.colorbar()
plt.show()
name = "z"
#plt.savefig(os.getcwd()+"/output/"+name+".png")
plt.clf()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(rho,z)
# Plot the surface.
surf = ax.plot_surface(X,Y,np.transpose(siatka), cmap=cm.plasma,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 10)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
narysuj(z,siatka[0],"V(0,z)","z V",3)
narysuj(rho,np.transpose(siatka)[int((M-1)*0.5)],"V(0,z)","z V",4)


"""
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(np.linspace(-N,N,2*(N)),np.linspace(-N,N,2*(N)),np.transpose(rho))
plt.colorbar()
#plt.show()
name = "z-rho"
plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(np.linspace(-N,N,2*(N)),np.linspace(-N,N,2*(N)),np.transpose(siatka))
plt.colorbar()
#plt.show()
name = "a_Rel"
plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)


siatka = np.zeros((2*N,2*N))
a = zad1(siatka,iter_num,z,rho,1.9)
narysuj(arr_iter_num,a,"a_zad_2","liczba_iteracji a")
"""