#parametry wejściowe
N  = 31
d  = 4
x0 = 4
dx = 1
iter_num = 500 
#importowanie bibliotek
import time
begin = time.time()
from numba import njit
from numba import jit
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('error')
csfont = {'fontname':'Comic Sans MS'}

def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    #plt.show()
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    #print("figure ",name," saved"),
    sys.stdout.flush()

@njit
def rho_lad(x,y):
    if x == -N or y == -N or x == N or y == N:
        return 0
    return np.exp(-((x-x0)**2+y**2)/d**2)-np.exp(-((x+x0)**2+y**2)/d**2)  
@njit
def rho_rev(i,j,siatka):

    return (siatka[i+1][j]+siatka[i-1][j]+siatka[i][j+1]+siatka[i][j-1]-4*siatka[i][j])/(dx**2)
    
@njit    
def fa(siatka,i,j):    
    return (0.5*siatka[i][j]*(siatka[i+1][j]+siatka[i-1][j]-4*siatka[i][j]+siatka[i][j+1]+siatka[i][j-1]))+rho_lad(i-N,j-N)*siatka[i][j]*(dx**2)
    
@njit
def zad1(siatka,iter_num,rho_r,rho,w=1):
    a = np.zeros(iter_num)
    #print(a)
    for k in range(0,iter_num-1):
        #print(siatka)
        for i in range(0,2*N):
            for j in range(0,2*N):
                if i == 0 or j == 0 or i == 2*N-1 or j == 2*N-1:
                    siatka[i][j] = 0
                    continue
                siatka[i][j]= (1-w)*siatka[i][j]+w*(siatka[i+1][j]+siatka[i-1][j]+siatka[i][j-1]+siatka[i][j+1]+rho_lad(i-N,j-N)*(dx**2))/4
                a[k+1] = a[k+1]-fa(siatka,i,j)

    for i in range(0,2*N):
        for j in range(0,2*N):  
            rho_r[i][j] = -rho_rev(i,j,siatka)
            rho[i][j]   = -rho_lad(i-N,j-N)-rho_rev(i,j,siatka)
    #print(a)
    return a
"""
@njit
def zad3(siatka,iter_num,rho_r,rho):
    a0 = np.zeros(iter_num)
    for k in range(0,iter_num-1):
        for i in range(0,2*N):
            for j in range(0,2*N):
                if i == 0 or j == 0 or i == 2*N-1 or j == 2*N-1:
                    siatka[i][j] = 0
                    continue
                siatka[i][j]= 
                a0[k+1] = a0[k+1]-fa(siatka,i,j)
                ag = a0-aloc+aloc
    return a
"""
siatka = np.zeros((2*N,2*N))
rho = np.zeros((2*N,2*N))
rho_r = np.zeros((2*N,2*N))

arr_iter_num = np.linspace(0,iter_num-1,iter_num)
#print(arr_iter_num)
#add heatmap
a = zad1(siatka,iter_num,rho_r,rho)

narysuj(arr_iter_num,a,"a","liczba_iteracji a")
#H = np.linspace(-N,N,2*(N)+1)


plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(np.linspace(-N,N,2*(N)),np.linspace(-N,N,2*(N)),np.transpose(rho_r))
plt.colorbar()
#plt.show()
name = "rho_r"
plt.savefig(os.getcwd()+"\\output\\"+name+".png")

plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(np.linspace(-N,N,2*(N)),np.linspace(-N,N,2*(N)),np.transpose(rho))
plt.colorbar()
#plt.show()
name = "rho_r-rho"
plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(np.linspace(-N,N,2*(N)),np.linspace(-N,N,2*(N)),np.transpose(siatka))
plt.colorbar()
#plt.show()
name = "a_Rel"
plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)


siatka = np.zeros((2*N,2*N))
a = zad1(siatka,iter_num,rho_r,rho,1.9)
narysuj(arr_iter_num,a,"a_zad_2","liczba_iteracji a")