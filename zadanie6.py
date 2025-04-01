#parametry wejściowe
N  = 201
M  = 101
dx = 1
dy = 1
u0 = 1
iter_num = 100000
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
csfont = {'fontname':'Comic Sans MS'}
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")

def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    #plt.show()
    plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
    #print("figure ",name," saved"),
    sys.stdout.flush()

@njit
def phi(x,y):
    return u0*x
    
@njit
def zad1(siatka,iter_num):
    for i in range(0,N):
        for j in range(0,M):
            siatka[i][j] = u0
    for k in range(0,iter_num-1):
        for i in range(0,N):
            for j in range(0,M):
                if i == 0:
                    siatka[i][j] = u0
                    continue
                elif j == 100:
                    siatka[i][j] = u0*(i+1)
                    continue
                elif i == 200:
                    siatka[i][j] = u0*201
                    continue
                elif j == 0: 
                    if i<94 or i>104:
                        siatka[i][j] = siatka[i][j+1] 
                        continue
                elif i == 94: 
                    if j < 20:
                        siatka[i][j] = siatka[i-1][j]
                        continue
                elif i == 104:
                    if j < 20:
                        siatka[i][j] = siatka[i+1][j]
                        continue
                elif i < 104 and i > 94:
                    if j == 20:
                        siatka[i][j] = siatka[i][j+1]
                        continue
                    elif j < 20:
                        siatka[i][j] = np.nan
                        continue
                elif i == 104 and j == 20:
                    siatka[i][j] = (siatka[i+1][j]+siatka[i][j+1])/2
                    continue
                elif i == 94 and j == 20:
                    siatka[i][j] = (siatka[i-1][j]+siatka[i][j+1])/2
                    continue
                siatka[i][j]= (siatka[i+1][j]+siatka[i-1][j]+siatka[i][j-1]+siatka[i][j+1])/4

@njit
def zad2(siatka,iter_num):
    for k in range(0,iter_num-1):
        for i in range(0,N):
            for j in range(0,M):
                if i == 0:
                    siatka[i][j] = u0*j
                    continue
                elif j == 100:
                    siatka[i][j] = u0*j
                    continue
                elif i == 200:
                    siatka[i][j] = u0*j
                    continue
                elif j == 0:
                    if i<94 or i>104:
                        siatka[i][j] = siatka[0][0] 
                        continue
                elif i == 94: 
                    if j < 20:
                        siatka[i][j] = siatka[0][0]
                        continue
                elif i == 104:
                    if j < 20:
                        siatka[i][j] = siatka[0][0]
                        continue
                elif i < 104 and i > 94:
                    if j == 20:
                        siatka[i][j] = siatka[0][0]
                        continue
                    elif j < 20:
                        siatka[i][j] = np.nan
                        continue
                elif i == 104 and j == 20:
                    siatka[i][j] = siatka[0][0]
                    continue
                elif i == 94 and j == 20:
                    siatka[i][j] = siatka[0][0]
                    continue
                siatka[i][j]= (siatka[i+1][j]+siatka[i-1][j]+siatka[i][j-1]+siatka[i][j+1])/4

siatka = np.zeros((N,M))
zad1(siatka,iter_num)
x = (np.linspace(0,N-1,N)+1)*dx
y = (np.linspace(0,M-1,M)+50)*dy
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.contour(x,y,np.transpose(siatka),40)
plt.xlabel("x",**csfont)
plt.ylabel("y",**csfont)
name = u'\u03A8(x,y)1'
plt.title(name,**csfont)
plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)

siatka = np.zeros((N,M))
zad2(siatka,iter_num)
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.contour(x,y,np.transpose(siatka),40)
plt.xlabel("x",**csfont)
plt.ylabel("y",**csfont)
name = u'\u03A8(x,y)2'
plt.title(name,**csfont)
plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)