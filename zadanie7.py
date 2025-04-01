#parametry wejściowe
N    = 251
M    = 81
dz   = 0.01
mi   = 1
rho  = 1
wait = 100
iter_num = 100000
y1 = 0.4
y2 = -0.4

import time
begin = time.time()
from numba import njit
from numba import jit
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import matplotlib.pyplot as plt
import numpy as np

#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")
    
#importowanie bibliotek
csfont = {'fontname':'Comic Sans MS'}
#rysowanie proste
def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    #plt.show()
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    print("figure  saved")
    sys.stdout.flush()
#rysowanie proste 2 wykresy w jednym
def narysujiporownaj(X,Y,name,labels,Yt):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1,)
    plt.scatter(X,Yt,marker=".",linewidths=0.1,)
    #plt.show()
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    print("figure  saved")
    sys.stdout.flush()


@njit    
def fphi(i,j,Q):
    x = (i - 100)*dz
    y = (j - 40)*dz
    return Q*((y**3)/3-(y**2)*(y1+y2)/2+y*y1*y2)/(2*mi) 
    
@njit
def fzeta(i,j,Q):
    x = (i - 100)*dz
    y = (j - 40)*dz
    return Q*(2*y-y1-y2)/(2*mi)
    
@njit
def zad1(siatkapsi,siatkazeta,iter_num,Q): 
    pastphi = 0
    pastzeta = 0
    breakflag = False
    for k in range(0,iter_num):
        for i in range(0,N):
            for j in range(0,M):
                if i == 0 or j == 0 or i == N-1 or j == M-1:
                    siatkapsi[i][j]  = fphi(i,j,Q)
                    siatkazeta[i][j] = fzeta(i,j,Q)
                    continue
                #cutoff
                if i == 149 and j == 39:
                    if k > wait:
                        if pastphi - siatkapsi[i][j] < 0.0000001 and pastzeta - siatkazeta[i][j] < 0.0000001:
                            breakflag = True

                    pastphi  = siatkapsi[i][j]
                    pastzeta = siatkazeta[i][j]
                siatkapsi[i][j]  = (siatkapsi[i+1][j]+siatkapsi[i-1][j]+siatkapsi[i][j-1]+siatkapsi[i][j+1]-siatkazeta[i][j]*dz**2)/4
                siatkazeta[i][j] = (siatkazeta[i+1][j]+siatkazeta[i-1][j]+siatkazeta[i][j-1]+siatkazeta[i][j+1])/4-((siatkapsi[i][j+1]-siatkapsi[i][j-1])*(siatkazeta[i+1][j]-siatkazeta[i-1][j])-(siatkapsi[i+1][j]-siatkapsi[i-1][j])*(siatkazeta[i][j+1]-siatkazeta[i][j-1]))/16                

        if breakflag:
            
            break
@njit
def zad2(siatkapsi,siatkazeta,iter_num,Q):
    dz= 0.01
    ik = 5
    jk = 10
    pastphi = 0
    pastzeta = 0
    breakflag = False
    #zmienić na zera
    for k in range(0,iter_num):
        for i in range(0,N):
            for j in range(0,M):

                if i == 0 or i == N-1:
                    siatkapsi[i][j]  = fphi(i,j,Q)
                    siatkazeta[i][j] = fzeta(i,j,Q)
                    continue
                elif i > 100 - ik and i < 100 + ik and j>0 and j< jk+40:
                    siatkapsi[i][j]  = np.nan
                    siatkazeta[i][j] = np.nan
                    continue
                elif j == 0:
                    if i > 100 - ik and i < 100 + ik:
                        siatkapsi[i][j]  = np.nan
                        siatkazeta[i][j] = np.nan
                        continue
                    siatkapsi[i][j]  = fphi(i,0,Q) 
                    siatkazeta[i][j] = 2*(siatkapsi[i][j+1]-siatkapsi[i][j])/dz**2
                    continue 
                elif j == M-1:
                    siatkapsi[i][j]  = fphi(i,j,Q) 
                    siatkazeta[i][j] = 2*(siatkapsi[i][j-1]-siatkapsi[i][j])/dz**2
                    continue
                elif i == 100-ik and j < jk+40 :
                    siatkapsi[i][j]  = fphi(i,0,Q) 
                    siatkazeta[i][j] = 2*(siatkapsi[i-1][j]-siatkapsi[i][j])/dz**2
                    continue
                elif i == 100+ik and j < jk+40 :
                    siatkapsi[i][j]  = fphi(i,0,Q) 
                    siatkazeta[i][j] = 2*(siatkapsi[i+1][j]-siatkapsi[i][j])/dz**2
                    continue
                elif j == jk+40:
                   
                    if i > 100 - ik and i < 100 + ik:      
                        siatkapsi[i][j]  = fphi(i,0,Q) 
                        siatkazeta[i][j] = 2*(siatkapsi[i][j+1]-siatkapsi[i][j])/dz**2
                        continue
                    elif i == 100 - ik :
                        siatkapsi[i][j]  = fphi(i,0,Q)
                        siatkazeta[i][j] = ((siatkapsi[i][j+1]-siatkapsi[i][j])/dz**2)+((siatkapsi[i+1][j]-siatkapsi[i][j])/dz**2)
                        continue
                    elif i == 100 + ik:
                        siatkapsi[i][j]  = fphi(i,0,Q)
                        siatkazeta[i][j] = ((siatkapsi[i][j+1]-siatkapsi[i][j])/dz**2)+((siatkapsi[i-1][j]-siatkapsi[i][j])/dz**2)
                        continue
                siatkapsi[i][j]  = (siatkapsi[i+1][j]+siatkapsi[i-1][j]+siatkapsi[i][j-1]+siatkapsi[i][j+1]-siatkazeta[i][j]*dz**2)/4
                siatkazeta[i][j] = (siatkazeta[i+1][j]+siatkazeta[i-1][j]+siatkazeta[i][j-1]+siatkazeta[i][j+1])/4-((siatkapsi[i][j+1]-siatkapsi[i][j-1])*(siatkazeta[i+1][j]-siatkazeta[i-1][j])-(siatkapsi[i+1][j]-siatkapsi[i-1][j])*(siatkazeta[i][j+1]-siatkazeta[i][j-1]))/16                

        if breakflag:
            break

siatkapsi  = np.zeros((N,M))
siatkazeta = np.zeros((N,M))
x = (np.linspace(0,N-1,N)-100)*dz
y = (np.linspace(0,M-1,M)-40)*dz

Q1 = -1
zad1(siatkapsi,siatkazeta,iter_num,Q1)

siatkateoretycznaphi  = np.zeros((N,M))
siatkateoretycznazeta = np.zeros((N,M))
for i in range(0,N):
    for j in range(0,M):
        siatkateoretycznaphi [i][j] = fphi(i,j,Q1)
        siatkateoretycznazeta[i][j] = fzeta(i,j,Q1)

narysujiporownaj(y,siatkapsi[100],u'\u03A8 (0,y)',u'y \u03A8 ',siatkateoretycznaphi [100])
narysujiporownaj(y,siatkazeta[100],u'\u03B6 (0,y)',u'y \u03B6',siatkateoretycznazeta [100])
narysujiporownaj(y,siatkapsi[170],u'\u03A8 (0.7,y)',u'y \u03A8 ',siatkateoretycznaphi [170])
narysujiporownaj(y,siatkazeta[170],u'\u03B6 (0.7,y)',u'y \u03B6',siatkateoretycznazeta [170])
u = -1*(y-y1)*(y-y2)/(2*mi)
narysuj(y,u,"u(y)","y u")

for Q in [-1,-10,-100,-200,-400]:
    siatkapsi  = np.zeros((N,M))
    siatkazeta = np.zeros((N,M))
    zad1(siatkapsi,siatkazeta,iter_num,Q)
    zad2(siatkapsi,siatkazeta,iter_num,Q)
    vx = np.zeros((N-1,M-1))
    vy = np.zeros((N-1,M-1))
    for i in range(0,N-1):
        for j in range(0,M-1):
            vx[i][j] = (siatkapsi[i+1][j]-siatkapsi[i][j])/dz
            vy[i][j] = (siatkapsi[i][j+1]-siatkapsi[i][j])/dz


    plt.clf()
    #transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
    plt.contour(x,y,np.transpose(siatkapsi),20)
    plt.xlabel("x",**csfont)
    plt.ylabel("y",**csfont)
    name = u'\u03A8(x,y)_Q_'+str(Q)
    plt.title(name,**csfont)
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)

    plt.clf()
    #transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
    plt.pcolormesh(x,y,np.transpose(vx))
    plt.colorbar()
    name = "u(x,y)_Q_"+str(Q)
    plt.xlabel("x",**csfont)
    plt.ylabel("y",**csfont)
    plt.title(name,**csfont)
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)

    plt.clf()
    #transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
    plt.pcolormesh(x,y,np.transpose(vy))
    plt.colorbar()
    name = "v(x,y)_Q_"+str(Q)
    plt.xlabel("x",**csfont)
    plt.ylabel("y",**csfont)
    plt.title(name,**csfont)
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)