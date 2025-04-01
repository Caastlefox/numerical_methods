#parametry wejściowe
N    = 251
M    = 81
dz   = 0.01
mi   = 1
rho  = 1
wait = 100
iter_num = 10000
y1 = 0.4
y2 = -0.4

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

def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    plt.show()
    #plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
    #print("figure ",name," saved"),
    sys.stdout.flush()

def narysujiporownaj(X,Y,name,labels,Yt):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    plt.scatter(X,Yt,marker=".",linewidths=0.1)
    plt.show()
    #plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
    #print("figure ",name," saved"),
    sys.stdout.flush()


@njit    
def fphi(i,j):
    x = (i - 100)*dz
    y = (j - 40)*dz
    return Q*((y**3)/3-(y**2)*(y1+y2)/2+y*y1*y2)/(2*mi) 
    
@njit
def fzeta(i,j):
    x = (i - 100)*dz
    y = (j - 40)*dz
    return Q*(2*y-y1-y2)/(2*mi)
    
@njit
def zad1(siatkaphi,siatkazeta,iter_num,Q): 
    pastphi = 0
    pastzeta = 0
    breakflag = False
    for k in range(0,iter_num):
        for i in range(0,N):
            for j in range(0,M):
                

                if i == 0 or j == 0 or i == N-1 or j == M-1:
                    siatkaphi[i][j]  = fphi(i,j)
                    siatkazeta[i][j] = fzeta(i,j)
                    continue
                if i == 149 and j == 39:
                    if k > wait:
                        if pastphi - siatkaphi[i][j] < 0.0000001 and pastzeta - siatkazeta[i][j] < 0.0000001:
                            breakflag = True

                    pastphi  = siatkaphi[i][j]
                    pastzeta = siatkazeta[i][j]
                siatkaphi[i][j]  = (siatkaphi[i+1][j]+siatkaphi[i-1][j]+siatkaphi[i][j-1]+siatkaphi[i][j+1]-siatkazeta[i][j]*dz**2)/4
                siatkazeta[i][j] = (siatkazeta[i+1][j]+siatkazeta[i-1][j]+siatkazeta[i][j-1]+siatkazeta[i][j+1])/4-((siatkaphi[i][j+1]-siatkaphi[i][j-1])*(siatkazeta[i+1][j]-siatkazeta[i-1][j])-(siatkaphi[i+1][j]-siatkaphi[i-1][j])*(siatkazeta[i][j+1]-siatkazeta[i][j-1]))/16                

        if breakflag:
            
            break
@njit
def zad2(siatkaphi,siatkazeta,iter_num,Q):
    dz= 0.01
    ik = 5
    jk = 10
    pastphi = 0
    pastzeta = 0
    breakflag = False
    for k in range(0,iter_num):
        for i in range(0,N):
            for j in range(0,M):
                if j == 0:
                    if i > 100 - ik and i < 100 + ik:
                        continue
                    siatkaphi[i][j]  = fphi(i,j) 
                    siatkazeta[i][j] = 2*(siatkaphi[i][j+1]-siatkaphi[i][j])/dz**2
                    continue
                elif j == M-1:
                    siatkaphi[i][j]  = fphi(i,j) 
                    siatkazeta[i][j] = 2*(siatkaphi[i][j-1]-siatkaphi[i][j])/dz**2
                    continue
                elif i == 100-ik and j < jk+40 :
                    siatkaphi[i][j]  = fphi(i,j) 
                    siatkazeta[i][j] = 2*(siatkaphi[i-1][j]-siatkaphi[i][j])/dz**2
                    continue
                elif i == 100+ik and j < jk+40 :
                    siatkaphi[i][j]  = fphi(i,j) 
                    siatkazeta[i][j] = 2*(siatkaphi[i+1][j]-siatkaphi[i][j])/dz**2
                    continue
                elif j == jk+40:
                    if i > 100 - ik and i < 100 + ik:      
                        siatkaphi[i][j]  = fphi(i,j) 
                        siatkazeta[i][j] = 2*(siatkaphi[i][j+1]-siatkaphi[i][j])/dz**2
                        continue
                    elif i == 100 - ik :
                        siatkaphi[i][j]  = 0.5 *(siatkaphi[i][j-1]+siatkaphi[i+1][j])
                        siatkazeta[i][j] = 0.5 *(siatkazeta[i][j-1]+siatkazeta[i+1][j])
                        continue
                    elif i == 100 + ik:
                        siatkaphi[i][j]  = 0.5 *(siatkaphi[i][j-1]-siatkaphi[i-1][j])
                        siatkazeta[i][j] = 0.5 *(siatkazeta[i][j-1]-siatkazeta[i-1][j])
                        continue
                if i == 149 and j == 39:
                    if k > wait:
                        if pastphi - siatkaphi[i][j] < 0.0000001 and pastzeta - siatkazeta[i][j] < 0.0000001:
                            breakflag = True
                    pastphi  = siatkaphi[i][j]
                    pastzeta = siatkazeta[i][j]
                siatkaphi[i][j]  = (siatkaphi[i+1][j]+siatkaphi[i-1][j]+siatkaphi[i][j-1]+siatkaphi[i][j+1]-siatkazeta[i][j]*dz**2)/4
                siatkazeta[i][j] = (siatkazeta[i+1][j]+siatkazeta[i-1][j]+siatkazeta[i][j-1]+siatkazeta[i][j+1])/4-((siatkaphi[i][j+1]-siatkaphi[i][j-1])*(siatkazeta[i+1][j]-siatkazeta[i-1][j])-(siatkaphi[i+1][j]-siatkaphi[i-1][j])*(siatkazeta[i][j+1]-siatkazeta[i][j-1]))/16                

        if breakflag:
          
            break
        print(siatkaphi[10][10])
siatkaphi  = np.zeros((N,M))
siatkazeta = np.zeros((N,M))

y = (np.linspace(0,M-1,M)-40)*dz

Q = -1
zad1(siatkaphi,siatkazeta,iter_num,Q)

siatkateoretycznaphi  = np.zeros((N,M))
siatkateoretycznazeta = np.zeros((N,M))
for i in range(0,N):
    for j in range(0,M):
        siatkateoretycznaphi [i][j] = fphi(i,j)
        siatkateoretycznazeta[i][j] = fzeta(i,j)
        
narysujiporownaj(y,siatkaphi[100],"y","y phi",siatkateoretycznaphi [100])
narysujiporownaj(y,siatkazeta[100],"y","y zeta",siatkateoretycznazeta [100])
narysujiporownaj(y,siatkaphi[170],"y","y phi",siatkateoretycznaphi [100])
narysujiporownaj(y,siatkazeta[170],"y","y zeta",siatkateoretycznazeta [100])
u = Q*(y-y1)*(y-y2)/(2*mi)
narysuj(y,u,"u(y)","y predkosc")

Q = -1

zad2(siatkaphi,siatkazeta,iter_num,Q)
#nie działa zwaraca NaN powód nieznany
print(siatkaphi)
vx = np.zeros((N-1,M-1))
vy = np.zeros((N-1,M-1))
for i in range(0,N-1):
    for j in range(0,M-1):
        vx[i][j] = (siatkaphi[i+1][j]-siatkaphi[i][j])/dz
        vy[i][j] = (siatkaphi[i][j+1]-siatkaphi[i][j])/dz


plt.clf()
plt.streamplot((np.linspace(0,N-2,N-1)-100)*dz,(np.linspace(0,M-2,M-1)-40)*dz,np.transpose(vx),np.transpose(vy))

plt.show()


"""

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
"""