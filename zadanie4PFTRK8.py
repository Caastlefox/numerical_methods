
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
from matplotlib import colormaps
#deklaracja domyślnych etrów wejściowych
iter_num = 50000
N = 51
dt = 0.002
d = 0.1
alpha = 1
m = 1
xmax = N*d
csfont = {'fontname':'Comic Sans MS','size':16}
#rysowanie
def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    plt.show()
    #plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    print("figure ",name," saved"),
    sys.stdout.flush()
@njit
def xinit(x):
    return x+d*np.exp(-((x-xmax/2)**2)/(2*(3*d)**2))/3

@njit
def vinit(x):
    return 0

@njit
def f1(vx,i):
    if i == 0 or i == N-1:
        return 0
    return vx[i]

@njit  
def f2(x,i):
    if i == 0 or i == N-1:
        return 0
    return alpha*(x[i-1]-2*x[i]+x[i+1])/m
  
@njit
def RK4(x,vx,Ekin,Epot,iter_num):
    k = np.zeros((N,2))

    for t in range(0,iter_num):
        for i in range(0,N):
            
            k[0][0] = f1(vx[t],i)
            k[0][1] = f2(x[t],i)
            
            k[1][0] = f1(vx[t]+dt*k[0][0]*0.5,i)
            k[1][1] = f2(x[t] +dt*k[0][1]*0.5,i)
            
            k[2][0] = f1(vx[t]+dt*k[1][0]*0.5,i)
            k[2][1] = f2(x[t] +dt*k[1][1]*0.5,i)
            
            k[3][0] = f1(vx[t]+dt*k[2][0],i)
            k[3][1] = f2(x[t] +dt*k[2][1],i)
            
            x[t+1][i]  = x[t][i]  + (k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])*dt/6.0
            vx[t+1][i] = vx[t][i] + (k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])*dt/6.0        
            Ekin[t+1] = Ekin[t+1] + 0.5*m*vx[t+1][i]**2 
            if i == 0:
                continue
            Epot[t+1] = Epot[t+1]+alpha*0.5*(x[t+1][i-1]-x[t+1][i]+d)**2

def RK4_check(iter_num,x0,vx0):
    stop0 = time.time()
    #deklaracje
    t   = np.linspace(0,iter_num-1,iter_num)
    x   = np.zeros((iter_num,N))
    vx = np.zeros((iter_num,N))
    Ekin = np.zeros(iter_num)
    Epot = np.zeros(iter_num)
    #warunki początkowe
    x[0]  = x0
    vx[0] = vx0
    
    for i in range(1,N):
        Ekin[0] += 0.5*m*vx[0][i]**2
        Epot[0] += alpha*0.5*(x[0][i-1]-x[0][i]+d)**2
        
    t  = dt*t
    RK4(x,vx,Ekin,Epot,iter_num)
    plt.scatter(t,Ekin,marker=".",linewidths=0.1,label="Ekin")
    plt.scatter(t,Epot,marker=".",linewidths=0.1,label="Epot")
    plt.scatter(t,Ekin+Epot,marker=".",linewidths=0.1,label="Ec")
    #ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.xlabel("t",**csfont)
    plt.ylabel("r",**csfont)
    plt.title("r(t)",**csfont)
    #stop1 = time.time()
    plt.show()
    plt.clf()
    xcolor = np.zeros((N,iter_num))
    for i in range (0,N):
        xcolor[i]=d*i-np.transpose(x)[i]
    print(xcolor)
    #plt.scatter(t,np.transpose(x)[1],marker=".",linewidths=0.1,label="Ec")
    plt.pcolormesh(np.linspace(0,N-1,N),t*dt,np.transpose(xcolor),shading='nearest')
    plt.colorbar()
    plt.show()
    stop1 = time.time()
    print("Calculation time: ",stop1-stop0,"s.")
    
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")

x0 = np.zeros(N)
vx0= np.zeros(N)
for i in range(0,N):
    x0[i] = xinit(i*d)
    vx0[i] = vinit(i*d)

RK4_check(iter_num,x0,vx0)
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()
