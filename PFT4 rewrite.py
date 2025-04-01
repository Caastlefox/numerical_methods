
#importowanie bibliotek
import time
begin = time.time()
from numba import njit
from numba import jit
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
#deklaracja domyślnych etrów wejściowych
iter_num = 50000
N = 50
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
def f1(s,i):
    return s[N+1+i]

@njit  
def f2(s,i):
    return alpha*(s[i-1]-2*s[i]+s[i+1])/m
  
@njit
def RK4(s,Ekin,Epot,iter_num):
    k = np.zeros((4,2))
    for t in range(1,iter_num-1):
        s[t+1][0]     = s[t][0]
        s[t+1][N]     = s[t][N]
        s[t+1][N+1]   = s[t][N+1]
        s[t+1][2*N+1] = s[t][2*N+1]
        for is in range(0,)
        for i in range(1,N-1):
            k[0][0] = f1(s[t],i)
            k[0][1] = f2(s[t],i)
            
            k[1][0] = f1(s[t]+dt*k[0][0]*0.5,i)
            k[1][1] = f2(s[t]+dt*k[0][1]*0.5,i)
            
            k[2][0] = f1(s[t]+dt*k[1][0]*0.5,i)
            k[2][1] = f2(s[t]+dt*k[1][1]*0.5,i)
            
            k[3][0] = f1(s[t]+dt*k[2][0],i)
            k[3][1] = f2(s[t]+dt*k[2][1],i)
            
            s[t+1][i]     = s[t][i] + (k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])*dt/6.0
            s[t+1][N+1+i] = s[t][N+1+i] + (k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])*dt/6.0 
            print(Epot)
            Ekin[t+1] += 0.5*m*s[t+1][i]**2 
            Epot[t+1] +=  alpha*0.5*(s[t+1][i-1]-s[t+1][i]+d)**2

def RK4_check(iter_num,s0):
    stop0 = time.time()
    #deklaracje
    t    = np.linspace(0,iter_num-1,iter_num)
    s    = np.zeros((iter_num,2*N+2))
    Ekin = np.zeros(iter_num)
    Epot = np.zeros(iter_num)
    #warunki początkowe
    s[0]  = s0
    
    for i in range(1,N):
        Ekin[0] += 0.5*m*s[0][N+1+i]**2
        Epot[0] += alpha*0.5*(s[0][i-1]-s[0][i]+d)**2
    """    
    t  = dt*t
    RK4(s,Ekin,Epot,iter_num)
    #plt.scatter(t,Ekin,marker=".",linewidths=0.1,label="Ekin")
    plt.scatter(t,Epot,marker=".",linewidths=0.1,label="Epot")
    #    plt.scatter(t,Ekin+Epot,marker=".",linewidths=0.1,label="Ec")
    #ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.xlabel("t",**csfont)
    plt.ylabel("r",**csfont)
    plt.title("r(t)",**csfont)
    
    #stop1 = time.time()
    plt.show()
    """
    stop1 = time.time()
    print("Calculation time: ",stop1-stop0,"s.")
    
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")

s0 = np.zeros(2*N+2)
for i in range(0,N):
    s0[i] = xinit(i*d)
    s0[N+1+i] = 0

RK4_check(iter_num,s0)
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()



