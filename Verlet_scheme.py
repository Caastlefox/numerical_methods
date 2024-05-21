
t_end    = 5
dx       = 0.01
dt       = 0.005
left     = 0
right    = 1
N        = 101
iter_num = int(t_end/dt)

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
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    print("figure ",name," saved"),
    sys.stdout.flush()

def narysuj_2D_color(X,Y,Z,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.pcolormesh(x,t,u)
    plt.colorbar()
    #plt.show()
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    print("figure ",name," saved"),
    sys.stdout.flush()

@njit
def u0(x):
    return np.exp(-100*(x-0.5)**2)

@njit
def a(x,t,u,beta):
    return ((u[1][x+1]+u[1][x-1]-2*u[1][x])/(dx**2))-2*beta*(u[1][x]-u[0][x])/dt

@njit
def zadanie1(u,beta):
    tmp    = np.zeros((3,N))
    tmp[0] = u[0]
    tmp[1] = u[0]
    for t in range(2,iter_num):
        for x in range(1,N-1):
            tmp[2][x]  = 2*tmp[1][x]-tmp[0][x]+a(x,t,tmp,beta)*dt**2
        tmp[2][0]   = 0
        tmp[2][N-1] = 0
        u[t]      = tmp[2]
        tmp[0]    = tmp[1]
        tmp[1]    = tmp[2]
        
@njit  
def zadanie2(u,beta):
    tmp    = np.zeros((3,N))
    tmp[0] = u[0]
    tmp[1] = u[0]
    tmp[1][0]   = tmp[1][1]
    tmp[1][N-1]   = tmp[1][N-2]

    for t in range(2,iter_num):   
        for x in range(1,N-1):
            tmp[2][x]  = 2*tmp[1][x]-tmp[0][x]+a(x,t,tmp,beta)*dt**2
        tmp[2][0]   = tmp[2][1]
        tmp[2][N-1]   = tmp[2][N-2]
        u[t]        = tmp[2]
        tmp[0]      = tmp[1]
        tmp[1]      = tmp[2]
        
@njit
def zadanie3(u,beta):
    tmp    = np.zeros((3,N))
    tmp[0] = u[0]
    tmp[1] = u[0]
    for t in range(2,iter_num):
        for x in range(0,N):
            tmp[2][x]  = 2*tmp[1][x]-tmp[0][x]+a(x,t,tmp,beta)*dt**2
            tmp[2][0]  = 0
            tmp[2][N-1]= 0
        u[t]   = tmp[2]
        tmp[0] = tmp[1]
        tmp[1] = tmp[2]
        
@njit
def zadanie4(u,beta,x0,wym):
    x0 = int(x0/dx)
    tmp    = np.zeros((3,N))
    tmp[0] = u[0]
    tmp[1] = u[0]
    Tflag = False
    for t in range(2,iter_num):
        for x in range(0,N):
            tmp[2][x]  = 2*tmp[1][x]-tmp[0][x]+a(x,t,tmp,beta)*dt**2
            if x == x0:
                tmp[2][x] += np.cos(wym*t*dt)*dt**2
            tmp[2][0]  = 0
            tmp[2][N-1]= 0
        if tmp[2][x0] < tmp[1][x0]:
            if Tflag:
                Tstart= t*dt
                Tflag = False
        else:
            if  Tflag == False:
                Tstop = t*dt
                Tflag = True
            
        u[t]   = tmp[2]
        tmp[0] = tmp[1]
        tmp[1] = tmp[2]
        
    okres = Tstop-Tstart
    return 2*okres
    
@njit
def zadanie5(u,beta,x0,wym):
    x0 = int(x0/dx)
    tmp    = np.zeros((3,N))
    tmp[0] = u[0]
    tmp[1] = u[0]
    Tflag = False
    for t in range(2,iter_num):
        for x in range(0,N):
            tmp[2][x]  = 2*tmp[1][x]-tmp[0][x]+a(x,t,tmp,beta)*dt**2
            if x == x0:
                tmp[2][x] += np.cos(wym*t*dt)*dt**2
            tmp[2][0]  = 0
            tmp[2][N-1]= 0
        u[t]   = tmp[2]
        tmp[0] = tmp[1]
        tmp[1] = tmp[2]  
    
   

def Energia(u):
    result   = 0
    #for t in range(int((16)/dt),int((20)/dt)):
    for t in range(3600,4000):     
        for x in range(1,N-1):

            result = result+(0.5*(((u[t][x]-u[t-1][x])/dt)**2)+0.5*(((u[t][x+1]-u[t][x-1])/(2*dx))**2)) 
    return 0.25*result

u = np.zeros((iter_num,N))
x = np.linspace(0,N-1,N)
x = dx*x-left
t = np.linspace(0,iter_num-1,iter_num)
t = dt*t

#stiff edges
u[0][0]   = 0
u[0][N-1] = 0
for i in range(1,N-1):
    u[0][i] = u0(i*dx)
#narysuj(x,u[0],"u(x,t)","x t")
beta = 0
zadanie1(u,beta)
narysuj_2D_color(x,t,u,"u(x,t)_zad1","x t")
#loose edges
u = np.zeros((iter_num,N))
for i in range(0,N-1):
    u[0][i] = u0(i*dx)
beta = 0
zadanie2(u,beta)
narysuj_2D_color(x,t,u,"u(x,t)_zad2","x t")
#
for beta in [0.5,2,4]:
    u = np.zeros((iter_num,N))
    for i in range(1,N-2):
        u[0][i] = u0(i*dx)
    u[0][0]   = 0
    u[0][N-1] = 0
    zadanie3(u,beta)
    narysuj_2D_color(x,t,u,"u(x,t)_zad3_"+str(beta),"x t")

#    
t_end    = 15
iter_num = int(t_end/dt)
t = np.linspace(0,iter_num-1,iter_num)
t = dt*t
u = np.zeros((iter_num,N))
beta = 1 
x0   = 0.5
wym  = np.pi*0.5
okres = zadanie4(u,beta,x0,wym)
print("Okres drgan w stanie ustalonym "+str(okres))
narysuj_2D_color(x,t,u,"u(x,t)_zad4","x t")
plt.close()

#    
t_end    = 20
iter_num = int(t_end/dt)

t = np.linspace(t_end,iter_num-1,iter_num)
t = dt*t
E = np.zeros(101)
for wn in range (0,101):
    u = np.zeros((iter_num,N))
    beta = 1 
    x0   = 0.5
    wym  = wn*np.pi*0.1
    zadanie5(u,beta,x0,wym)
    E[wn] = Energia(u)
w = np.linspace(0,100,101)*0.1
narysuj(w,E,"E(w)_1_zad5","w["+u'\u03C0'+"]"+" E")

t_end    = 20
iter_num = int(t_end/dt)
t = np.linspace(0,iter_num,iter_num+1)
t = dt*t
u = np.zeros((iter_num,N))
beta = 1 
x0   = 0.4
E = np.zeros(101)
for wn in range (1,101):
    u = np.zeros((iter_num,N))
    u[0][0]   = 0
    u[0][N-1] = 0
    beta = 1 
    x0   = 0.4
    wym  = wn*np.pi*0.1
    zadanie5(u,beta,x0,wym)
    E[wn] = Energia(u)
w = np.linspace(0,100,101)*0.1

narysuj(w,E,"E(w)_2_zad5","w["+u'\u03C0'+"]"+" E")
end = time.time()
print("total time elapsed"+str(end-begin))
