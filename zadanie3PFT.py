
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
q = 1
B = 1
m = 1
w = q*B/m
T = 2*np.pi/w
dt= 5*T/iter_num
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
def f1(pr):
   return pr/m
   
@njit  
def f2(pphi,r):
   return pphi/(m*r**2)-q*B*0.5/m

@njit
def f3(pz):
   return pz/m 

@njit
def f4(pphi,r):
   return pphi**2/(m*r**3)-(q**2)*(B**2)*r*0.25/m

@njit
def f5():
   return 0

@njit
def f6():
   return 0
   
@njit
def RK4(r,phi,z,pr,pphi,pz,iter_num):
    k = np.zeros((4,6))
    for i in range(0,iter_num):
          
        k[0][0] = f1(pr[i])
        k[0][1] = f2(pphi[i],r[i])
        k[0][2] = f3(pz[i])
        k[0][3] = f4(pphi[i],r[i])
        k[0][4] = f5()
        k[0][5] = f6()
        
        k[1][0] = f1(pr[i])
        k[1][1] = f2(pphi[i],r[i])
        k[1][2] = f3(pz[i])
        k[1][3] = f4(pphi[i],r[i])
        k[1][4] = f5()
        k[1][5] = f6()
        
        k[2][0] = f1(pr[i])
        k[2][1] = f2(pphi[i],r[i])
        k[2][2] = f3(pz[i])
        k[2][3] = f4(pphi[i],r[i])
        k[2][4] = f5()
        k[2][5] = f6()
        
        k[3][0] = f1(pr[i])
        k[3][1] = f2(pphi[i],r[i])
        k[3][2] = f3(pz[i])
        k[3][3] = f4(pphi[i],r[i])
        k[3][4] = f5()
        k[3][5] = f6()
        
        r[i+1]  = r[i]   + (k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])*dt/6.0
        phi[i+1]= phi[i] + (k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])*dt/6.0
        z[i+1]  = z[i]   + (k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])*dt/6.0
        pr[i+1] = pr[i]  + (k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])*dt/6.0   
        pphi[i+1]=pphi[i]+ (k[0][4]+2*k[1][4]+2*k[2][4]+k[3][4])*dt/6.0
        pz[i+1] = pz[i]  + (k[0][5]+2*k[1][5]+2*k[2][5]+k[3][5])*dt/6.0   
def RK4_check(iter_num):
    stop0 = time.time()
    #deklaracje
    t   = np.linspace(0,iter_num,iter_num+1)
    r   = np.array(t)
    phi = np.array(t)
    z   = np.array(t)
    pr  = np.array(t)
    pphi= np.array(t)
    pz  = np.array(t)   
    
    t  = dt*t
    #warunki początkowe
    r_lst = [1.5,1,2,2]
    pr_lst= [0,0,0,2]
    phi_lst= [1.25*np.pi,0,0,0]
    pphi_lst=[q*B*(r_lst[0]**2)*0.5,-q*B*(r_lst[1]**2)*0.5,-q*B*(r_lst[2]**2)*0.5,-q*B*(r_lst[3]**2)*0.5]
    plt.clf()
    for index in range (0,4):
        r   [0]= r_lst[index]
        phi [0]= phi_lst[index]
        z   [0]= 0
        pr  [0]= pr_lst[index]
        pphi[0]= pphi_lst[index]
        pz  [0]= 0     
        RK4(r,phi,z,pr,pphi,pz,iter_num)
        E = 0.5*(pr**2+(pphi/r)**2)/m-q*B*pphi*0.5/m+0.125*(q*B*r)**2/m
        plt.scatter(t,r,marker=".",linewidths=0.1,label="wp"+str(index))
    #ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right')
    plt.xlabel("t",**csfont)
    plt.ylabel("r",**csfont)
    plt.title("r(t)",**csfont)
    stop1 = time.time()
    plt.show()
    print("Calculation time: ",stop1-stop0,"s.")
    
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")

RK4_check(iter_num)
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()



