#deklaracja domyślnych parametrów wejściowych
g = 9.81
R = 1.0
m = 1.0
iter_num = 1000
dt= 0.01 
alpha =1.1
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

csfont = {'fontname':'Comic Sans MS'}
#rysowanie
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

def narysuj_noclear(X,Y,name,labels,lab):


    plt.scatter(X,Y,marker=".",linewidths=0.1,label=str(lab)+" deg")
    #plt.show()

@njit
def f1(theta,z,vtheta,vz):
    return -g*(np.cos(alpha)**2)*np.sin(theta)/(z*np.sin(alpha))-2*vz*vtheta/z
    
@njit    
def f2(theta,z,vtheta,vz):
   (np.sin(alpha)**2)*z*vtheta**2-g*np.sin(alpha)*(np.cos(alpha)**2)*(1-np.cos(theta))

@njit
def RK4(phi,z,vz,vphi,iter_num):
    k = np.zeros((4,4))
    for i in range(0,iter_num):
          
        k[0][0] = vphi[i]
        k[0][1] = vz[i]
        k[0][2] = f1(phi[i],z[i],vphi[i],vz[i])
        k[0][3] = f2(phi[i],z[i],vphi[i],vz[i])
        
        k[1][0] = vphi[i]+dt*k[0][0]*0.5
        k[1][1] = vz[i]+dt*k[0][1]*0.5
        k[1][2] = f1(phi[i]+dt*k[0][2]*0.5,z[i]+dt*k[0][2]*0.5,vphi[i]+dt*k[0][2]*0.5,vz[i]+dt*k[0][2]*0.5)
        k[1][3] = f2(phi[i]+dt*k[0][3]*0.5,z[i]+dt*k[0][3]*0.5,vphi[i]+dt*k[0][3]*0.5,vz[i]+dt*k[0][3]*0.5)
        
        k[2][0] = vphi[i]+dt*k[1][0]*0.5
        k[2][1] = f(phi[i]+dt*k[1][1]*0.5)
        k[2][2] = f1(phi[i]+dt*k[1][2]*0.5,z[i]+dt*k[1][2]*0.5,vphi[i]+dt*k[1][2]*0.5,vz[i]+dt*k[1][2]*0.5)
        k[2][3] = f2(phi[i]+dt*k[1][3]*0.5,z[i]+dt*k[1][3]*0.5,vphi[i]+dt*k[1][3]*0.5,vz[i]+dt*k[1][3]*0.5)
        
        k[3][0] = vphi[i]+dt*k[2][0]
        k[3][1] = f(phi[i]+dt*k[2][1]) 
        k[3][2] = f1(phi[i]+dt*k[2][2],z[i]+dt*k[2][2],vphi[i]+dt*k[2][2],vz[i]+dt*k[2][2])
        k[3][3] = f2(phi[i]+dt*k[2][3],z[i]+dt*k[2][3],vphi[i]+dt*k[2][3],vz[i]+dt*k[2][3])
        
        phi[i+1]  = phi[i]   + (k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])*dt/6.0
        z[i+1]    = z[i]     + (k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])*dt/6.0
        vphi[i+1] = vphi[i]  + (k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])*dt/6.0
        zphi[i+1] = zphi[i]  + (k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])*dt/6.0   
    
def RK4_check(iter_num,param):
    stop0 = time.time()
    #deklaracje
    t  = np.linspace(0,iter_num,int(iter_num+1))
    phi  = np.array(t)
    vphi = np.array(t)
    t = dt*t
    #warunki początkowe
    phi[0] = np.radians(4)
    vphi[0] = 0
    aphi = np.radians(4)*np.cos(np.sqrt(g/R)*t)
    RK4(phi,z,vz,vphi,iter_num)

    """
    t_d  = np.delete(t,slice(i,iter_num+1))
    x_d  = np.delete(x,slice(i,iter_num+1))
    y_d  = np.delete(y,slice(i,iter_num+1))
    """
    gparam = param
    stop1 = time.time()
    print("Calculation time: ",stop1-stop0,"s.")
    aphi = 4*np.pi*np.cos(np.sqrt(g/R)*t)/180
    Ekin = m*(vphi**2)*(R**2)*0.5
    Epot = -m*g*R*np.cos(phi) 
    Ec = Ekin+Epot
    #rysuj
    """
    HANDS OFF
    plt.clf()
    plt.xlabel("t[s]",**csfont)
    plt.ylabel("phi[rad]",**csfont)
    plt.title("phi(t)",**csfont)
    plt.scatter(t,phi,marker=".",linewidths=0.1,label="numerycznie")
    plt.scatter(t,aphi,marker=".",linewidths=0.1,label="analitycznie")
    plt.legend()
    plt.show()
    print("figure phi(t) saved"),
    sys.stdout.flush()
    """
    narysuj(t,phi,"phi(t)_"+param,"t[s] phi[rad]")
    narysuj(t,Ekin,"Ekin(t)_"+param,"t[s] Ekin[J]")
    narysuj(t,Epot,"Epot(t)_"+param,"t[s] Epot[J]")
    narysuj(t,Ec,"Ec(t)_"+param,"t[s] Ec[J]")
    narysuj(phi,vphi,"phi(vphi)_"+param,"vphi[rad/s] phi[rad]")
    plt.clf()
    for i in [4,45,90,135,175]:
        #warunki początkowe
        phi[0] = np.radians(i)
        vphi[0] = np.radians(0)
        RK4(phi,vphi,iter_num)
        stop1 = time.time()
        print("Calculation time: ",stop1-stop0,"s.")
        narysuj_noclear(t,phi,"phi(t)_"+param,"t[s] phi[m]",i)
        
    param =  gparam
    plt.xlabel("t[s]",**csfont)
    plt.ylabel("phi[rad]",**csfont)
    plt.title("phi(t)_"+param,**csfont)
    plt.legend()
    plt.show()
    plt.clf()
    for i in [4,45,90,135,175]:
        #warunki początkowe
        phi[0] = np.radians(i)
        vphi[0] = np.radians(0)
        RK4(phi,vphi,iter_num)
        stop1 = time.time()
        print("Calculation time: ",stop1-stop0,"s.")
        Ekin = m*(vphi**2)*(R**2)*0.5
        narysuj_noclear(t,Ekin,"Ekin(t)_"+param,"t[s] phi[m]",i)
        param =  gparam
    plt.xlabel("t[s]",**csfont)
    plt.ylabel("Ekin[J]",**csfont)
    plt.legend()
    plt.title("Ekin(t)_"+param,**csfont)
    plt.show()
    plt.clf()    
    for i in [4,45,90,135,175]:
        #warunki początkowe
        phi[0] = np.radians(i)
        vphi[0] = np.radians(0)
        RK4(phi,vphi,iter_num)
        stop1 = time.time()
        print("Calculation time: ",stop1-stop0,"s.")
        Epot = -m*g*R*np.cos(phi) 
        narysuj_noclear(t,Epot,"Epot(t)_"+param,"t[s] phi[m]",i)
    param =  gparam
    plt.xlabel("t[s]",**csfont)
    plt.ylabel("Epot[J]",**csfont)
    plt.title("Epot(t)_"+param,**csfont)
    plt.legend()
    plt.show()        
    plt.clf()    
    okres = []
    for i in [4,45,90,135,175]:
        #warunki początkowe

        phi[0] = np.radians(i)
        vphi[0] = np.radians(0)
        RK4(phi,vphi,iter_num)
        stop1 = time.time()
        print("Calculation time: ",stop1-stop0,"s.")
        Ec = m*(vphi**2)*(R**2)*0.5-m*g*R*np.cos(phi)
        narysuj_noclear(t,Ec,"Ec(t)_"+param,"t[s] phi[m]",i)
    param =  gparam
    plt.xlabel("t[s]",**csfont)
    plt.ylabel("Ec[J]",**csfont)
    plt.title("Ec(t)_"+param,**csfont)
    plt.legend()
    plt.show()
    plt.clf()
    for i in [4,45,90,135,175]:
        #warunki początkowe
        phi[0] = np.radians(i)
        vphi[0] = np.radians(0)
        RK4(phi,vphi,iter_num)
        stop1 = time.time()
        print("Calculation time: ",stop1-stop0,"s.")
        Ec = m*(vphi**2)*(R**2)*0.5-m*g*R*np.cos(phi)
        narysuj_noclear(phi,vphi,"vphi(phi)_"+param,"t[s] phi[m]",i)
    param =  gparam
    plt.xlabel("phi[rad]",**csfont)
    plt.ylabel("vphi[rad/s]",**csfont)
    plt.legend()
    plt.title("vphi(phi)_"+param,**csfont)
    plt.show()
    plt.clf()

    plt.scatter([4,45,90,135,175],[2,2.1,2.4,3.3,4.15])
    plt.xlabel("kąt[deg]",**csfont)
    plt.ylabel("okres[t]",**csfont)
    plt.legend()
    plt.title("okres(kąta)",**csfont)
    plt.show()
    plt.clf()
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")
param = str("_dt_")+str(dt)+str("_R_")+str(R)+str("_m_")+str(m)
RK4_check(iter_num,param)
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()
