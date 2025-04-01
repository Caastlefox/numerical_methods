#deklaracja domyślnych etrów wejściowych
g = 9.81
iter_num = 5000
dt= 0.01 
alpha =0.75
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

csfont = {'fontname':'Comic Sans MS','size':16}
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
   return (np.sin(alpha)**2)*z*vtheta**2-g*np.sin(alpha)*(np.cos(alpha)**2)*(1-np.cos(theta))

@njit
def RK4(phi,z,vz,vphi,iter_num):
    k = np.zeros((4,4))
    for i in range(0,iter_num):
          
        k[0][0] = vphi[i]
        k[0][1] = vz[i]
        k[0][2] = f1(phi[i],z[i],vphi[i],vz[i])
        k[0][3] = f2(phi[i],z[i],vphi[i],vz[i])
        
        k[1][0] = vphi[i]  +dt*k[0][2]*0.5
        k[1][1] = vz[i]    +dt*k[0][3]*0.5
        k[1][2] = f1(phi[i]+dt*k[0][0]*0.5,z[i]+dt*k[0][1]*0.5,vphi[i]+dt*k[0][2]*0.5,vz[i]+dt*k[0][3]*0.5)
        k[1][3] = f2(phi[i]+dt*k[0][0]*0.5,z[i]+dt*k[0][1]*0.5,vphi[i]+dt*k[0][2]*0.5,vz[i]+dt*k[0][3]*0.5)
        
        k[2][0] = vphi[i]  +dt*k[1][2]*0.5
        k[2][1] = vz[i]    +dt*k[1][3]*0.5
        k[2][2] = f1(phi[i]+dt*k[1][0]*0.5,z[i]+dt*k[1][1]*0.5,vphi[i]+dt*k[1][2]*0.5,vz[i]+dt*k[1][3]*0.5)
        k[2][3] = f2(phi[i]+dt*k[1][0]*0.5,z[i]+dt*k[1][1]*0.5,vphi[i]+dt*k[1][2]*0.5,vz[i]+dt*k[1][3]*0.5)
        
        k[3][0] = vphi[i]  +dt*k[2][2]
        k[3][1] = vz[i]    +dt*k[2][3] 
        k[3][2] = f1(phi[i]+dt*k[2][0],z[i]+dt*k[2][1],vphi[i]+dt*k[2][2],vz[i]+dt*k[2][3])
        k[3][3] = f2(phi[i]+dt*k[2][0],z[i]+dt*k[2][1],vphi[i]+dt*k[2][2],vz[i]+dt*k[2][3])
        
        phi[i+1]  = phi[i]   + (k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])*dt/6.0
        z[i+1]    = z[i]     + (k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])*dt/6.0
        vphi[i+1] = vphi[i]  + (k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])*dt/6.0
        vz[i+1] = vz[i]      + (k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])*dt/6.0   
    
def RK4_check(iter_num):
    stop0 = time.time()
    #deklaracje
    t  = np.linspace(0,iter_num,int(iter_num+1))
    phi  = np.array(t)
    z    = np.array(t)
    vphi = np.array(t)
    vz   = np.array(t)
    t = dt*t
    #warunki początkowe
    phi[0]  = 9.1
    z[0]    = 5.0
    vphi[0] = 1.6
    vz[0]   = -0.8
    RK4(phi,z,vz,vphi,iter_num)
    
    stop1 = time.time()
    print("Calculation time: ",stop1-stop0,"s.")
    
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
    narysuj(t,phi,"phi(t)","t[s] phi[rad]")
    narysuj(t,z,"z(t)","t[s] z[m]")
    narysuj(t,vphi,"vphi(t)","t[s] prędkość_kątowa[rad/s]")
    narysuj(t,vz,"vz(t)","t[s] prędkość_w_osi_z[m/s]")
    E = 0.5*(np.tan(alpha)**2*z**2*vphi**2+(vz/np.cos(alpha))**2)+g*z*np.sin(alpha)*(1-np.cos(phi))
    narysuj(t,E,"E(t)","t[s] Energia[J]")
    
    plt.clf()
    #przemnożyć razy macierz (wstawić do równania(14))
    r = z*np.tan(alpha)
    xp = r  * np.sin(alpha) * np.cos(phi) + z*np.cos(alpha)
    yp = r  * np.sin(phi)
    zp = -r * np.cos(alpha) * np.cos(phi) + z*np.sin(alpha)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(xp, yp, zp, label='krzywa parametryczna')
    ax.set_xlabel("X",**csfont)
    ax.set_ylabel("Y",**csfont)
    ax.set_zlabel("Z",**csfont)
    ax.legend(prop={'family':'Comic Sans MS','size':20})
    plt.show()
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")

RK4_check(iter_num)
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()



