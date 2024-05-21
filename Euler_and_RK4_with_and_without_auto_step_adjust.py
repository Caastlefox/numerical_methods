
mass = 1.989e30 #int
dt = 3600 #int sekundy
r0 = 149597870700*0.586 #int
v0 = 54600 #int
G = 6.6741e-11#int
obroty = 3 #
tol = 1000

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
#draw
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
    

@njit   
def Euler1(x,y,vx,vy,iter_num,obroty):
    rev_count = 0
    neg_flag = False
    #Explicit Euler
    for i in range(0,iter_num):
        x[i+1]  = x[i]  + vx[i]*dt
        y[i+1]  = y[i]  + vy[i]*dt
        vx[i+1] = vx[i] - G*mass*x[i]*dt/(np.sqrt((x[i])**2+(y[i])**2))**3
        vy[i+1] = vy[i] - G*mass*y[i]*dt/(np.sqrt((x[i])**2+(y[i])**2))**3
        if neg_flag:
            if x[i+1] > 0:
                rev_count += 1
                neg_flag = False         
                if rev_count == obroty:
                    iteration = i
                    break      
            continue
        if x[i+1] < 0:
            neg_flag = True
    return iteration
    
def zadanie_1(iter_num,r0,v0,param,obroty):
    stop0 = time.time()

    t  = np.array(np.linspace(0,iter_num,iter_num+1))    
    x  = np.array(t)
    y  = np.array(t)
    vx = np.array(t)
    vy = np.array(t)

    x[0] = 0
    y[0] = r0
    vx[0]= v0
    vy[0]= 0   
    i  = Euler1(x,y,vx,vy,iter_num,obroty)

    t_d  = np.delete(t,slice(i,iter_num+1))
    x_d  = np.delete(x,slice(i,iter_num+1))
    y_d  = np.delete(y,slice(i,iter_num+1))
    stop1 = time.time()
    print("Zadanie1 Calculation time: ",stop1-stop0,"s.")

    param = str("Zadanie_1") + param 
    narysuj(x_d,y_d,"y(x)_"+param,"x[m] y[m]")
    narysuj(t_d,x_d,"x(t)_"+param,"x[m] y[m]")
    stop2 = time.time()
    print("Zadanie1 Drawing time: ",stop2-stop1,"s.")

@njit
def RK4(x,y,vx,vy,iter_num,obroty):
    k = np.zeros((4,4))
    u = np.zeros((4))
    rev_count = 0
    neg_flag = False
    #RK4
    for i in range(0,iter_num):
        
        k[0][0] = vx[i]
        k[0][1] = vy[i]
        k[0][2] = -G*mass*x[i]/(np.sqrt((x[i])**2+(y[i])**2))**3
        k[0][3] = -G*mass*y[i]/(np.sqrt((x[i])**2+(y[i])**2))**3
        
        k[1][0] = vx[i]+dt*k[0][2]/2
        k[1][1] = vy[i]+dt*k[0][3]/2
        k[1][2] = -G*mass*(x[i]+dt*k[0][0]/2)/(np.sqrt((x[i]+dt*k[0][0]/2)**2+(y[i]+dt*k[0][1])**2))**3
        k[1][3] = -G*mass*(y[i]+dt*k[0][1]/2)/(np.sqrt((x[i]+dt*k[0][0]/2)**2+(y[i]+dt*k[0][1])**2))**3
        
        k[2][0] = vx[i]+dt*k[1][2]/2
        k[2][1] = vy[i]+dt*k[1][3]/2
        k[2][2] = -G*mass*(x[i]+dt*k[1][0]/2)/(np.sqrt((x[i]+dt*k[1][0]/2)**2+(y[i]+dt*k[1][1]/2)**2))**3
        k[2][3] = -G*mass*(y[i]+dt*k[1][1]/2)/(np.sqrt((x[i]+dt*k[1][0]/2)**2+(y[i]+dt*k[1][1]/2)**2))**3

        k[3][0] = vx[i]+dt*k[2][2]
        k[3][1] = vy[i]+dt*k[2][3]
        k[3][2] = -G*mass*(x[i]+dt*k[2][0])/(np.sqrt((x[i]+dt*k[2][0])**2+(y[i]+dt*k[2][1])**2))**3
        k[3][3] = -G*mass*(y[i]+dt*k[2][1])/(np.sqrt((x[i]+dt*k[2][0])**2+(y[i]+dt*k[2][1])**2))**3

        x[i+1]  = x[i]  + (k[0][0]+2*k[1][0]+2*k[2][0]+k[3][0])*dt/6
        y[i+1]  = y[i]  + (k[0][1]+2*k[1][1]+2*k[2][1]+k[3][1])*dt/6
        vx[i+1] = vx[i] + (k[0][2]+2*k[1][2]+2*k[2][2]+k[3][2])*dt/6
        vy[i+1] = vy[i] + (k[0][3]+2*k[1][3]+2*k[2][3]+k[3][3])*dt/6
        
        if neg_flag:
            if x[i+1] > 0:
                rev_count += 1
                neg_flag = False         
                if rev_count == obroty:
                    iteration = i
                    break      
            continue
        if x[i+1] < 0:
            neg_flag = True
    
    return i
    
def zadanie_2(iter_num,r0,v0,param,obroty):
    stop0 = time.time()

    t  = np.linspace(0,iter_num,iter_num+1)
    x  = np.array(t)
    y  = np.array(t)
    vx = np.array(t)
    vy = np.array(t)

    x[0] = 0
    y[0] = r0
    vx[0]= v0
    vy[0]= 0   
    i = RK4(x,y,vx,vy,iter_num,obroty)

    t_d  = np.delete(t,slice(i,iter_num+1))
    x_d  = np.delete(x,slice(i,iter_num+1))
    y_d  = np.delete(y,slice(i,iter_num+1))

    param = str("Zadanie_2") + param 
    stop1 = time.time()
    print("Zadanie2 Calculation time: ",stop1-stop0,"s.")
    #rysuj
    narysuj(x_d,y_d,"y(x)_"+param,"x[m] y[m]")
    narysuj(t_d,x_d,"x(t)_"+param,"x[m] y[m]")
    
    stop2 = time.time()
    print("Zadanie2 Drawing time: ",stop2-stop1,"s.")

def Euler3(x,y,vx,vy,iter_num,obroty,dt,tol):
    rev_count = 0
    neg_flag  = False
    epsilon   = 0.0
    c         = 0.9
    n         = 1
    iteration = 0

    for i in range(0,iter_num):
        while True:
            x1     = x[i]  + vx[i]*dt
            y1     = y[i]  + vy[i]*dt
            vx1    = vx[i] - G*mass*x[i]*dt/(np.sqrt((x[i])**2+(y[i])**2))**3
            vy1    = vy[i] - G*mass*y[i]*dt/(np.sqrt((x[i])**2+(y[i])**2))**3
            xpos   = x[i]  + vx[i]*dt/2
            ypos   = y[i]  + vy[i]*dt/2
            vxpos  = vx[i] - G*mass*x[i]*(dt/2)/(np.sqrt((x[i])**2+(y[i])**2))**3
            vypos  = vy[i] - G*mass*y[i]*(dt/2)/(np.sqrt((x[i])**2+(y[i])**2))**3
            xprim  = xpos + vxpos*dt/2
            yprim  = ypos + vypos*dt/2
            vxprim = vxpos- G*mass*xpos*(dt/2)/(np.sqrt((x[i])**2+(y[i])**2))**3
            vyprim = vypos- G*mass*ypos*(dt/2)/(np.sqrt((x[i])**2+(y[i])**2))**3
            epsilon= np.max([(xprim - x1)/((2**n)-1),(yprim - y1)/((2**n)-1)])
            if epsilon <= tol:
                x[i+1]  = xprim
                y[i+1]  = yprim
                vx[i+1] = vxprim
                vy[i+1] = vyprim  
                break
            dt = c*dt*(tol/epsilon)**(1/(n+1))
        
        if neg_flag:
            if x[i+1] > 0:
                rev_count += 1
                neg_flag = False         
                if rev_count == obroty:
                    iteration = i
                    break      
            continue
        if x[i+1] < 0:
            neg_flag = True
    return iteration

def zadanie_3(iter_num,r0,v0,param,obroty,dt,tol):
    stop0 = time.time()

    t  = np.array(np.linspace(0,iter_num,iter_num+1))    
    x  = np.array(t)
    y  = np.array(t)
    vx = np.array(t)
    vy = np.array(t)

    x[0] = 0
    y[0] = r0
    vx[0]= v0
    vy[0]= 0   
    i  = Euler3(x,y,vx,vy,iter_num,obroty,dt,tol)
    
    t_d  = np.delete(t,slice(i,iter_num+1))
    x_d  = np.delete(x,slice(i,iter_num+1))
    y_d  = np.delete(y,slice(i,iter_num+1))
    stop1 = time.time()
    print("Zadanie3 Calculation time: ",stop1-stop0,"s.")
    #draw
    param = str("Zadanie_3") + param + "_tol_" + str(tol) 
    narysuj(x_d,y_d,"y(x)_"+param,"x[m] y[m]")
    narysuj(t_d,x_d,"x(t)_"+param,"x[m] y[m]")
    stop2 = time.time()
    print("Zadanie3 Drawing time: ",stop2-stop1,"s.")


@njit
def RK44(x,y,vx,vy,iter_num,obroty,tol,dt4):
    k1= np.zeros((4,4))
    k2= np.zeros((4,4))
    k3= np.zeros((4,4))
    n = 4
    c = 0.9
    epsilon = 0.0
    rev_count = 0
    neg_flag = False

    #RK4
    for i in range(0,iter_num):
        while True:
            k1[0][0] = vx[i]
            k1[0][1] = vy[i]
            k1[0][2] = -G*mass*x[i]/(np.sqrt((x[i])**2+(y[i])**2))**3
            k1[0][3] = -G*mass*y[i]/(np.sqrt((x[i])**2+(y[i])**2))**3
            
            k1[1][0] = vx[i]+dt4*k1[0][2]/2
            k1[1][1] = vy[i]+dt4*k1[0][3]/2
            k1[1][2] = -G*mass*(x[i]+dt4*k1[0][0]/2)/(np.sqrt((x[i]+dt4*k1[0][0]/2)**2+(y[i]+dt4*k1[0][1])**2))**3
            k1[1][3] = -G*mass*(y[i]+dt4*k1[0][1]/2)/(np.sqrt((x[i]+dt4*k1[0][0]/2)**2+(y[i]+dt4*k1[0][1])**2))**3
            
            k1[2][0] = vx[i]+dt4*k1[1][2]/2
            k1[2][1] = vy[i]+dt4*k1[1][3]/2
            k1[2][2] = -G*mass*(x[i]+dt4*k1[1][0]/2)/(np.sqrt((x[i]+dt4*k1[1][0]/2)**2+(y[i]+dt4*k1[1][1]/2)**2))**3
            k1[2][3] = -G*mass*(y[i]+dt4*k1[1][1]/2)/(np.sqrt((x[i]+dt4*k1[1][0]/2)**2+(y[i]+dt4*k1[1][1]/2)**2))**3

            k1[3][0] = vx[i]+dt4*k1[2][2]
            k1[3][1] = vy[i]+dt4*k1[2][3]
            k1[3][2] = -G*mass*(x[i]+dt4*k1[2][0])/(np.sqrt((x[i]+dt4*k1[2][0])**2+(y[i]+dt4*k1[2][1])**2))**3
            k1[3][3] = -G*mass*(y[i]+dt4*k1[2][1])/(np.sqrt((x[i]+dt4*k1[2][0])**2+(y[i]+dt4*k1[2][1])**2))**3

            x1  = x[i]  + (k1[0][0]+2*k1[1][0]+2*k1[2][0]+k1[3][0])*dt4/6
            y1  = y[i]  + (k1[0][1]+2*k1[1][1]+2*k1[2][1]+k1[3][1])*dt4/6
            vx1 = vx[i] + (k1[0][2]+2*k1[1][2]+2*k1[2][2]+k1[3][2])*dt4/6
            vy1 = vy[i] + (k1[0][3]+2*k1[1][3]+2*k1[2][3]+k1[3][3])*dt4/6
            
            k2[0][0] = vx[i]
            k2[0][1] = vy[i]
            k2[0][2] = -G*mass*x[i]/(np.sqrt((x[i])**2+(y[i])**2))**3
            k2[0][3] = -G*mass*y[i]/(np.sqrt((x[i])**2+(y[i])**2))**3
            
            k2[1][0] = vx[i]+dt4/2*k2[0][2]/2
            k2[1][1] = vy[i]+dt4/2*k2[0][3]/2
            k2[1][2] = -G*mass*(x[i]+dt4/2*k2[0][0]/2)/(np.sqrt((x[i]+dt4/2*k2[0][0]/2)**2+(y[i]+dt4/2*k2[0][1])**2))**3
            k2[1][3] = -G*mass*(y[i]+dt4/2*k2[0][1]/2)/(np.sqrt((x[i]+dt4/2*k2[0][0]/2)**2+(y[i]+dt4/2*k2[0][1])**2))**3
            
            k2[2][0] = vx[i]+dt4/2*k2[1][2]/2
            k2[2][1] = vy[i]+dt4/2*k2[1][3]/2
            k2[2][2] = -G*mass*(x[i]+dt4/2*k2[1][0]/2)/(np.sqrt((x[i]+dt4/2*k2[1][0]/2)**2+(y[i]+dt4/2*k2[1][1]/2)**2))**3
            k2[2][3] = -G*mass*(y[i]+dt4/2*k2[1][1]/2)/(np.sqrt((x[i]+dt4/2*k2[1][0]/2)**2+(y[i]+dt4/2*k2[1][1]/2)**2))**3

            k2[3][0] = vx[i]+dt4/2*k2[2][2]
            k2[3][1] = vy[i]+dt4/2*k2[2][3]
            k2[3][2] = -G*mass*(x[i]+dt4/2*k2[2][0])/(np.sqrt((x[i]+dt4/2*k2[2][0])**2+(y[i]+dt4/2*k2[2][1])**2))**3
            k2[3][3] = -G*mass*(y[i]+dt4/2*k2[2][1])/(np.sqrt((x[i]+dt4/2*k2[2][0])**2+(y[i]+dt4/2*k2[2][1])**2))**3

            x2  = x[i]  + (k2[0][0]+2*k2[1][0]+2*k2[2][0]+k2[3][0])*dt4/2/6
            y2  = y[i]  + (k2[0][1]+2*k2[1][1]+2*k2[2][1]+k2[3][1])*dt4/2/6
            vx2 = vx[i] + (k2[0][2]+2*k2[1][2]+2*k2[2][2]+k2[3][2])*dt4/2/6
            vy2 = vy[i] + (k2[0][3]+2*k2[1][3]+2*k2[2][3]+k2[3][3])*dt4/2/6
            
            k3[0][0] = vx2
            k3[0][1] = vy2
            k3[0][2] = -G*mass*x2/(np.sqrt((x2)**2+(y2)**2))**3
            k3[0][3] = -G*mass*y2/(np.sqrt((x2)**2+(y2)**2))**3
            
            k3[1][0] = vx2+dt4/2*k3[0][2]/2
            k3[1][1] = vy2+dt4/2*k3[0][3]/2
            k3[1][2] = -G*mass*(x2+dt4/2*k3[0][0]/2)/(np.sqrt((x2+dt4/2*k3[0][0]/2)**2+(y2+dt4/2*k3[0][1])**2))**3
            k3[1][3] = -G*mass*(y2+dt4/2*k3[0][1]/2)/(np.sqrt((x2+dt4/2*k3[0][0]/2)**2+(y2+dt4/2*k3[0][1])**2))**3
            
            k3[2][0] = vx2+dt4/2*k3[1][2]/2
            k3[2][1] = vy2+dt4/2*k3[1][3]/2
            k3[2][2] = -G*mass*(x2+dt4/2*k3[1][0]/2)/(np.sqrt((x2+dt4/2*k3[1][0]/2)**2+(y2+dt4/2*k3[1][1]/2)**2))**3
            k3[2][3] = -G*mass*(y2+dt4/2*k3[1][1]/2)/(np.sqrt((x2+dt4/2*k3[1][0]/2)**2+(y2+dt4/2*k3[1][1]/2)**2))**3

            k3[3][0] = vx2+dt4/2*k3[2][2]
            k3[3][1] = vy2+dt4/2*k3[2][3]
            k3[3][2] = -G*mass*(x2+dt4/2*k3[2][0])/(np.sqrt((x2+dt4/2*k3[2][0])**2+(y2+dt4/2*k3[2][1])**2))**3
            k3[3][3] = -G*mass*(y2+dt4/2*k3[2][1])/(np.sqrt((x2+dt4/2*k3[2][0])**2+(y2+dt4/2*k3[2][1])**2))**3

            x3  = x2  + (k3[0][0]+2*k3[1][0]+2*k3[2][0]+k3[3][0])*dt4/2/6
            y3  = y2  + (k3[0][1]+2*k3[1][1]+2*k3[2][1]+k3[3][1])*dt4/2/6
            vx3 = vx2 + (k3[0][2]+2*k3[1][2]+2*k3[2][2]+k3[3][2])*dt4/2/6
            vy3 = vy2 + (k3[0][3]+2*k3[1][3]+2*k3[2][3]+k3[3][3])*dt4/2/6
            
            epsilon= np.max(np.array([(x3 - x1)/((2**n)-1),(y3 - y1)/((2**n)-1)]))
            if epsilon <= tol:
                x[i+1]  = x3
                y[i+1]  = y3
                vx[i+1] = vx3
                vy[i+1] = vy3 
                break
            dt4 = c*dt4*(tol/epsilon)**(1/(n+1))
        if neg_flag:
            if x[i+1] > 0:
                rev_count += 1
                neg_flag = False         
                if rev_count == obroty:
                    iteration = i
                    break      
            continue
        if x[i+1] < 0:
            neg_flag = True
    
    return i
    
def zadanie_4(iter_num,r0,v0,param,obroty,tol,dt4):
    stop0 = time.time()

    t  = np.linspace(0,iter_num,iter_num+1)
    x  = np.array(t)
    y  = np.array(t)
    vx = np.array(t)
    vy = np.array(t)

    x[0] = 0
    y[0] = r0
    vx[0]= v0
    vy[0]= 0   
    i = RK44(x,y,vx,vy,iter_num,obroty,tol,dt)

    t_d  = np.delete(t,slice(i,iter_num+1))
    x_d  = np.delete(x,slice(i,iter_num+1))
    y_d  = np.delete(y,slice(i,iter_num+1))

    param = str("Zadanie_4") + param + "_tol_" + str(tol)
    stop1 = time.time()
    print("Zadanie4 Calculation time: ",stop1-stop0,"s.")
    #rysuj
    narysuj(x_d,y_d,"y(x)_"+param,"x[m] y[m]")
    narysuj(t_d,x_d,"x(t)_"+param,"x[m] y[m]")
    
    stop2 = time.time()
    print("Zadanie4 Drawing time: ",stop2-stop1,"s.")

if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")
param = str("_dt_"+str(dt)+"_obroty_"+str(obroty))

iter_num = int(obroty*2*80*365*24*(3600/dt))
zadanie_1(iter_num,r0,v0,param,obroty)
zadanie_2(iter_num,r0,v0,param,obroty)

iter_num = int(obroty*8*80*365*24*(3600/dt))
#tol 1000
zadanie_3(iter_num,r0,v0,param,obroty,dt,tol)
#tol 100
zadanie_3(iter_num,r0,v0,param,obroty,dt,100)

zadanie_4(iter_num,r0,v0,param,obroty,tol,dt)
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()