
start = 0 #int
stop_lst = [1000] #list
mass = 1 #int
dt_lst = [0.01,0.001] #list 
alpha_lst = [0] #list


import time
begin = time.time()
import numba as nb
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import math 
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('error')
csfont = {'fontname':'Comic Sans MS'}
#rysowanie
def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
    print("figure ",name," saved"),
    sys.stdout.flush()

@nb.njit
def pot(x):
    return -1*np.exp(-x**2)-1.2*np.exp(-(x-2)**2)

@nb.njit
def dpodx(x):
    dx = 0.001
    return (pot(x+dx)-pot(x-dx))/(2*dx)

@nb.njit
def dpodx2(x):
    dx = 0.001
    return (pot(x+dx)+pot(x-dx)-2*pot(x))/(dx**2)

@nb.njit
def F1(xn1,vn1,xn,vn,dt):
    return xn1-xn-dt*(vn1+vn)/2
@nb.njit
  
def F2(xn1,vn1,xn,vn,alpha,dt):
    m  = 1
    return vn1-vn+dt*(((dpodx(xn1)+dpodx(xn))/m)+(vn1+vn)*alpha)/2
def zad_1_AND_2(start,stop,dx,dt,mass,alpha,param):

    iter_num = int((stop - start)/dt+1)
    t = np.linspace(start,stop,iter_num)
    x = np.array(t)
    v = np.array(t)
    #initial
    x[0] = 2.8
    v[0] = 0
    #Explicit Euler
    for i in range(0,iter_num-1):
        x[i+1] = x[i] + v[i]*dt
        v[i+1] = v[i] - dt*dpodx(x[i])/(mass) - v[i]*alpha*dt
    #Kinetic Energy
    Ekin = (1/2)*mass*v**2
    #Potential Energy
    Epot = pot(x)    
    #Total Energy
    Ec = Ekin + Epot
    #draw
    narysuj(t,x,"zad_1_AND_2_x_"+param,"czas[s] pozycja[m]")
    narysuj(t,v,"zad_1_AND_2_v_"+param,"czas[s] prędkość[m/s]")
    narysuj(t,Ekin,"zad_1_AND_2_Ek"+param,"czas[s] Energia_kinetyczna[J]")
    narysuj(t,Epot,"zad_1_AND_2_Ep"+param,"czas[s] Energia_potencjalna[J]")
    narysuj(t,Ec,"zad_1_AND_2_Ec"+param,"czas[s] Energia_całkowita[J]")
    narysuj(x,v,"zad_1_AND_2_xv"+param,"pozycja[m] prędkość[m/s]")
    end = time.time()
    return end
#zadanie3
def zad_3(dt,m,alpha):

    x0 = 2.8
    v0 = 0
    xit = []
    vit = []
    xit.append(x0)
    vit.append(v0)
    i = 0
    
    vfound = 1
    xfound = 1
    while True:

        a11 = 1
        a12 = -dt/2
        a21 = dt*dpodx2(xit[i])/(2*m)
        a22 = 1+alpha*dt/2
        W   = a11*a22-a12*a21
        Wx  = -F1(xit[i],vit[i],xit[0],vit[0],dt)*a22-F2(xit[i],vit[i],xit[0],vit[0],alpha,dt)*a21
        Wv  = -F2(xit[i],vit[i],xit[0],vit[0],alpha,dt)*a11+F1(xit[i],vit[i],xit[0],vit[0],dt)*a12

        dx  = Wx/W
        dv  = Wv/W
   
        if math.isclose(xit[i],xit[i]+dx):
            xit.append(xit[i])
     
            xfound = i

        else:
            xit.append(xit[i]+dx)
            xfound += 1
        if  math.isclose(vit[i],vit[i]+dv):
        
            vit.append(vit[i])
            
            
            
        else:
            vit.append(vit[i]+dv)
            vfound += 1
        if math.isclose(xit[i],xit[i]+dx) and math.isclose(vit[i],vit[i]+dv):
            break        
        i += 1

    print("zbieznosc xn=1 dla: ",xit[-1],"w kroku: " ,xfound)
    print("zbieznosc vn=1 dla: ",vit[-1],"w kroku: " ,vfound)
    return time.time()
@nb.njit

def zbieznosc(x0,v0,dt,m,alpha):

    xit = [x0]
    vit = [v0]
    i = 0
    flag = True
    while True:
v
        a11 = 1
        a12 = -dt/2
        a21 = dt*dpodx2(xit[i])/(2*m)
        a22 = 1+alpha*dt/2
        W   = a11*a22-a12*a21
        Wx  = -F1(xit[i],vit[i],x0,v0,dt)*a22-F2(xit[i],vit[i],x0,v0,alpha,dt)*a21
        Wv  = -F2(xit[i],vit[i],x0,v0,alpha,dt)*a11+F1(xit[i],vit[i],x0,v0,dt)*a12

        dx  = Wx/W
        dv  = Wv/W
 
        if np.isclose(xit[i],xit[i]+dx,atol=1e-12):
            xit.append(xit[i])
        else:
            xit.append(xit[i]+dx)
        if np.isclose(vit[i],vit[i]+dv,atol=1e-12):
            vit.append(vit[i])
        else:
            vit.append(vit[i]+dv)
        if np.isclose(xit[i],xit[i]+dx,atol=1e-12) and np.isclose(vit[i],vit[i]+dv,atol=1e-12):
            break        
        i += 1
    return [xit[i],vit[i]]

def zad_4(start,stop,dt,m,alpha,param):

    iter_num = int((stop - start)/dt+1)
    t = np.linspace(start,stop,iter_num)
    x = np.array(t)
    v = np.array(t)

    x[0]= 2.8
    v[0]= 0

    for I in range (0,iter_num-1):  
        sol = zbieznosc(x[I],v[I],dt,m,alpha)
        x[I+1] = sol[0]
        v[I+1] = sol[1]

    Ekin = (1/2)*m*v**2

    Epot = pot(x)

    Ec = Ekin + Epot

    narysuj(t,x,"zad_4_x_"+param,"czas[s] pozycja[m]")
    narysuj(t,v,"zad_4_v_"+param,"czas[s] prędkość[m/s]")
    narysuj(t,Ekin,"zad_4_Ek"+param,"czas[s] Energia_kinetyczna[J]")
    narysuj(t,Epot,"zad_4_Ep"+param,"czas[s] Energia_potencjalna[J]")
    narysuj(t,Ec,"zad_4_Ec"+param,"czas[s] Energia_całkowita[J]")
    narysuj(x,v,"zad_4_xv"+param,"pozycja[m] prędkość[m/s]")
    return time.time()
t
if not os.path.exists(os.getcwd()+"\\output"):
    os.makedirs(os.getcwd()+"\\output")


end = 0 
dx = 0.001


for alpha in alpha_lst:
    for stop in stop_lst:
        for dt in dt_lst:
            param = str("_dt_"+str(dt)+"_t_(0_"+str(stop)+")_alpha_"+str(alpha))
            try:
                checkpoint_begin = time.time()
                checkpoint_end = zad_1_AND_2(start,stop,dx,dt,mass,alpha,param)
                print("Calculation time of block zad_1_AND_2",param,": ",checkpoint_end-checkpoint_begin,"s.")
            except Exception as e:
                warnings.filterwarnings("default")
                print ("warning ",e," raised at: ", param)
                checkpoint_begin = time.time()
                checkpoint_end = zad_1_AND_2(start,stop,dx,dt,mass,alpha,param)
                print("Calculation time of block zad_1_AND_2",param,": ",checkpoint_end-checkpoint_begin,"s.")
                warnings.filterwarnings("error")
                
                

alpha = 0
deltat = 0.01
checkpoint_begin = time.time()
checkpoint_end = zad_3(deltat,mass,alpha)
print("Calculation time of block zad_3: ",checkpoint_end-checkpoint_begin,"s.")

for alpha in alpha_lst:
    for stop in stop_lst:
        for dt in dt_lst:
            param = str("_dt_"+str(dt)+"_t_(0_"+str(stop)+")_alpha_"+str(alpha))
            try:
                checkpoint_begin = time.time()
                checkpoint_end = zad_4(start,stop,dt,mass,alpha,param)
                print("Calculation time of block zad_4",param,": ",checkpoint_end-checkpoint_begin,"s.")
            except Exception as e:
                warnings.filterwarnings("default")
                print ("warning ",e," raised at: ", param)
                checkpoint_begin = time.time()
                checkpoint_end = zad_4(start,stop,dt,mass,alpha,param)
                print("Calculation time of block zad_4",param,": ",checkpoint_end-checkpoint_begin,"s.")
                warnings.filterwarnings("error")

end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()