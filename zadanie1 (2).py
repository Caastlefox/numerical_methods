#importowanie bibliotek
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
#rysowanie
def narysuj(X,Y,name,stop):
    plt.clf()
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
    print("figure ",name," saved"),
    sys.stdout.flush()
#potencjał
@nb.njit
def pot(x):
    return -1*np.exp(-x**2)-1.2*np.exp(-(x-2)**2)
#pierwsza pochodna
@nb.njit
def dpodx(x):
    dx = 0.001
    return (pot(x+dx)-pot(x-dx))/(2*dx)
#druga pochodna
@nb.njit
def dpodx2(x):
    dx = 0.001
    return (pot(x+dx)+pot(x-dx)-2*pot(x))/(dx**2)
#funkcja F1 z instrukcji
@nb.njit
def F1(xn1,vn1,xn,vn,dt):
    return xn1-xn-dt*(vn1+vn)/2
@nb.njit
#funkcja F2 z instrukcji    
def F2(xn1,vn1,xn,vn,alpha,dt):
    m  = 1
    return vn1-vn+dt*(((dpodx(xn1)+dpodx(xn))/m)+(vn1+vn)*alpha)/2
def zad_1_AND_2(start,stop,dx,dt,mass,alpha,param):
    #deklaracje
    iter_num = int((stop - start)/dt+1)
    t = np.linspace(start,stop,iter_num)
    x = np.array(t)
    v = np.array(t)
    #warunki początkowe
    x[0] = 2.8
    v[0] = 0
    #Jawny schemat eulera
    for i in range(0,iter_num-1):
        x[i+1] = x[i] + v[i]*dt
        v[i+1] = v[i] - dt*dpodx(x[i])/(mass) - v[i]*alpha*dt
    #Kinetyczna
    Ekin = (1/2)*mass*v**2
    #potencjalna
    Epot = pot(x)    
    #całkowita
    Ec = Ekin + Epot
    #rysuj
    narysuj(t,x,"zad_1_AND_2_x"+param,stop)
    narysuj(t,v,"zad_1_AND_2_v"+param,stop)
    narysuj(t,Ekin,"zad_1_AND_2_Ekin"+param,stop)
    narysuj(t,Epot,"zad_1_AND_2_Epot"+param,stop)
    narysuj(t,Ec,"zad_1_AND_2_Ec"+param,stop)
    narysuj(x,v,"zad_1_AND_2_xv"+param,stop)
    end = time.time()
    return end
#zadanie3
def zad_3(dt,m,alpha):
    #warunki początkowe i deklaracje
    x0 = 2.8
    v0 = 0
    xit = []
    vit = []
    xit.append(x0)
    vit.append(v0)
    i = 0
    flag = True
    vfound = 0
    xfound = 0
    while True:
        #obliczenie wyznaczników W Wx i Wv
        a11 = 1
        a12 = -dt/2
        a21 = dt*dpodx2(xit[i])/(2*m)
        a22 = 1+alpha*dt/2
        W   = a11*a22-a12*a21
        Wx  = -F1(xit[i],vit[i],xit[0],vit[0],dt)*a22-F2(xit[i],vit[i],xit[0],vit[0],alpha,dt)*a21
        Wv  = -F2(xit[i],vit[i],xit[0],vit[0],alpha,dt)*a11+F1(xit[i],vit[i],xit[0],vit[0],dt)*a12
        #rozwiązanie układu równań podanego w instrucji
        dx  = Wx/W
        dv  = Wv/W
        #warunki wyjścia z pętli - gdy x[n]=x[n+1] iteracja dla x jest zatrzymywana, analogicznie dla v, spełnienie obu warunków kończy pętlę
        if math.isclose(xit[i],xit[i]+dx):
            xit.append(xit[i])
            if flag == True:
                xfound = i
            flag = False
        else:
            xit.append(xit[i]+dx)
        if  math.isclose(vit[i],vit[i]+dv):
            vit.append(vit[i])
            if flag == True:
                vfound = i
            flag = False
        else:
            vit.append(vit[i]+dv)
        if math.isclose(xit[i],xit[i]+dx) and math.isclose(vit[i],vit[i]+dv):
            break        
        i += 1

    print("zbieznosc xn=1 dla: ",xit[-1],"w kroku: " ,xfound)
    print("zbieznosc vn=1 dla: ",vit[-1],"w kroku: " ,vfound)
    return time.time()
@nb.njit
#funkcja podobna funkcji do zad_3, ale przystosowana pod zad_4
def zbieznosc(x0,v0,dt,m,alpha):
    #deklaracje i warunki początkowe
    xit = [x0]
    vit = [v0]
    i = 0
    flag = True
    while True:
        #obliczenie wyznaczników W Wx i Wv
        a11 = 1
        a12 = -dt/2
        a21 = dt*dpodx2(xit[i])/(2*m)
        a22 = 1+alpha*dt/2
        W   = a11*a22-a12*a21
        Wx  = -F1(xit[i],vit[i],x0,v0,dt)*a22-F2(xit[i],vit[i],x0,v0,alpha,dt)*a21
        Wv  = -F2(xit[i],vit[i],x0,v0,alpha,dt)*a11+F1(xit[i],vit[i],x0,v0,dt)*a12
        #rozwiązanie układu równań
        dx  = Wx/W
        dv  = Wv/W
        #warunki wyjścia z pętli - gdy x[n]=x[n+1] iteracja dla x jest zatrzymywana, analogicznie dla v, spełnienie obu warunków kończy pętlę
        if np.isclose(xit[i],xit[i]+dx,atol=1e-09):
            xit.append(xit[i])
        else:
            xit.append(xit[i]+dx)
        if np.isclose(vit[i],vit[i]+dv,atol=1e-09):
            vit.append(vit[i])
        else:
            vit.append(vit[i]+dv)
        if np.isclose(xit[i],xit[i]+dx,atol=1e-09) and np.isclose(vit[i],vit[i]+dv,atol=1e-09):
            break        
        i += 1
    return np.array([xit[i],vit[i]])
#zadanie4
def zad_4(start,stop,dt,m,alpha,param):
    #deklaracje wstępne
    iter_num = int((stop - start)/dt+1)
    t = np.linspace(start,stop,iter_num)
    x = np.array(t)
    v = np.array(t)
    #warunki początkowe
    x[0]= 2.8
    v[0]= 0
    #rozwiązanie przy pomocy schematu trapezów
    for I in range (0,iter_num-1):  
        sol = zbieznosc(x[I],v[I],dt,m,alpha)
        x[I+1] = sol[0]
        v[I+1] = sol[1]
    #Kinetyczna
    Ekin = (1/2)*m*v**2
    #potencjalna
    Epot = pot(x)
    #całkowita
    Ec = Ekin + Epot
    #rysuj
    narysuj(t,x,"zad_4_x"+param,stop)
    narysuj(t,v,"zad_4_v"+param,stop)
    narysuj(t,Ekin,"zad_4_Ekin"+param,stop)
    narysuj(t,Epot,"zad_4_Epot"+param,stop)
    narysuj(t,Ec,"zad_4_Ec"+param,stop)
    narysuj(x,v,"zad_4_xv"+param,stop)
    return time.time()
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"\\output"):
    os.makedirs(os.getcwd()+"\\output")
#deklaracja zmiennych
start = 0
stop_lst = [30,100,1000]
dx = 0.001
mass = 1
end = 0 
dt_lst = [0.01,0.001]
alpha_lst = [0,0.5,5,201]
#zadanie 1+2
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
#zadanie 3
alpha = 0
dt = 0.01
checkpoint_begin = time.time()
checkpoint_end = zad_3(dt,mass,alpha)
print("Calculation time of block zad_3: ",checkpoint_end-checkpoint_begin,"s.")
#zadanie 4
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
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")