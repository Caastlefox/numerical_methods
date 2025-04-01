#importowanie bibliotek
import time
begin = time.time()
from numba import njit
from numba import jit
import numpy.random as rd
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import matplotlib.pyplot as plt
import numpy as np
#tworzenie ścieżki output
if not os.path.exists(os.getcwd()+"/output"):
    os.makedirs(os.getcwd()+"/output")
    
#deklaracja domyślnych etrów wejściowych
iter_num = 10000000

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
def narysujl(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.plot(X,Y, linewidth=2,linestyle='solid')
    #plt.show()
    plt.savefig(os.getcwd()+"/output/"+name+".png",dpi=100)
    print("figure ",name," saved"),
    sys.stdout.flush()
    
@njit
def rho(x):
    return(np.exp(-x**2)/np.sqrt(np.pi))
@njit
def phi(x,y):
    return(np.exp(-0.5*(x**2+y**2)/np.sqrt(np.pi)))


@njit
def zadanie(xw,In,control,n):
    xwtmp = np.zeros(25001)
    Intmp = np.zeros(25001)
    iteration = 0
    while True:
        if control < 25000:
            for i in range(1,control):
                iteration +=1
                dx = (rd.rand()-0.5)/2
                xp = xwtmp[i-1] + dx
                y  = rd.rand()
                if y < rho(xp)/rho(xwtmp[i-1]):
                    xwtmp[i] =xp
                else:
                    xwtmp[i]=xwtmp[i-1]
                Intmp[i] = Intmp[i-1]*(i-1) + (xwtmp[i]**n)
                Intmp[i] = Intmp[i]/i
            xw[iteration-control+1:iteration+1] = xwtmp[0:control]
            In[iteration-control+1:iteration+1] = Intmp[0:control]
            print("endflag passed")
            break

        for i in range(1,25001):
            iteration +=1
            dx = (rd.rand()-0.5)/2
            xp = xwtmp[i-1] + dx
            y  = rd.rand()
            if y < rho(xp)/rho(xwtmp[i-1]):
                xwtmp[i] =xp
            else:
                xwtmp[i]=xwtmp[i-1]
            Intmp[i] = Intmp[i-1]*(iteration-1) + (xwtmp[i]**n)
            Intmp[i] = Intmp[i]/iteration
            
        xw[iteration-25000:iteration] = xwtmp[:25000]
        In[iteration-25000:iteration] = Intmp[:25000]       
        control  -= 25000
        xwtmp[0] = xwtmp[25000]
        Intmp[0] = Intmp[25000]
        print("Iterations done: " + str(iteration))

@njit       
def zadanie2DE(xw,yw,Epot,control):
    xwtmp = np.zeros(25001)
    ywtmp = np.zeros(25001)
    Intmp = np.zeros(25001)
    iteration = 0
    while True:
        if control < 25000:
            for i in range(1,control):
                iteration +=1
                dx = (rd.rand()-0.5)/2
                dy = (rd.rand()-0.5)/2
                xp = xwtmp[i-1] + dx
                yp = ywtmp[i-1] + dy
                
                y  = rd.rand()
                if y < (phi(xp,ywtmp[i-1]))/(phi(xwtmp[i-1],ywtmp[i-1])):
                    xwtmp[i] = xp
                else:
                    xwtmp[i] = xwtmp[i-1]
                if y < (phi(xwtmp[i-1],yp))/(phi(xwtmp[i-1],ywtmp[i-1])):
                    ywtmp[i] = yp    
                else:
                    ywtmp[i] = ywtmp[i-1]
                Intmp[i] = Intmp[i-1]*(iteration-1) + (0.5*xwtmp[i]**2 + 0.5*ywtmp[i]**2)
                Intmp[i] = Intmp[i]/iteration
                
            xw[iteration-control+1:iteration+1] = xwtmp[0:control]
            yw[iteration-control+1:iteration+1] = ywtmp[0:control]
            Epot[iteration-control+1:iteration+1] = Intmp[0:control]
            print("endflag passed")
            break

        for i in range(1,25001):
            iteration +=1
            dx = (rd.rand()-0.5)/2
            dy = (rd.rand()-0.5)/2
            xp = xwtmp[i-1] + dx
            yp = ywtmp[i-1] + dy
            y  = rd.rand()
            if y < (phi(xp,ywtmp[i-1]))/(phi(xwtmp[i-1],ywtmp[i-1])):
                xwtmp[i] = xp
            else:
                xwtmp[i] = xwtmp[i-1]
            if y < (phi(xwtmp[i-1],yp))/(phi(xwtmp[i-1],ywtmp[i-1])):
                ywtmp[i] = yp    
            else:
                ywtmp[i] = ywtmp[i-1]
                
            Intmp[i] = Intmp[i-1]*(iteration-1) + (xwtmp[i]**2+ywtmp[i]**2)*0.5
            Intmp[i] = Intmp[i]/iteration
            
        xw[iteration-25000:iteration]   = xwtmp[:25000]
        yw[iteration-25000:iteration]   = ywtmp[:25000]
        Epot[iteration-25000:iteration] = Intmp[:25000]       
        control  -= 25000
        xwtmp[0] = xwtmp[25000]
        ywtmp[0] = ywtmp[25000]
        Intmp[0] = Intmp[25000]
        print("Iterations done: " + str(iteration))

xw = np.zeros(iter_num)
In = np.zeros(iter_num)

control = iter_num

n=1
zadanie(xw,In,control,n)
sys.stdout.flush()
larr = np.linspace(0,iter_num-1,iter_num)
narysuj(larr,In,"In(l)_r1","liczba_iteracji moment_rzedu_1")
sys.stdout.flush()

n=2
zadanie(xw,In,control,n)
sys.stdout.flush()
larr = np.linspace(0,iter_num-1,iter_num)
narysuj(larr,In,"In(l)_r2","liczba_iteracji moment_rzedu_2")
sys.stdout.flush()

n=3
zadanie(xw,In,control,n)
sys.stdout.flush()
larr = np.linspace(0,iter_num-1,iter_num)
narysuj(larr,In,"In(l)_r3","liczba_iteracji moment_rzedu_3")
sys.stdout.flush()

n=4
zadanie(xw,In,control,n)
sys.stdout.flush()
larr = np.linspace(0,iter_num-1,iter_num)
narysuj(larr,In,"In(l)_r4","liczba_iteracji moment_rzedu_4")
sys.stdout.flush()

xw = np.zeros(100)
yw = np.zeros(100)
Epot = np.zeros(100)
control = iter_num
zadanie2DE(xw,yw,Epot,100)
narysujl(xw,yw,"x(y)","x y")
sys.stdout.flush()

xw   = np.zeros(iter_num)
yw   = np.zeros(iter_num)
Epot = np.zeros(iter_num)
zadanie2DE(xw,yw,Epot,control)
larr = np.linspace(0,iter_num-1,iter_num)
narysuj(larr,Epot,"Epot(l)","liczba_iteracji Energia_potencjalna")

#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()
