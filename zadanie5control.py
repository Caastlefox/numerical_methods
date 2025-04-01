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
#deklaracja domyślnych etrów wejściowych
iter_num = 10000
n = 2
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
def rho(x):
    return(np.exp(-x**2)/np.sqrt(np.pi))
    
@njit
def zadanie(xw,In):
    for i in range(1,iter_num):
        
        dx = (rd.rand()-0.5)/2
        xp = xw[i-1] + dx
        
        y  = rd.rand()
        if y < rho(xp)/rho(xw[i-1]):
            xw[i] =xp
        else:
            xw[i]=xw[i-1]

        In[i] = In[i-1]*(i-1) + (xw[i]**n)
        In[i] = In[i]/i
        
xw = np.zeros(iter_num)
In = np.zeros(iter_num)
xw[0] = 0.5
In[0] = 0.5
In[1] = 0.5
xw[1] = 0.5
zadanie(xw,In)

    
larr = np.linspace(0,iter_num-1,iter_num)
narysuj(larr,In,"namez","e e")

#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()

@njit       
def zadanie2DE(xw,yw,Epot):
    for i in range(1,iter_num):
        dx = (rd.rand()-0.5)/2
        dy = (rd.rand()-0.5)/2
        xp = xw[i-1] + dx
        yp = yw[i-1] + dy
        yrand  = rd.rand()
        if yrand < phi(xp,yp)/phi(xw[i],yw[i]):
            xw[i] = xp
            yw[i] = yp
        else:
            xw[i] = xw[i-1]
            yw[i] = yw[i-1]
        for l in range(0,i):
            Epot[i] += (xw[l]**2+yw[l]**2)*0.5*(phi(xw[l],yw[l]))**2
        Epot[i] = Epot[i]/i 