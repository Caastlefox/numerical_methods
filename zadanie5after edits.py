#importowanie bibliotek
import time
begin = time.time()
from numba import njit
from numba import jit
import numpy.random as rd
import sys
#sys.stdout = open("log_file.log","w") #uncomment to create log(no console printout)
import os
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
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
    plt.show()
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
def zadanie(xw,In,n):
    for i in range(1,iter_num):
        
        dx = (rd.rand()-0.5)/2
        xp = xw[i-1] + dx
        y  = rd.rand()
        if y < rho(xp)/rho(xw[i]):
            xw[i] = xp
        else:
            xw[i] = xw[i-1]
        for l in range(0,i):
            In[i] = In[i] + xw[l]**n
        In[i] = In[i]/i
        if i%10000 == 0:
            print(i)
        
    print("calc finished")

@njit       
def zadanie2D(xw,yw):
    for i in range(1,iter_num):
        dx = (rd.rand()-0.5)/2
        dy = dx#(rd.rand()-0.5)/2
        xp = xw[i-1] + dx
        yp = yw[i-1] + dy
        yrand  = rd.rand()
        if yrand < phi(xp,yp)**2/phi(xw[i],yw[i]**2):
            xw[i] = xp
            yw[i] = yp
        else:
            xw[i] = xw[i-1]
            yw[i] = yw[i-1]

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
"""
larr = np.linspace(0,iter_num-1,iter_num)
for n in [1,2,3,4]:
    xw = np.zeros(iter_num)
    In = np.zeros(iter_num)
    zadanie(xw,In,n)
    narysuj(larr,In,'In(l)'+n,u'l In')
"""
iter_num = 500
xw = np.zeros(iter_num)
yw = np.zeros(iter_num)
iter_num_arr = np.linspace(0,iter_num-1,iter_num)
zadanie2D(xw,yw)
narysujl(xw,yw,'yw(xw)','xw yw')
iter_num = 100000
iter_num_arr = np.linspace(0,iter_num-1,iter_num)
xw = np.zeros(iter_num)
yw = np.zeros(iter_num)
Epot = np.zeros(iter_num)
zadanie2DE(xw,yw,Epot)

narysuj(iter_num_arr,Epot,'Epot(i)','i Epot')
#czas wykonania
end = time.time()
print("Overall_Calculation time: ",end-begin,"s.")
sys.stdout.close()
