import time
begin = time.time()
N  = 201
M  = 201
L  = 3.0
K = 41
dx = 2*L/K
dy = 2*L/K
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import csv

def narysuj(X,Y,name,labels):
    plt.clf()
    labels = labels.split(" ")
    plt.xlabel(labels[0],**csfont)
    plt.ylabel(labels[1],**csfont)
    plt.title(name,**csfont)
    plt.scatter(X,Y,marker=".",linewidths=0.1)
    #plt.show()
    plt.savefig(os.getcwd()+"\\output\\"+name+".png",dpi=100)
    #print("figure ",name," saved"),
    sys.stdout.flush()


@njit
def wspolczynnik (i):
    if (i == 0 or i == N):
        return 1
    elif ( i %2==1 ):
        return 4
    elif ( i %2==0 ):
        return 2

@njit
def iloczyn_wektorowy(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

@njit
def odleglosc(r):
    return np.sqrt(r[0]**2+r[1]**2+r[2]**2)
   
@njit
def zadanie(varr,earr,barr,alpha):
    #inicjalizacja parametrów :

    omega  = 0.1
    R  = 1
    sigma  = 1
    dtheta = np.pi/N
    dphi = 2*np.pi/M

    omegavec  = np.array([omega*np.sin(alpha),omega*np.cos(alpha),0])
    Rprim = np.zeros(3)
    r = np.zeros(3)
    z  = 0
    v  = 0
    ex = 0
    ey = 0
    bx = 0
    by = 0
    r  = np.zeros(3)
    garr = np.zeros(3)
    for xindex in range(int(2*L/dx)+1):
        for yindex in range(int(2*L/dy)+1):
            x = xindex*dx - 3
            y = yindex*dy - 3
            z  = 0
            #//==== zerowanie zmiennych przed całkowaniem ============================
            v  = 0
            ex = 0
            ey = 0
            bx = 0
            by = 0
            r[0] = x
            r[1] = y
            r[2] = 0
            #//========= całkowanie = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
            for i in range(0,N):
                for j in range(0,M):
                    thetaiprim= dtheta*i
                    phijprim  = dphi*j
                    Rprim[0]  = R*np.cos(thetaiprim)*np.cos(phijprim)
                    Rprim[1]  = R*np.cos(thetaiprim)*np.sin(phijprim)
                    Rprim[2]  = R*np.sin(thetaiprim)
                    rminRprim = odleglosc(r-Rprim)
                    W    = (sigma*(R**2)*dtheta*dphi)/(4*np.pi*3*3)
                    v    = v  + W*(wspolczynnik(i)**2)*np.sin(thetaiprim)/rminRprim
                    ex   = ex + W*(wspolczynnik(i)**2)*np.sin(thetaiprim)*(r[0]-Rprim[0])/(rminRprim**3)
                    ey   = ey + W*(wspolczynnik(i)**2)*np.sin(thetaiprim)*(r[1]-Rprim[1])/(rminRprim**3)
                    garr = iloczyn_wektorowy(r-Rprim,iloczyn_wektorowy(omegavec,Rprim))
                    bx   = bx - W*(wspolczynnik(i)**2)*np.sin(thetaiprim)*garr[0]/rminRprim**3
                    by   = by - W*(wspolczynnik(i)**2)*np.sin(thetaiprim)*garr[1]/rminRprim**3
            varr[xindex][yindex]    = v
            earr[0][xindex][yindex] = ex
            earr[1][xindex][yindex] = ey
            barr[0][xindex][yindex] = bx
            barr[1][xindex][yindex] = by

alpha  = 0
varr = np.zeros((int(2*L/dx)+1,int(2*L/dy)+1))
earr = np.zeros((2,int(2*L/dx)+1,int(2*L/dy)+1))
barr = np.zeros((2,int(2*L/dx)+1,int(2*L/dy)+1))


    
zadanie(varr,earr,barr,alpha)


X = np.linspace(-int(L),int(L),int(2*L/dy)+1)
Y = np.linspace(-int(L),int(L),int(2*L/dy)+1)
plt.clf()

plt.pcolormesh(np.linspace(-int(L),int(L),int(2*L/dy)+1),np.linspace(-int(L),int(L),int(2*L/dx)+1),varr)
plt.colorbar()
fig = plt.gcf()
ax = plt.gca()
circle1 = Circle((0, 0), radius=1, color='black',fill=False)
ax.add_patch(circle1)
plt.show()

plt.clf()
plt.quiver(X,Y,np.transpose(earr[0]),np.transpose(earr[1]))
fig = plt.gcf()
ax = plt.gca()
circle1 = Circle((0, 0), radius=1, color='blue',fill=False)
ax.add_patch(circle1)
ax.set_aspect('equal')
plt.show()

plt.clf()
plt.quiver(X,Y,np.transpose(barr[0]),np.transpose(barr[1]))
fig = plt.gcf()
ax = plt.gca()
circle1 = Circle((0, 0), radius=1, color='blue',fill=False)
ax.add_patch(circle1)
ax.set_aspect('equal')
plt.show()

plt.clf()
plt.plot(Y,varr[21])
U = 1/(4*np.pi*abs(Y))
i=0
for item in Y:
    
    if item < 1 and item > -1:
        U[i] = 1/(4*np.pi)
    i += 1

plt.plot(Y,U)
plt.show()
"""
print("alpha 0.5")
sys.stdout.flush()
varr = np.zeros((int(2*L/dx)+1,int(2*L/dy)+1))
earr = np.zeros((2,int(2*L/dx)+1,int(2*L/dy)+1))
barr = np.zeros((2,int(2*L/dx)+1,int(2*L/dy)+1))
zadanie(varr,earr,barr,0.25*np.pi)

plt.clf()

plt.pcolormesh(np.linspace(-int(L),int(L),int(2*L/dy)+1),np.linspace(-int(L),int(L),int(2*L/dx)+1),varr)
plt.colorbar()
fig = plt.gcf()
ax = plt.gca()
circle1 = Circle((0, 0), radius=1, color='black',fill=False)
ax.add_patch(circle1)
plt.show()

plt.clf()
plt.quiver(X,Y,np.transpose(earr[0]),np.transpose(earr[1]))
fig = plt.gcf()
ax = plt.gca()
circle1 = Circle((0, 0), radius=1, color='blue',fill=False)
ax.add_patch(circle1)
ax.set_aspect('equal')
plt.show()

plt.clf()
plt.quiver(X,Y,np.transpose(barr[0]),np.transpose(barr[1]))
fig = plt.gcf()
ax = plt.gca()
circle1 = Circle((0, 0), radius=1, color='blue',fill=False)
ax.add_patch(circle1)
ax.set_aspect('equal')
plt.show()

plt.clf()
plt.plot(Y,np.transpose(varr)[0])
plt.show()


end = time.time()
print("total time elapsed "+str(end-begin))
"""