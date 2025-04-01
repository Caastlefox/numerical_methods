import numpy as np
from numba import njit
import time
import matplotlib.pyplot as plt
H    = 5
dz   = 0.01
dt   = 60
N    = int(H/dz + 1) 
iter_num = 200000
Tm = 8
Talpha = 5
alpha = 6*10**(-7)#m^2/s
P = 24*3600#s
D = np.sqrt(alpha*P/np.pi)#m
@njit
def T(z,t):
    return Tm + Talpha * np.sin(2*np.pi*t/P-(z/D))*np.exp(-z/D)#Unit K

@njit
def T0(t):
    return Tm + Talpha * np.sin(2*np.pi*t/P)

def zmiennosc(siatka,h):
    plt.clf()
    pos = int(h/dz)+1
    mesh_sliced = siatka[iter_num-10000:iter_num,pos] 
    plt.plot(np.linspace(iter_num-10000,iter_num-1,10000)/60,mesh_sliced)
    name = "Zmienność czasowa temperatury na głębokości "+str(h)+" m"
    plt.xlabel("czas[h]")
    plt.ylabel("Temperatura na głębokości "+str(h)+" m[K]")
    plt.title(name)
    plt.show()
    start = False
    counter = 2
    for i in range(1,10000):
        if start:
            if (mesh_sliced[i]-8)*(mesh_sliced[i-1]-8) < 0:
                counter -= 1
                if counter == 0:
                    print("wyznaczono zmiennosc czasowa jako"+str((i-istart)/60)) 
                    break
            continue
        if (mesh_sliced[i]-8)*(mesh_sliced[i-1]-8) < 0:
            istart = i
            start = True
    

def glebokosc(siatka):
    for i in range(0,N):
        for t in range(0,iter_num):
            if abs(np.transpose(siatka)[i][t]-8) > 0.1:
                break
            if t == iter_num-1:
                return i
    return "no detected"
    
@njit
def zad1():
    siatka = np.zeros((iter_num,N))
    for i in range(0,N):
        siatka[0][i] = 8
    for t in range(1,iter_num):
        siatka[t][0] = T0(t*dt)
        siatka[t][N-1] = 8
        for i in range(1,N-1):
            siatka[t][i] = siatka[t-1][i]+alpha*dt*(siatka[t-1][i+1]-2*siatka[t-1][i]+siatka[t-1][i-1])/(dz**2)
    return siatka

@njit
def zad2():
    siatka = np.zeros((iter_num,N))
    for t in range(0,iter_num):
        for i in range(0,N):
            siatka[t][i] = T(i*dz,t*dt)
    return siatka

@njit
def FTCS():
    siatka = np.zeros((iter_num,N))
    siatka += 8
    
    for t in range(1,iter_num):
        siatka[t][0] = T0(t*dt)
        siatka[t][N-1] = 8
        for i in range(1,N-1):
            siatka[t][i] = siatka[t-1][i]+alpha*dt*(siatka[t-1][i+1]-2*siatka[t-1][i]+siatka[t-1][i-1])/(dz**2)
    return siatka

@njit
def CN():
    r = alpha*dt/(dz**2)
    siatka = np.zeros((iter_num,N))
    A = np.zeros((N,N))
    B = np.zeros(N)
    
    siatka += 8

    for t in range(1,iter_num):
        siatka[t][0] = T0(t*dt)
        siatka[t][N-1] = 8
        A[0][0] = 1
        A[N-1][N-1] = 1
        B[0] = T0(t*dt)
        B[N-1] = 8
        for i in range(1,N-1):
            A[i][i-1:i+2] = np.array((-r/2,1+r,-r/2))
            B[i] =  (r/2)*siatka[t-1][i-1]+(1-r)*siatka[t-1][i]+(r/2)*siatka[t-1][i+1]
     
        siatka[t][1:N-1] = np.linalg.solve(A,B)[1:N-1]
              
    return siatka
    
begin = time.time()
sol = zad1()
end1 = time.time()

tarr = np.linspace(0,iter_num-1,iter_num)/60
xarr = np.linspace(0,N-1,N)*dz*-1
X,Y = np.meshgrid(tarr,xarr)
zmiennosc(sol,0)
zmiennosc(sol,0.5)
zmiennosc(sol,1)
glebokosc(sol)


plt.clf()
plt.pcolormesh(X,Y,np.transpose(sol))
plt.colorbar()
name = "T(x,t) - metoda FTCS"
plt.xlabel("czas[h]")
plt.ylabel("glebokosc[m]")
plt.title(name)
plt.show()



#analityczne
ansol = zad2()
plt.clf()
#transpozycja wymagana ponieważ matplotlib oczekuje y jako pierwszą współrzędną
plt.pcolormesh(X,Y,np.transpose(ansol))
plt.colorbar()
name = "T(x,t) - analityczne"
plt.xlabel("czas[h]")
plt.ylabel("glebokosc[m]")
plt.title(name)
plt.show()

#Metoda CN 
begin2 = time.time()
sol2 = CN()
end2 = time.time()

tarr = np.linspace(0,iter_num-1,iter_num)/60
xarr = np.linspace(0,N-1,N)*dz*-1
X,Y = np.meshgrid(tarr,xarr)

plt.clf()
plt.pcolormesh(X,Y,np.transpose(sol2))
plt.colorbar()
name = "T(x,t) - metoda C-N"
plt.xlabel("czas[h]")
plt.ylabel("glebokosc[m]")
plt.title(name)
plt.show()

plt.clf()
plt.scatter(xarr,sol[int(24*60)]-ansol[int(24*60)])
name = "blad numeryczny metody FTCS(czas-24h)"
plt.xlabel("glebokosc[m]")
plt.ylabel("blad[K]")
plt.title(name)
plt.show()

plt.clf()
plt.scatter(xarr,sol2[int(24*60)]-ansol[int(24*60)])
name = "blad numeryczny metody C-N(czas-24h)"
plt.xlabel("glebokosc[m]")
plt.ylabel("blad[K]")
plt.title(name)
plt.show()

end = time.time()
print("czas obliczeń metody FTCS: " + str(end1-begin)  + " s")
print("czas obliczeń metody C-N: " + str(end2-begin2) + " s")
print("czas wykonania programu: " + str(end-begin) + " s")