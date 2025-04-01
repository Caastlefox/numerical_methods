import time
begin = time.time()

import math
import matplotlib.pyplot as plt
import numpy as np

def pot(x):
    return -1*np.exp(-x**2)-1.2*np.exp(-(x-2)**2)

def ifconverge(x,y,tolerance=1e-09):
    if abs(x/tolerance) == abs(y/tolerance):
        print(x)
        print(abs(y))
        return True
    return False    

def dpodx(x):
    dx = 0.001
    return (pot(x+dx)-pot(x-dx))/(2*dx)

def dpodx2(x):
    dx = 0.001
    return (pot(x+dx)+pot(x-dx)-2*pot(x))/(dx**2)

def F1(xn1,vn1,xn,vn):
    dt =0.01
    return xn1-xn-dt*(vn1-vn)/2
    
def F2(xn1,vn1,xn,vn,alpha):
    dt =0.01
    return vn1-vn+dt*(dpodx2(xn1)/m+vn1*alpha+dpodx2(xn)/m+vn*alpha)/2


x0 = 2.8
v0 = 0
m =1
alpha = 0
dt = 0.001
x = []
v = []
x.append(x0)
v.append(v0)
xit = []
vit = []
xit.append(x0)
vit.append(v0)
iters = 10
i = 0
flag = True
while True:
    a11 = 1
    a12 = -dt/2
    a21 = dt*dpodx2(xit[i])/(2*m)
    a22 = 1+alpha*dt/2
    W   = a11*a22-a12*a21
    Wx  = -F1(xit[i],vit[i],x[0],v[0])*a22-F2(xit[i],vit[i],x[0],v[0],alpha)*a12
    Wv  = -F2(xit[i],vit[i],x[0],v[0],alpha)*a11+F1(xit[i],vit[i],x[0],v[0])*a21
    dx  = Wx/W
    dv  = Wv/W
    if not math.isclose(xit[i],xit[i]+dx):
        xit.append(xit[i]+dx)
    else:
        xit.append(xit[i])
        if flag == True:
            xfound = i
        flag = False
    if not math.isclose(vit[i],vit[i]+dv):
        vit.append(vit[i]+dv)
    else:
        vit.append(vit[i])
        vfound = i
    if math.isclose(xit[i],xit[i]+dx) and math.isclose(vit[i],vit[i]+dv):
        break        
    i += 1

print("zbieznosc xn=1 dla: ",xit[-1],"w kroku: " ,xfound)
print("zbieznosc vn=1 dla: ",vit[-1],"w kroku: " ,vfound)

end = time.time()
print(begin-end)