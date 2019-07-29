# -*- coding: utf-8 -*-
"""
No debería haber archivos llamados temp.py en el proyecto!

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname as up
import os 
import rutas as r

def clip(data,minv,maxv):
    return np.minimum(maxv, np.maximum(minv,data) )

os.chdir("../modelado_ro")

nroCentral = 5

carpeta =r.carpeta(nroCentral)
archiSCADA = r.archiSCADA(nroCentral,carpeta)


legends = ('vel','dir','pot','tmp','pre','hum','cgm','dis')
data=np.loadtxt(archiSCADA,skiprows=6)
n,m = data.shape
marcas = np.zeros((n,4))

data = clip(data,-10,360)
rmax = np.std(np.abs(data),axis=0)
print(rmax)

for i in range(m):
    data[:,i] = data[:,i]/rmax[i]
    
seg = np.arange(0,n,n/20,dtype=np.int)
i = 0
ns = len(seg)
for i in range(len(seg)-1):
    print(i)
    a = seg[i]
    b = seg[i+1]
    plt.close('all')
    plt.figure(1,figsize=(150,20))
    clip = data[a:b,:-1]
    clip_marcas = marcas[a:b,:]
    plt.plot(clip,lw=1)
    plt.legend(legends,fontsize='xx-large')
    plt.grid(True)
    plt.savefig(DATADIR + f"c5/c5-{a}-{b}.png",dpi=150)
    np.savez_compressed(DATADIR + f"c5/c5-{a}-{b}.npz",clip=clip)
