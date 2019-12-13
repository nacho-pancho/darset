# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:38:02 2019

@author: usuario
"""
import datos
import numpy as np
import matplotlib.pyplot as plt

def plot_meds(M,F,nom,xnom,ynom,_fig=None,color=(0,0,0,0.1)):
    
    x_col = nom.index(xnom)
    y_col = nom.index(ynom)
    z_col = nom.index('dirPRONOS')
    
    x_med = M[:,x_col]
    y_med = M[:,y_col]
    z_med = M[:,z_col]
    
    x_ok = (F[:,x_col] == 0 ) & (x_med > datos.FUERA_DE_RANGO)
    y_ok = (F[:,y_col] == 0 ) & (y_med > datos.FUERA_DE_RANGO)
    z_ok = (F[:,z_col] == 0 ) 
    
    
    todo_ok = x_ok & y_ok & z_ok
    
    x_med_ok = x_med[todo_ok]
    y_med_ok = y_med[todo_ok]
    if _fig is None:
        fig = plt.figure(figsize=(10,10))
    else:
        fig = _fig
    plt.scatter(x_med_ok, y_med_ok, marker = '.',color=color)
    return fig

if __name__ == '__main__':
    M = np.load('M7.npz')['arr_0']
    F = np.load('F7.npz')['arr_0']
    nom = open('n7.txt').read().split()
    col = nom.index('velPRONOS')
    M[:,col] = M[:,col]/3.6
    #f = plot_meds(M,F,nom,'velSCADA','potSCADA',color=(0.25,0,0,0.1))
    f = plot_meds(M,F,nom,'velPRONOS','potSCADA',color=(0,0,0.25,0.1))
    plt.grid(True)
    plt.show()
