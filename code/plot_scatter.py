# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:38:02 2019

@author: usuario
"""
import datos
import matplotlib.pyplot as plt

def plot_meds(M,F,nom,xnom,ynom,_fig=None):
    
    x_col = nom.index(xnom)
    y_col = nom.index(ynom)
    
    x_med = M[:,x_col]
    y_med = M[:,y_col]
    
    x_ok = (F[:,x_col] == 0 ) & (x_med > datos.FUERA_DE_RANGO)
    y_ok = (F[:,y_col] == 0 ) & (y_med > datos.FUERA_DE_RANGO)
    
    todo_ok = x_ok & y_ok
    
    x_med_ok = x_med[todo_ok]
    y_med_ok = y_med[todo_ok]
    if _fig is None:
        fig = plt.figure(figsize=(10,10))
    else:
        fig = _fig
    plt.scatter(x_med_ok, y_med_ok, marker = '.',color=(0,0,0,0.1))
    return fig

#if __name__ == '__main__':
#    M = np.load('M7.npz')
#    F = np.load('F7.npz')    
#    plot_scatter.plot_meds(M,F,nom,'velSCADA','potSCADA')
#    f = plot_scatter.plot_meds(M,F,nom,'velPRONOS','potSCADA',_fig=f)