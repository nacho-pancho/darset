# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:38:02 2019

@author: usuario
"""
import datos
import numpy as np
import matplotlib.pyplot as plt


def S(x,a,b,c):
    return a / ( 1.0 + np.exp(-b*(x-c)) )


def fit_vel_pot(x,y):
    """
    Ajuste por descenso por gradiante de la funcion y = f(x; a, b, c) = a / { 1 + e^[-b(x-c)] } = a * S(b(x-c))
    
    l(y,x;a,b,c) = (1/2)|| y - f(x; a, b, c) ||_2^2
   
    dS/da = S(b(x-c))
    dS/db = S(b(x-c))'(x-c) = S(b(x-c)) * [1-S(b(x-c))] * (x-c)
    dS/dc = S(b(x-c))'(-b)  =             ,,            * (-b)
    
    df/db = df/dS * dS/db
    df/dc = df/dS * dS/dc

    En realidad 'a' ya lo sabemos a partir del rango de los datos, por lo que solo ajustamos b y c
    c lo inicializamos al valor medio de x tal que y está cerca de a/2
    y b lo inicializamos a 1
    """
    a = np.max(y)
    y = y / a
    b = 1
    c = np.mean( x[(y >= 0.45) | (y <= 0.55)] )
    currS = S(x, 1, b, c)
    currE = y - currS
    f     = 0.5*np.sum(np.abs(currE)**2)
    tol   = 1e-8
    df    = 1e20
    iter  = 0
    alpha = 1e-4
    beta = 0.8
    while np.abs(df)/(f+1e-10) > tol:
        #
        # gradiente
        #
        dfdS = currS*(1-currS)
        dfdb = np.sum(dfdS*(x-c))
        dfdc = np.sum(dfdS)*(-b)
        #print(np.sum(dfdS), dfdb, dfdc)
        b = b - alpha*dfdb
        c = c - alpha*dfdc
        #
        # evaluar y seguir
        #
        fprev = f
        currS = S(x, 1, b, c)
        currE = y - currS
        f = 0.5*np.sum(currE**2)
        df = fprev - f
        iter = iter + 1
        print(f'iter\t{iter}\tb\t{b}\tc\t{c}\tf\t{f}\tdf\t{df}')
        alpha = alpha*beta
    RMSE = np.sqrt(f/len(y))
    print(f'RMSE={RMSE}')
    return (a,b,c)
    
if __name__ == '__main__':
    M = np.load('M7.npz')['arr_0']
    F = np.load('F7.npz')['arr_0']
    nom = open('n7.txt').read().split()
    col = nom.index('velPRONOS')
    M[:,col] = M[:,col]/3.6

    x_col = nom.index('velPRONOS')
    y_col = nom.index('potSCADA')
    z_col = nom.index('dirSCADA')

    x_med = M[:,x_col]
    y_med = M[:,y_col]
    z_med = M[:,z_col]
    zc_med = np.cos(z_med)
    zs_med = np.sin(z_med)

    x_ok = (F[:,x_col] == 0 ) # & (x_med > datos.FUERA_DE_RANGO)
    y_ok = (F[:,y_col] == 0 ) # & (y_med > datos.FUERA_DE_RANGO)
        
    todo_ok = x_ok & y_ok
    
    x_med_ok = x_med[todo_ok]
    y_med_ok = y_med[todo_ok]
    fig = plt.figure(figsize=(10,10))
    plt.scatter(x_med_ok, y_med_ok, marker = '.',color=(0,0,0,0.1))
    
    a,b,c = fit_vel_pot(x_med_ok,y_med_ok)
    x_test = np.arange( 0.0, 15.0, 0.25 )
    plt.plot( x_test, S(x_test,a,b,c) )
    plt.show()
    #
    # hay algo raro en los puntos que tienen pot=10... parece que se hubieran
    # filtrado puntos bajo consigna 10MW como si no estuvieran restringidos
    # porque hay generación de 10MW a todas las velocidades de viento
    # (se ve una raya gruesa horizontal @10MW)
    #
    # vamos a sacar esos puntos para hacer el ajuste
    
