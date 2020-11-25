# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:18:57 2020

@author: jfpbf
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math


def estimar_pot_PE(y_test_e, y_train_e, y_test, y_train, y_RO_e, y_RO_gen,
                   carpeta_ro, Pmax):

    '''
    # para calcular PE 70 % uso datos de entrenamiento y validación.
    y_e = np.concatenate((y_test_e, y_train_e), axis=0)
    y_r = np.concatenate((y_test, y_train), axis=0)
    '''
    
    y_e = y_test_e
    y_r = y_test    
    
    # cantidad de datos que se consideran para calcular las dist empíricas
    NDatos_hist = 300
    delta_datos = math.trunc(NDatos_hist/2)
    
    y_e_acum = np.sum(y_e, axis = 1)
    y_e_acum = y_e_acum/6

    y_r_acum = np.sum(y_r, axis = 1)
    y_r_acum = y_r_acum/6
    
    y_dif_acum = np.subtract(y_e_acum, y_r_acum) 
    
    # calculo las dist empíricas para cada rango de ptos
    sort = np.argsort(y_e_acum)
    y_e_acum_sort = np.array(y_e_acum)[sort]
    y_dif_acum_sort = np.array(y_dif_acum)[sort]       
    
    y_dif_acum_sort_PE70 = np.zeros(len(y_dif_acum_sort))
    y_dif_acum_sort_PE30 = np.zeros(len(y_dif_acum_sort))
    y_dif_acum_sort_PE50 = np.zeros(len(y_dif_acum_sort))
    y_dif_acum_sort_VE = np.zeros(len(y_dif_acum_sort))
    
    for k in range(len(y_e_acum_sort)):
        
        idx_izq = max(0, k - delta_datos) 
        idx_der = min(len(y_e_acum_sort), k + delta_datos)
        
        y_dif_delta = y_dif_acum_sort[idx_izq:idx_der]
        
        y_dif_acum_sort_PE70[k] = np.quantile(y_dif_delta, 0.3) 
        y_dif_acum_sort_PE30[k] = np.quantile(y_dif_delta, 0.7)
        y_dif_acum_sort_PE50[k] = np.quantile(y_dif_delta, 0.5)
        y_dif_acum_sort_VE[k] = np.mean(y_dif_delta)
    
    
    # interpolo las dist empíricas en la energía estimada para la RO
    # para calcular las excedencias del error        
    
    y_e_RO_acum = np.sum(y_RO_e)/6        
    E_dif_PE70 = np.interp(y_e_RO_acum,y_e_acum_sort
                               , y_dif_acum_sort_PE70)      
    E_dif_PE30 = np.interp(y_e_RO_acum,y_e_acum_sort
                               , y_dif_acum_sort_PE30)
    E_dif_VE = np.interp(y_e_RO_acum, y_e_acum_sort
                               , y_dif_acum_sort_VE)

    y_r_RO_acum = sum(y_RO_gen)/6
            
    ENS_VE = max(y_e_RO_acum - y_r_RO_acum, 0)
    
    delta_70 = np.abs(E_dif_VE - E_dif_PE70)

    #E_est_MWh_PE70 = E_est_MWh - delta_70
    E_est_PE70 = y_e_RO_acum + E_dif_PE70           
    
    ENS_PE_70 = max(E_est_PE70 - y_r_RO_acum, 0)
    
    pPE70 = y_RO_e * E_est_PE70 / y_e_RO_acum
    
    # Esta función topea la potencia y redistribuye el resto en las horas
    # en las que no aplica el tope
    
    # aca tengo que hacer una funcion que distribuya ENS_PE_70 de modo 
    # que cuando sume (PE_70_calc - PGen) =  ENS_PE_70         
      
    plt.figure()
    plt.scatter(y_e_acum_sort,y_dif_acum, marker = '.',
                color=(0,0,0,0.1), label = 'Datos')
    plt.plot(y_e_acum_sort, y_dif_acum_sort_PE70, label = 'PE70')
    plt.plot(y_e_acum_sort, y_dif_acum_sort_PE50, label = 'PE50')
    plt.plot(y_e_acum_sort, y_dif_acum_sort_PE30, label = 'PE30')
    plt.plot(y_e_acum_sort, y_dif_acum_sort_VE, label = 'VE')

    plt.axvline(y_e_RO_acum, color='k', linestyle='--', label = 'E_Estimada')
    EMax = Pmax * len(y_RO_e) / 6
    plt.axvline(EMax, color='k', linestyle='-.', label = 'E_Max')
    plt.legend()
    plt.grid()
    plt.xlabel('E_modelo [MWh]')
    plt.ylabel('E_modelo - E_gen [MWh]')        
    #plt.show()
    plt.savefig(carpeta_ro + 'errores.png')
    
    return pPE70, ENS_VE, delta_70, E_est_PE70, ENS_PE_70, E_dif_PE30, E_dif_PE70, E_dif_VE


def calcular_distE_yi(y_r, y_e, carpeta_ro):

    
    for kyi in range(len(y_r[0])):
        
        err = [y_r[kdato][kyi] - y_e[kdato][kyi] for kdato in range(len(y_r))]
        
        # Fit a normal distribution to the data:
        mu, std = norm.fit(err)
        
        # Plot the histogram.
        plt.figure()
        plt.hist(err, bins=60, density=True, alpha=0.6, color='g')
        
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)        
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        
        plt.savefig(carpeta_ro + 'errores_y_' + str(kyi) + '.png')   