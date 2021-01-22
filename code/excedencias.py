# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:18:57 2020

@author: jfpbf
"""
import numpy as np
from scipy.stats import norm, laplace, spearmanr, pearsonr
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
    
    if isinstance(y_RO_e, list): 
        N10min = len(y_RO_e) 
    else: 
        N10min = 1
    
    EMax = Pmax * N10min / 6
    
    
    plt.axvline(EMax, color='k', linestyle='-.', label = 'E_Max')
    plt.legend()
    plt.grid()
    plt.xlabel('E_modelo [MWh]')
    plt.ylabel('E_modelo - E_gen [MWh]')        
    #plt.show()
    plt.savefig(carpeta_ro + 'errores.png')
    
    return pPE70, ENS_VE, delta_70, E_est_PE70, ENS_PE_70, E_dif_PE30, E_dif_PE70, E_dif_VE


def get_realizaciones_RO(y_r, y_e, y_RO_e, y_RO, cgm_RO, dt_RO, carpeta_ro, 
                         NDatosDist, PMax, NSorteos):

    y_RO_e_rand = []
    y_RO_e_PE95 = []
    y_RO_e_PE05 = []
    y_ENS_e_rand = []
    
    if len(y_RO_e) == 1:
        y_RO_e = y_RO_e[0]  
    
    for kyi in range(len(y_r[0])):
        
        flg_print = (kyi == 0) or (kyi == int(len(y_r[0])/2)) or (kyi == len(y_r[0]) - 1)
        #err = [y_r[kdato][kyi] - y_e[kdato][kyi] for kdato in range(len(y_r))]
        
        y_e_kyi = [y_e[kdato][kyi] for kdato in range(len(y_e))]
        y_r_kyi = [y_r[kdato][kyi] for kdato in range(len(y_r))]
        
        archi = carpeta_ro + 'errores_y_' + str(kyi) + '_ojo.png'
        #print(y_RO_e)
        err = get_dist_empirica_error(y_r_kyi, y_e_kyi, y_RO_e[kyi], NDatosDist, 
                                      archi, PMax, flg_print)
        '''
        y_err = np.subtract(y_e_kyi, y_r_kyi)
        if kyi > 0:
            coef, p =  pearsonr(y_err, y_err_ant)
            print('Corr. Spearman: %.3f' % coef)
        y_err_ant = y_err    
        '''
        
        # Fit a normal distribution to the data:
        #mu, std = norm.fit(err)
        
        ag, bg = laplace.fit(err)  
          
        # Imprimo los ojos de la 1er, última y la muestra yi del medio de la RO.
        if flg_print:           
            # Plot the histogram.
            plt.figure()
            plt.hist(err, bins=60, density=True, alpha=0.6, color='g')
            
            # Plot the PDF.
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            #p = norm.pdf(x, mu, std)
            p = laplace.pdf(x, ag, bg)        
            plt.plot(x, p, 'k', linewidth=2)
            title = "Fit results Laplace: mu = %.2f,  b = %.2f" % (ag, bg)
            plt.title(title)
            archi = carpeta_ro + 'errores_y_' + str(kyi) + '_dist.png' 
            plt.savefig(archi)
        
        
        # Genero realizaciones para cada muestra yi dentro de la RO
        
        yi_RO_e_rand = generar_realizaciones_yi(y_RO_e[kyi], err, dt_RO[0][kyi], 
                                             NSorteos)
        y_RO_e_rand.append(yi_RO_e_rand)
        
        if abs(cgm_RO[kyi] - y_RO[kyi]) <= 4:            
            y_ens_e_rand = [max(yi_RO_e_rand[k] - y_RO[kyi], 0) for 
                            k in range(len(yi_RO_e_rand))]
        else:
            y_ens_e_rand = [0]*len(yi_RO_e_rand)            
        
        y_ENS_e_rand.append(y_ens_e_rand)        
        
        y_RO_e_PE95.append(np.quantile(yi_RO_e_rand, 0.05))
        y_RO_e_PE05.append(np.quantile(yi_RO_e_rand, 0.95))
    
    
    ENS = np.array(y_ENS_e_rand)
    ENS_acum = np.sum(ENS, axis=0)
    ENS_acum_PE70 = np.quantile(ENS_acum, 0.7, interpolation='nearest')
    idx_PE70 = abs(ENS_acum-ENS_acum_PE70).argmin()

    y_RO_e_PE70 = [y_RO_e_rand[k][idx_PE70] for k in range(len(y_RO_e_rand))] 
        
    return y_RO_e_rand, y_RO_e_PE95, y_RO_e_PE05, y_RO_e_PE70
        
        
        
def get_dist_empirica_error(y_r, y_e, y_e_VE, NDatosDist, archi, Pmax, flg_print):

    y_err = np.subtract(y_e, y_r) 
    
    # calculo las dist empíricas para cada rango de ptos
    sort = np.argsort(y_e)
    y_e_sort = np.array(y_e)[sort]
    y_err_sort = np.array(y_err)[sort]       
    
    y_err_sort_PE70 = np.zeros(len(y_err_sort))
    y_err_sort_PE30 = np.zeros(len(y_err_sort))
    y_err_sort_PE50 = np.zeros(len(y_err_sort))
    y_err_sort_VE = np.zeros(len(y_err_sort))
    
    y_err_delta_arr = []
    for k in range(len(y_e_sort)):
        
        idx_izq = max(0, k - NDatosDist) 
        idx_der = min(len(y_e_sort), k + NDatosDist)
        
        y_err_delta = y_err_sort[idx_izq:idx_der]
        y_err_delta_arr.append(y_err_delta)
        
        y_err_sort_PE70[k] = np.quantile(y_err_delta, 0.3) 
        y_err_sort_PE30[k] = np.quantile(y_err_delta, 0.7)
        y_err_sort_PE50[k] = np.quantile(y_err_delta, 0.5)
        y_err_sort_VE[k] = np.mean(y_err_delta)
                  
    
    if flg_print:
        plt.figure()
        plt.scatter(y_e_sort,y_err_sort, marker = '.',
                    color=(0,0,0,0.1), label = 'Datos')
        plt.plot(y_e_sort, y_err_sort_PE70, label = 'PE70')
        plt.plot(y_e_sort, y_err_sort_PE50, label = 'PE50')
        plt.plot(y_e_sort, y_err_sort_PE30, label = 'PE30')
        plt.plot(y_e_sort, y_err_sort_VE, label = 'VE')
        
        plt.axvline(y_e_VE, color='k', linestyle='--', label = 'P_Estimada')
        plt.axvline(Pmax, color='k', linestyle='-.', label = 'P_Max')
        plt.legend()
        plt.grid()
        plt.xlabel('P_modelo [MWh]')
        plt.ylabel('P_modelo - P_gen [MWh]')        
        #plt.show()
        plt.savefig(archi)

    # busco índice con distribuciones más cercano a y_e_VE 
    index = (np.abs(y_e_sort - y_e_VE)).argmin()
    y_err_delta_y_e_VE = y_err_delta_arr[index]
    
    return y_err_delta_y_e_VE


def generar_realizaciones_yi(ye, err, dt, NSorteos):

    NSemilla = NSemilla_dt(dt)
    np.random.seed(NSemilla)
    q = np.random.uniform(low=0, high=1, size=NSorteos)
    
    x_err = np.quantile(err, q)
    
    return ye + x_err
       
    
def NSemilla_dt(dt):
    return int(str(dt.strftime("%Y"))[-2:] + str(dt.strftime("%m")) +
               str(dt.strftime("%d")) + str(dt.strftime("%H")) +
               str(dt.strftime("%M")))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    