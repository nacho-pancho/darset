# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:30:25 2020

@author: fpalacio
"""

from sklearn.model_selection import train_test_split

import archivos
import matplotlib.pyplot as plt
import gen_series_analisis_serial as seriesAS
import pandas as pd
import datos
import graficas
import filtros 
import datetime
import copy
import math
import numpy as np
import plot_scatter as pltxy

# 1. Set `PYTHONHASHSEED` envcondaironment variable at a fixed value
import os
# Seed value (can actually be different for each attribution step)
seed_value= 1231987

os.environ['PYTHONHASHSEED']=str(seed_value)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['OMP_NUM_THREADS'] = '1'

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow.compat.v1 as tf

tf.random.set_random_seed(seed_value)
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.layers import Dense, Dropout, LSTM
from tensorflow.compat.v1.keras.regularizers import l2
from tensorflow.compat.v1.keras.models import Sequential



def my_loss(y_true, y_pred):
    
    custom_loss = K.square(K.mean(y_pred, axis=1)- K.mean(y_true, axis=1))
   
    return custom_loss


def normalizar_datos(M, F, t, nom_series):
    # Normalizo datos 
    
    df_M = pd.DataFrame(M, index=t, columns=nom_series) 
    df_F = pd.DataFrame(F, index=t, columns=nom_series)
    
    df_M_ = df_M[(df_F == 0).all(axis=1)]
    
    stats = df_M_.describe()
    stats = stats.transpose()

    M_max = np.tile(stats['max'].values,(len(M),1))
    M_min = np.tile(stats['min'].values,(len(M),1))

    M_n = (M - M_min)/(M_max - M_min)
        
    max_pot = stats.at[ nom_series[-1], 'max']
    min_pot = stats.at[ nom_series[-1], 'min']        

    return M_n, max_pot, min_pot


def desnormalizar_datos(datos, min_, max_):
    
    for kvect in range(len(datos)): 
        datos[kvect] = datos[kvect] * (max_-min_) + min_

    return datos       


# Esta función crea RO y sus patrones para posteriormente ser calculados
def patrones_ro(delta, F, M_n, t, dt_ini_calc, dt_fin_calc):

    filt_pot = F[:,-1]
    
    dt = t[1] - t[0]    

    k_ini_calc = [round((dt_ - t[0])/dt) for dt_ in dt_ini_calc]
    k_fin_calc = [round((dt_ - t[0])/dt) for dt_ in dt_fin_calc]
    
    Pats_Data_n = list()
    Pats_Filt = list()
    Pats_Calc = list()   
    

    dtini_ro = list()
    dtfin_ro = list()
    kini_ro = list()
    
    for kcalc in range(len(k_ini_calc)):
            
        kini_calc = k_ini_calc[kcalc]
        kfin_calc = k_fin_calc[kcalc]                 
        
        k = kini_calc
        
        while k <= kfin_calc:
            #print (k)
            if filt_pot[k]:
                # Encuentro RO
                kiniRO = k
                kini_ro.append(kiniRO) 
                dtiniRO = t[0] + kiniRO * dt
                dtini_ro.append(dtiniRO)
                # Avanzo RO hasta salir 
                while (k <= kfin_calc) and (filt_pot[k+1]):
                    k = k + 1
               
                kfinRO = k                  
                dtfinRO = t[0] + kfinRO * dt
                dtfin_ro.append(dtfinRO)
                
                #Agrego sequencia con RO patron
                pat_data_n = M_n[(kiniRO - delta):(kfinRO + delta),:]
                pat_filt = F[(kiniRO - delta):(kfinRO + delta),:]
            
                Pats_Data_n.append(pat_data_n)
                Pats_Filt.append(pat_filt)
                          
                x_calc_n = np.full(len(pat_data_n), False)
                x_calc_n[delta:-delta+1] = True
                Pats_Calc.append(x_calc_n)
    
                k = k + 1
            else:
                k = k + 1  
    
    
    return Pats_Data_n, Pats_Filt, Pats_Calc, dtini_ro, dtfin_ro, kini_ro


def estimar_ro(train_pu, X_n, y_n, X_RO_n, carpeta_ro, k1, k2):

        
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
                                X_n, y_n, test_size=1-train_pu, random_state=42)
        
        n_features = X_n.shape[1]
        n_output = y_n.shape[1]
        
        #defino la red
        
        l2_ = l2(0.000001)

        init_seed = 42699930
        initializer = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05,
                                                       seed=init_seed)
        initializer_b = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01,
                                                       seed=init_seed)

        
        model = Sequential()

        model.add(Dense(int(k1*n_output), input_dim=n_features,#, activation = 'tanh',
                        kernel_regularizer=l2_, bias_regularizer=l2_,
                        kernel_initializer = initializer,
                        bias_initializer= initializer_b))
        
        model.add(Dense(int(k2*n_output), activation = 'tanh',
                        kernel_regularizer=l2_, bias_regularizer=l2_,
                        kernel_initializer = initializer,
                        bias_initializer= initializer_b))

        model.add(Dense(n_output, activation = 'sigmoid',
                        kernel_regularizer=l2_, bias_regularizer=l2_,
                        kernel_initializer = initializer,
                        bias_initializer= initializer_b))
    
        '''
        pregunta 1:
        (k1,k2) optimos??
        grid search: probamos k1 y k2 en cierto rango
        evaluamos b_v y nos quedamos con el mejor en cada caso
        (para un agujero solo)

        pregunta 2:
        orden optimo?
        por que 2 capas?
        por que no sqrt(n_output)?
        por que no cte?

        '''

        '''
        N_MIXES = int(n_output*3)
        
        model.add(mdn.MDN(n_output, N_MIXES))        
        '''
        
        #model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     
        '''
        model.compile(optimizer=keras.optimizers.Adam(), loss=mdn.get_mixture_loss_func(n_output, N_MIXES))
        '''
        
            
        
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     
        
        # verificado, respeta seed
        #w = model.get_weights()
        #print(w[0])

        #b = model.get_bias()
        #print(b[0])


        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # fit model
        
        
        history = model.fit(X_train_n, y_train_n, validation_data=(X_test_n, y_test_n), 
                            epochs=100, verbose=1, callbacks=[es])
       
        # evaluate the model
        
        _, train_acc = model.evaluate(X_train_n, y_train_n, verbose=0)
        _, test_acc = model.evaluate(X_test_n, y_test_n, verbose=0)

        '''
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))        
        
        
        #print(model.summary())
        
        # plot training history
        
        '''
        
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.grid()
        #plt.show() 
        
        plt.savefig(carpeta_ro + 'convergencia.png')
        
        
       
        y_test_predict_n = model.predict(X_test_n) 
        y_train_predict_n = model.predict(X_train_n) 

        
        '''
        y_test_predict_n = np.apply_along_axis(mdn.sample_from_output,1,y_test_predict_n,
                                               n_output,N_MIXES)
        y_test_predict_n = y_test_predict_n[:,0,:]
        

        y_train_predict_n = np.apply_along_axis(mdn.sample_from_output,1,y_train_predict_n,
                                               n_output,N_MIXES)
        y_train_predict_n = y_train_predict_n[:,0,:]
        '''


        y_RO_predict_n = model.predict(X_RO_n)
        
        '''
        y_RO_n = np.apply_along_axis(mdn.sample_from_output,1,y_RO_n,
                                               n_output,N_MIXES)
        y_RO_n = y_RO_n[:,0,:]
        '''        
        
        
        return  y_test_predict_n, y_test_n, y_train_predict_n, y_train_n, np.squeeze(y_RO_predict_n)   

def main_ro(flg_estimar_RO, parque1, parque2, nom_series_p1, nom_series_p2, dt_ini_calc,
            dt_fin_calc, delta_print_datos, meds_plot_p1, meds_plot_p2, flg_print_datos = False): 

    
    nid_parque = parque2.id 
    carpeta_central = archivos.path(nid_parque)    
    
    if flg_estimar_RO:
            
        t, M, F, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                     nom_series_p1, nom_series_p2)    
        
        #pltxy.plot_meds(M,F,nom_series,'velGEN_9','potSCADA_57')


    
        # creo df donde voy a guardar los resultados de las RO
        columns_ro = ['dt_ini', 'dt_fin', 'largo','Estimacion [MWh]', 'Error_PE_70% [MWh]',
                      'Error_PE_30% [MWh]', 'Error_VE [MWh]', 'Delta Error VE - PE70 [MWh]',
                      'EG [MWh]', 'ENS VE [MWh]', 'ENS PE_70 [MWh]', 'k1_opt', 'k2_opt',
                       'error_pu_opt', 'std_pu_opt', 'b_v_pu_opt']
            
        df_ro = pd.DataFrame(columns=columns_ro)    
    
    
        # Normalizo datos 
        
        M_n, max_pot, min_pot = normalizar_datos(M, F, t, nom_series)
                           
        pot = M[:,-1]
       
        # inicializo la pot igual a la real, luego relleno huecos
        pot_estimada = np.zeros(len(pot))
        pot_estimada_PE70 = np.zeros(len(pot))

        
        # Busco secuencias de patrones que quiero calcular su pot
        
        delta = 5 # agrega delta datos 10min antes y despues de las RO encontradas
        Pats_Data_n, Pats_Filt, Pats_Calc, dtini_ro, dtfin_ro, kini_ro = \
            patrones_ro(delta, F, M_n, t, dt_ini_calc, dt_fin_calc)       
        n_ro = len(Pats_Data_n)
        largos_ro = np.array(dtfin_ro) - np.array(dtini_ro)
        indices = np.argsort(largos_ro)
        diome = int(n_ro/2)
        
        print(indices[0],largos_ro[indices[0]])
        print(indices[diome],largos_ro[indices[diome]])
        print(indices[-1],largos_ro[indices[-1]])

        ros = list()        
        #ros = indices[1:2]
        #ros = indices
        #ros = range(0, min(2,n_ro))
        #ros = [4]
        
        for kRO in ros:
            carpeta_ro = archivos.path_ro(kRO + 1, carpeta_central)
            
            
            print(f"Calculando RO {kRO+1} de {len(Pats_Data_n)}")
            
            X_n,y_n = seriesAS.split_sequences_patrones(F, M_n, Pats_Data_n[kRO],
                                                 Pats_Filt[kRO], Pats_Calc[kRO])
    
            kini_RO = kini_ro[kRO] 
            X_RO_n = Pats_Data_n[kRO][~Pats_Filt[kRO]].flatten()
            X_RO_n = np.asmatrix(X_RO_n)  
            
            train_pu = 0.7

            # el mejor esta dando  en (3,3), tope de escala por ahora!
            # k1_lst = [0.25, 0.5, 1, 2, 3]
            #k2_lst = [0.25, 0.5, 1, 2, 3]
            k1_lst = [3]
            k2_lst = [3]
           
            # itero hasta encontrar k1 y k2 óptimo
            b_v_opt = 99999
            # creo df donde voy a guardar los resultados de las RO
            cols_iter_k = ['k1', 'k2', 'error_pu', 'std_pu', 'b_v_pu']
            
            idx = np.arange(0, len(k1_lst)*len(k2_lst))
            df_iter_k = pd.DataFrame(index=idx, columns=cols_iter_k)    
            k_idx = 0
            for k1 in k1_lst:
                for k2 in k2_lst:
                    
                    y_test_e, y_test, y_train_e, y_train, y_RO_e = \
                        estimar_ro(train_pu, X_n, y_n, X_RO_n, carpeta_ro, k1, k2)
            
                    # desnormalizo series
                        
                    datos_norm = [y_RO_e, y_test_e, y_train_e, y_test, y_train]
                    [y_RO_e, y_test_e, y_train_e, y_test, y_train] = \
                        desnormalizar_datos(datos_norm, min_pot, max_pot)
                    
                            
                    # concateno (train + test) salidas del modelo y datos reales 
                    y_e_all = np.concatenate((y_test_e, y_train_e), axis=0)
                    y_all = np.concatenate((y_test, y_train), axis=0)
                    
                                                   
                    y_e_all_acum = np.sum(y_e_all, axis = 1)
                    y_e_all_acum_MWh = y_e_all_acum/6
            
                    y_all_acum = np.sum(y_all, axis = 1)
                    y_all_acum_MWh = y_all_acum/6
                    
                    y_dif_all_acum_MWh = np.subtract(y_e_all_acum_MWh, y_all_acum_MWh)        
                    
                    error_pu = np.mean(y_dif_all_acum_MWh)/np.mean(y_all_acum_MWh)
                    
                    #print('Error medio [p.u] = ' , error_pu)
                    
                    std_pu = np.std(y_dif_all_acum_MWh)/np.mean(y_all_acum_MWh)
            
                    #print('std [p.u] = ' , std_pu)
            
                    b_v = (error_pu ** 2 + std_pu ** 2) ** (1/2)
                    
                    #print('bias-variance [p.u] = ', b_v)
                    
                    
                    df_iter_k.loc[k_idx] = [k1, k2, error_pu, std_pu, b_v]
                    k_idx = k_idx + 1
                    L = largos_ro[kRO]
                    print(f"ro: {kRO} L: {L} k1: {k1}, k2: {k2}, b_v: {b_v}")
                    
                    if b_v < b_v_opt:
                        k1_opt = k1
                        k2_opt = k2
                        b_v_opt = b_v
                        error_pu_opt = error_pu
                        std_pu_opt = std_pu                        
                        y_RO_e_opt = np.array(y_RO_e, copy=True)
                        y_e_all_acum_MWh_opt = np.array(y_e_all_acum_MWh, copy=True)
                        y_dif_all_acum_MWh_opt = np.array(y_dif_all_acum_MWh, copy=True)
                        
            
            df_iter_k.to_csv(carpeta_ro + 'iter_k1k2.txt', index=True, sep='\t',
                     float_format='%.2f')
            
            
            b_v = b_v_opt
            error_pu = error_pu_opt       
            std_pu = std_pu_opt
            y_RO_e = y_RO_e_opt
            y_e_all_acum_MWh = y_e_all_acum_MWh_opt
            y_dif_all_acum_MWh = y_dif_all_acum_MWh_opt
            
            # calculo las dist empíricas para cada rango de ptos
            sort = np.argsort(y_e_all_acum_MWh)
            y_e_all_acum_MWh_sort = np.array(y_e_all_acum_MWh)[sort]
            y_dif_acum_MWh_sort = np.array(y_dif_all_acum_MWh)[sort]
            
            NDatos_hist = 300
            delta_datos = math.trunc(NDatos_hist/2)
            
            y_dif_acum_MWh_sort_PE70 = np.zeros(len(y_dif_acum_MWh_sort))
            y_dif_acum_MWh_sort_PE30 = np.zeros(len(y_dif_acum_MWh_sort))
            y_dif_acum_MWh_sort_PE50 = np.zeros(len(y_dif_acum_MWh_sort))
            y_dif_acum_MWh_sort_VE = np.zeros(len(y_dif_acum_MWh_sort))
            
            for k in range(len(y_e_all_acum_MWh_sort)):
                idx_izq = max(0, k-delta_datos) 
                idx_der = min(len(y_e_all_acum_MWh_sort), k+delta_datos)
                
                y_dif_delta = y_dif_acum_MWh_sort[idx_izq:idx_der]
                
                y_dif_acum_MWh_sort_PE70[k] = np.quantile(y_dif_delta, 0.3) 
                y_dif_acum_MWh_sort_PE30[k] = np.quantile(y_dif_delta, 0.7)
                y_dif_acum_MWh_sort_PE50[k] = np.quantile(y_dif_delta, 0.5)
                y_dif_acum_MWh_sort_VE[k] = np.mean(y_dif_delta)
            
            
            pot_estimada[kini_RO:kini_RO+y_RO_e.size] = y_RO_e        
            
            E_est_MWh = np.sum(y_RO_e)/6        
            E_dif_MWh_PE70 = np.interp(E_est_MWh,y_e_all_acum_MWh_sort
                                       , y_dif_acum_MWh_sort_PE70)      
            E_dif_MWh_PE30 = np.interp(E_est_MWh,y_e_all_acum_MWh_sort
                                       , y_dif_acum_MWh_sort_PE30)
            E_dif_MWh_VE = np.interp(E_est_MWh,y_e_all_acum_MWh_sort
                                       , y_dif_acum_MWh_sort_VE)

            E_gen_RO = sum(pot[kini_RO:kini_RO+y_RO_e.size])/6
                    
            ENS_VE = max(E_est_MWh-E_gen_RO,0)
            
            delta_70 = np.abs(E_dif_MWh_VE-E_dif_MWh_PE70)
    
            E_est_MWh_PE70 = E_est_MWh - delta_70
            
            ENS_PE_70 = max(E_est_MWh_PE70-E_gen_RO,0)
            
            pPE70 = y_RO_e * E_est_MWh_PE70 / E_est_MWh
            
            # Esta función topea la potencia y redistribuye el resto en las horas
            # en las que no aplica el tope
            
            # aca tengo que hacer una funcion que distribuya ENS_PE_70 de modo 
            # que cuando sume (PE_70_calc - PGen) =  ENS_PE_70        
            
            
            '''
            PAut = parque2.PAutorizada 
            
            filt_mayor_PAut = (pPE70 >= PAut)
            
            while np.any(filt_mayor_PAut):
                recorte =np.sum(pPE70[filt_mayor_PAut]) 
                pPE70[filt_mayor_PAut] = PAut
                E_sin_recorte = np.sum(pPE70[~filt_mayor_PAut])
                factor = (E_sin_recorte + recorte)/E_sin_recorte
                pPE70[~filt_mayor_PAut] = pPE70[~filt_mayor_PAut] * factor
                filt_mayor_PAut = pPE70 >= PAut
            '''
            
            pot_estimada_PE70[kini_RO:kini_RO+y_RO_e.size] = pPE70
            
            plt.figure()
            plt.scatter(y_e_all_acum_MWh_sort,y_dif_all_acum_MWh, marker = '.',
                        color=(0,0,0,0.1), label = 'Datos')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE70, label = 'PE70')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE50, label = 'PE50')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE30, label = 'PE30')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_VE, label = 'VE')
    
            plt.axvline(E_est_MWh, color='k', linestyle='--', label = 'ENS_estimada')
            plt.legend()
            plt.grid()
            plt.xlabel('E_modelo [MWh]')
            plt.ylabel('E_dif [MWh]')        
            #plt.show()
            plt.savefig(carpeta_ro + 'errores.png')
    
    
            archi_ro = carpeta_ro + 'resumen.txt'
    
            calculos_ro = [dtini_ro[kRO], dtfin_ro[kRO], dtfin_ro[kRO]-dtini_ro[kRO], E_est_MWh, E_dif_MWh_PE70,
                           E_dif_MWh_PE30, E_dif_MWh_VE, delta_70, E_gen_RO, 
                           ENS_VE, ENS_PE_70, k1_opt, k2_opt, error_pu_opt, 
                           std_pu_opt, b_v_opt]                     
            
            s = pd.Series(calculos_ro, index=columns_ro)
            s.to_csv(archi_ro, index=True, sep='\t') 
            
            df_ro = df_ro.append(s,ignore_index=True)
            
    
        # guardo resumen RO
        
        df_ro.to_csv(carpeta_central + 'resultados/resumen.txt', index=True, sep='\t',
                     float_format='%.4f') 
    
            
        # creo la potencia estimada
        tipoDato = 'pot'
        nrep = filtros.Nrep(tipoDato)
        pot_p2_mod = datos.Medida('estimacion',pot_estimada , t,
                               tipoDato,'pot_estimada', parque2.pot.minval, parque2.pot.maxval, nrep)     
        # potencia 10min con probabilidad 70% de ser excedida
        pot_p2_mod_PE70 = datos.Medida('estimacion',pot_estimada_PE70 , t,
                               tipoDato,'pot_estimada_PE_70', parque2.pot.minval, parque2.pot.maxval, nrep)     
        
        # imprimo el detalle de la energía no suministrada por hora (formato DTE)
        # divido entre 6 para pasar de MW a MWh
        archivos.generar_ens_dte(pot_estimada_PE70/6, pot/6, t,
                                 carpeta_central + 'resultados/')
        
   
    meds = meds_plot_p1 + meds_plot_p2

    if flg_estimar_RO:
        meds.append(pot_p2_mod)
        meds.append(pot_p2_mod_PE70)

    graficas.clickplot(meds)
    
    # Guardo capturas de pantalla de los datos y estimación de todas las RO

    if flg_estimar_RO:
        for kRO in ros:
        #for kRO in range(len(Pats_Data_n)):            
            dtini_w = dtini_ro[kRO] - datetime.timedelta(minutes=delta_print_datos)
            dtfin_w = dtfin_ro[kRO] + datetime.timedelta(minutes=delta_print_datos)
            
            graficas.window = [dtini_w, dtfin_w]
            
            graficas.clickplot_redraw()
            
            carpeta_ro = archivos.path_ro(kRO+1, carpeta_central)
            plt.savefig(carpeta_ro + 'datos.png', dpi=300)
    elif flg_print_datos:
        for kcalc in range(len(dt_ini_calc)):           
            dtini_w = dt_ini_calc[kcalc] - datetime.timedelta(minutes=delta_print_datos)
            dtfin_w = dt_fin_calc[kcalc] + datetime.timedelta(minutes=delta_print_datos)
            
            graficas.window = [dtini_w, dtfin_w]
            
            graficas.clickplot_redraw()
            
            carpeta_datos = archivos.path_carpeta_datos(carpeta_central) 
            plt.savefig(carpeta_datos + str(kcalc) + '.png' )
        
   
    
