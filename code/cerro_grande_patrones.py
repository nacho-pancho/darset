# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:32:25 2019

@author: fpalacio
"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import archivos
import matplotlib.pyplot as plt
import numpy as np
import gen_series_analisis_serial as seriesAS
import pandas as pd
import datos
import graficas
import filtros 
import datetime
import math
import os
import copy

import keras.backend as K

def my_loss(y_true, y_pred):
    
    custom_loss = K.square(K.mean(y_pred, axis=1)-K.mean(y_true, axis=1))
   
    return custom_loss

from keras.layers import Dense, Dropout, LSTM

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# import regularizer
from keras.regularizers import l2

if __name__ == '__main__':



    
    plt.close('all')
    
    

    parque1 = archivos.leerArchivosCentral(5)
    parque1.registrar() 
    medidor1 = parque1.medidores[0]
    filtros1 = parque1.get_filtros()
    M1, F1, nom1, t1 = parque1.exportar_medidas()
    #nom_series_p1 = ['velGEN','dirGEN','velPRONOS','dirPRONOS','potSCADA']
    nom_series_p1 = ['velGEN','cosdirGEN','sindirGEN']
    vel_GEN_5 = parque1.medidores[0].get_medida('vel','gen')
    vel_scada_5 = parque1.medidores[0].get_medida('vel','scada')
    dir_scada_5 = parque1.medidores[0].get_medida('dir','scada')
    dir_pronos_5 = parque1.medidores[0].get_medida('dir','pronos')

    parque2 = archivos.leerArchivosCentral(7)
    parque2.registrar()
    medidor2 = parque2.medidores[0]
    filtros2 = parque2.get_filtros()
    M2, F2, nom2, t2 = parque2.exportar_medidas()
    #nom_series_p2 = ['velPRONOS','dirPRONOS','potSCADA']
    #nom_series_p2 = ['velGEN','potSCADA']
    nom_series_p2 = ['cosdirPRONOS','sindirPRONOS','potSCADA']
    vel_PRONOS_7 = parque2.medidores[0].get_medida('vel','pronos')
    vel_GEN_7 = parque2.medidores[0].get_medida('vel','gen')
    vel_SCADA_7 = parque2.medidores[0].get_medida('vel','scada')
    
    carpeta_central = archivos.path(parque2.id)

    t, M, F, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                 nom_series_p1, nom_series_p2)    


    dt_ini_calc, dt_fin_calc = archivos.leer_ro_pendientes(parque2.id)
    


    # creo df donde voy a guardar los resultados de las RO
    columns_ro = ['dt_ini', 'dt_fin', 'Estimacion [MWh]', 'Error_PE_70% [MWh]',
                  'Error_PE_30% [MWh]', 'Error_VE [MWh]', 'Delta Error VE - PE70 [MWh]',
                  'EG [MWh]', 'ENS VE [MWh]', 'ENS PE_70 [MWh]']
        
    df_ro = pd.DataFrame(columns=columns_ro)    
  
    


    # Normalizo datos 
    
    df_M = pd.DataFrame(M, index=t, columns=nom_series) 
    df_F = pd.DataFrame(F, index=t, columns=nom_series)
    
    df_M_ = df_M[(df_F == 0).all(axis=1)]
    
    
    
    stats = df_M_.describe()
    stats = stats.transpose()

    M_max = np.tile(stats['max'].values,(len(M),1))
    M_min = np.tile(stats['min'].values,(len(M),1))
    
    M_n = (M - M_min )/(M_max - M_min)
        
    max_pot = stats.at['potSCADA_7', 'max']
    min_pot = stats.at['potSCADA_7', 'min']        


    
    # Busco secuencias de patrones que quiero calcular su pot
    
    pot = M[:,-1]
   
    #inicializo la pot igual a la real, luego relleno huecos
    pot_estimada = copy.copy(pot)
    
    filt_pot = F[:,-1]

    delta = 5
    
    
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
            


    for kRO in range(len(Pats_Data_n)):

        carpeta_ro = archivos.path_ro(kRO+1, carpeta_central)
        
        
        print(f"Calculando RO {kRO+1} de {len(Pats_Data_n)}")
        
        X_n,y_n = seriesAS.split_sequences_patrones(F, M_n, Pats_Data_n[kRO],
                                                    Pats_Filt[kRO], Pats_Calc[kRO])

        train_pu = 0.7
        
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
                                X_n, y_n, test_size=1-train_pu, random_state=42)
        
        '''
        k_train = round(len(X_n)*train_pu)
        X_train_n = X_n[:k_train]
        X_test_n = X_n[k_train:]
        
        y_train_n = y_n[:k_train]
        y_test_n = y_n[k_train:]
        '''
   
        
        n_features = X_n.shape[1]
        n_output = y_n.shape[1]
        #defino la red
        
        l2_ = l2(0.000001)
        
        model = Sequential()
        model.add(Dense(n_output*5, input_dim=n_features, kernel_regularizer=l2_, bias_regularizer=l2_))
        model.add(Dense(n_output*10, input_dim=n_features, kernel_regularizer=l2_, bias_regularizer=l2_))
        #model.add(Dense(n_features*5, activation='tanh', kernel_regularizer=l2_, bias_regularizer=l2_))
        #model.add(Dense(n_features*5, activation='tanh', kernel_regularizer=l2_, bias_regularizer=l2_))
        model.add(Dense(n_output, activation='sigmoid', kernel_regularizer=l2_, bias_regularizer=l2_))
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        # fit model
        history = model.fit(X_train_n, y_train_n, validation_data=(X_test_n, y_test_n), 
                            epochs=2, verbose=1, callbacks=[es])
        # evaluate the model
        _, train_acc = model.evaluate(X_train_n, y_train_n, verbose=0)
        _, test_acc = model.evaluate(X_test_n, y_test_n, verbose=0)

        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))        
        #print(model.summary())
        
        # plot training history
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.grid()
        #plt.show() 
        plt.savefig(carpeta_ro + 'convergencia.png')
        
        
        
        y_test_predict_n = model.predict(X_test_n) 
        y_train_predict_n = model.predict(X_train_n) 

        kini_RO = kini_ro[kRO] 
        X_RO_n = Pats_Data_n[kRO][~Pats_Filt[kRO]].flatten()
        X_RO_n = np.asmatrix(X_RO_n)
        y_RO_n = model.predict(X_RO_n)
        y_RO = y_RO_n * (max_pot-min_pot) + min_pot
        y_RO_= np.squeeze(y_RO)
        pot_estimada[kini_RO:kini_RO+y_RO_.size] = y_RO_        
        
        E_gen_RO = sum(pot[kini_RO:kini_RO+y_RO_.size])/6
        
        # desnormalizo 

        y_test_predict = y_test_predict_n * (max_pot-min_pot) + min_pot
        y_train_predict = y_train_predict_n * (max_pot-min_pot) + min_pot
        y_predict_all = np.concatenate((y_test_predict, y_train_predict), axis=0)
        
        
        y_test = y_test_n * (max_pot-min_pot) + min_pot
        y_train = y_train_n * (max_pot-min_pot) + min_pot
        y_all = np.concatenate((y_test, y_train), axis=0)
        
        y_predict_all_acum = np.sum(y_predict_all, axis = 1)
        y_predict_all_acum_MWh = y_predict_all_acum/6

        y_all_acum = np.sum(y_all, axis = 1)
        y_all_acum_MWh = y_all_acum/6
        
        y_dif_all_acum_MWh = np.subtract(y_predict_all_acum_MWh, y_all_acum_MWh)        


        # calculo las dist empíricas para cada rango de ptos
        sort = np.argsort(y_predict_all_acum_MWh)
        y_predict_all_acum_MWh_sort = np.array(y_predict_all_acum_MWh)[sort]
        y_dif_acum_MWh_sort = np.array(y_dif_all_acum_MWh)[sort]
        
        NDatos_hist = 300
        delta_datos= math.trunc(NDatos_hist/2)
        
        y_dif_acum_MWh_sort_PE70 = np.zeros(len(y_dif_acum_MWh_sort))
        y_dif_acum_MWh_sort_PE30 = np.zeros(len(y_dif_acum_MWh_sort))
        y_dif_acum_MWh_sort_PE50 = np.zeros(len(y_dif_acum_MWh_sort))
        y_dif_acum_MWh_sort_VE = np.zeros(len(y_dif_acum_MWh_sort))
        
        for k in range(len(y_predict_all_acum_MWh_sort)):
            idx_izq = max(0, k-delta_datos) 
            idx_der = min(len(y_predict_all_acum_MWh_sort), k+delta_datos)
            
            y_dif_delta = y_dif_acum_MWh_sort[idx_izq:idx_der]
            
            y_dif_acum_MWh_sort_PE70[k] = np.quantile(y_dif_delta, 0.3) 
            y_dif_acum_MWh_sort_PE30[k] = np.quantile(y_dif_delta, 0.7)
            y_dif_acum_MWh_sort_PE50[k] = np.quantile(y_dif_delta, 0.5)
            y_dif_acum_MWh_sort_VE[k] = np.mean(y_dif_delta)
        
        E_est_MWh = sum(y_RO_)/6
        E_dif_MWh_PE70 = np.interp(E_est_MWh,y_predict_all_acum_MWh_sort
                                   , y_dif_acum_MWh_sort_PE70)      
        E_dif_MWh_PE30 = np.interp(E_est_MWh,y_predict_all_acum_MWh_sort
                                   , y_dif_acum_MWh_sort_PE30)
        E_dif_MWh_VE = np.interp(E_est_MWh,y_predict_all_acum_MWh_sort
                                   , y_dif_acum_MWh_sort_VE)

        
        ENS_VE = max(E_est_MWh-E_gen_RO,0)
        
        delta_70 = np.abs(E_dif_MWh_VE-E_dif_MWh_PE70)
        
        ENS_PE_70 = max(E_est_MWh-E_gen_RO-delta_70,0)
        
        plt.figure()
        plt.scatter(y_predict_all_acum_MWh_sort,y_dif_all_acum_MWh, marker = '.',
                    color=(0,0,0,0.1), label = 'Datos')
        plt.plot(y_predict_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE70, label = 'PE70')
        plt.plot(y_predict_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE50, label = 'PE50')
        plt.plot(y_predict_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE30, label = 'PE30')
        plt.plot(y_predict_all_acum_MWh_sort, y_dif_acum_MWh_sort_VE, label = 'VE')

        plt.axvline(E_est_MWh, color='k', linestyle='--', label = 'ENS_estimada')
        plt.legend()
        plt.grid()
        plt.xlabel('E_modelo [MWh]')
        plt.ylabel('E_dif [MWh]')        
        #plt.show()
        plt.savefig(carpeta_ro + 'errores.png')


        archi_ro = carpeta_ro + 'resumen.txt'

        calculos_ro = [dtini_ro[kRO], dtfin_ro[kRO], E_est_MWh, E_dif_MWh_PE70,
                       E_dif_MWh_PE30, delta_70, E_dif_MWh_VE, E_gen_RO, 
                       ENS_VE, ENS_PE_70]
        s = pd.Series(calculos_ro,index=columns_ro)
        s.to_csv(archi_ro, index=True, sep='\t') 
        
        df_ro = df_ro.append(s,ignore_index=True)
        
        
        
    #calculo la medida
    tipoDato = 'pot'
    nrep = filtros.Nrep(tipoDato)
    pCG_mod = datos.Medida('estimacion',pot_estimada , t,
                           tipoDato,'pot_estimada', 0, 60, nrep)     


    pot_scada_mw = parque1.pot
    pot_scada_cg = parque2.pot
    cgm_cg = parque2.cgm
    
    meds = list()
    
    meds.append(pot_scada_cg)
    #meds.append(pot_scada_mw)
    meds.append(cgm_cg)
    meds.append(pCG_mod)
    
    meds.append(vel_GEN_5)
    meds.append(vel_scada_5)

    #meds.append(vel_GEN_7)
    #meds.append(vel_SCADA_7)
    
    #meds.append(vel_PRONOS_7)
    
    #meds.append(dir_pronos_5)
    #meds.append(dir_scada_5)
    

    graficas.clickplot(meds)
    #plt.show()  
    
    # Guardo capturas de pantalla de los datos y estimación de todas las RO

    for kRO in range(len(Pats_Data_n)):
        
        dtini_w = dtini_ro[kRO] - datetime.timedelta(minutes=delta*100)
        dtfin_w = dtfin_ro[kRO] + datetime.timedelta(minutes=delta*100)
        
        graficas.window = [dtini_w, dtfin_w]
        
        graficas.clickplot_redraw()
        
        carpeta_ro = archivos.path_ro(kRO+1, carpeta_central)
        plt.savefig(carpeta_ro + 'datos.png')
    

    # guardo resumen RO
    
    df_ro.to_csv(carpeta_central + 'resumen.txt', index=True, sep='\t') 

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    