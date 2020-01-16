# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:32:25 2019

@author: fpalacio
"""
import archivos
import matplotlib.pyplot as plt
import numpy as np
import gen_series_analisis_serial as seriesAS
import pandas as pd
import datos
import graficas
import filtros 
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, LSTM
from keras.layers import SimpleRNN, Embedding
from keras.models import Sequential

if __name__ == '__main__':
    
    plt.close('all')
    

    parque1 = archivos.leerArchivosCentral(5)
    parque1.registrar()
    medidor1 = parque1.medidores[0]
    filtros1 = parque1.get_filtros()
    M1, F1, nom1, t1 = parque1.exportar_medidas()
    #nom_series_p1 = ['velGEN','dirGEN','velPRONOS','dirPRONOS','potSCADA']
    nom_series_p1 = ['velGEN','dirGEN']
    vel_GEN_5 = parque1.medidores[0].get_medida('vel','gen')
    vel_scada_5 = parque1.medidores[0].get_medida('vel','scada')
    dir_scada_5 = parque1.medidores[0].get_medida('dir','scada')
    dir_pronos_5 = parque1.medidores[0].get_medida('dir','pronos')

    parque2 = archivos.leerArchivosCentral(7)
    parque2.registrar()
    medidor2 = parque2.medidores[0]
    filtros2 = parque2.get_filtros()
    M2, F2, nom2, t2 = parque2.exportar_medidas()
    nom_series_p2 = ['velPRONOS','dirPRONOS','potSCADA']
    vel_PRONOS_7 = parque2.medidores[0].get_medida('vel','pronos')
    
    
    t, M, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                 nom_series_p1, nom_series_p2)    

    
    # Busco secuencias de patrones que quiero calcular su pot
    
    pot = M[:,-1]
   
    #inicializo la pot igual a la real, luego relleno huecos
    pot_estimada = pot
    
    filt_pot = pot < -1
    k = 0
    delta = 10
    dt_ini_calc = datetime.datetime(2018, 5, 10)
    dt_fin_calc = datetime.datetime(2018, 5, 15 )

    dt = t[1] - t[0]    
    k_ini_calc = round((dt_ini_calc - t[0])/dt)
    k_fin_calc = round((dt_fin_calc - t[0])/dt)
        
    k = k_ini_calc
    
    X_Pats = list()
    X_calc = list()
    kini_calc = list()
    
    while k <= k_fin_calc:
        
        if filt_pot[k]:
            # Encuentro RO
            kiniRO = k
            kini_calc.append(kiniRO)
            # Avanzo RO hasta salir 
            while filt_pot[k+1]:
                k = k + 1
           
            kfinRO = k                  
            #Agrego sequencia con RO patron
            x_pat = M[(kiniRO - delta):(kfinRO + delta),:]
            
            X_Pats.append(x_pat)
            
            x_calc = np.full(len(x_pat), False)
            x_calc[delta:-delta+1] = True
            X_calc.append(x_calc)
            print(f"cantidad de ROs = {len(X_Pats)}")
            k = k + 1
        else:
            k = k + 1
                
    print(f"{len(X_Pats)} RO encontradas en el periodo {dt_ini_calc} a {dt_fin_calc}")
    
    for i in range(len(X_Pats)):
        
        print(f"Calculando RO {i} de {len(X_Pats)}")
        
        X,y = seriesAS.split_sequences_patrones(M, X_Pats[i], X_calc[i])
            
        train_pu = 0.7
        
        k_train = round(len(X)*train_pu)
        X_train = X[:k_train]
        X_test = X[k_train:]
        
        y_train = y[:k_train]
        y_test = y[k_train:]
        
        
        n_features = X.shape[1]
        n_output = y.shape[1]
        #defino la red
        model = Sequential()
        model.add(Dense(n_features, activation='linear', input_dim=n_features))
        model.add(Dense(10, activation='linear'))
        model.add(Dense(n_output, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     
        
        #print(model.summary())
        
        
        # fit model
        model.fit(X_train, y_train, epochs=15, verbose=0)
        
        y_test_predict = model.predict(X_test)    
        y_dif = np.subtract(y_test,y_test_predict)
        
        MSE = np.square(y_dif).mean()
        RMSE = MSE ** .5
        
        print(f"Error medio = {y_dif.mean()} MW")
        print(f"Error cuadrático medio = {RMSE} MW")
        
        
        #plt.figure
        #plt.plot(y_test,y_test_predict,'b,')
    
        filt_pat = X_Pats[i] < -1000
        
        X_RO = X_Pats[i][~filt_pat].flatten()
        
        X_RO_ = np.asmatrix(X_RO)
        
        y_RO = model.predict(X_RO_)
        
        y_RO_= np.squeeze(y_RO)
        
        kini_RO = kini_calc[i] 
        
        pot_estimada[kini_RO:kini_RO+y_RO_.size] = y_RO_
    


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
    meds.append(vel_PRONOS_7)
    
    #meds.append(dir_pronos_5)
    #meds.append(dir_scada_5)
    

    graficas.clickplot(meds)
    plt.show()
    
    
    
    
    