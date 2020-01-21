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
    
    
    t, M, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                 nom_series_p1, nom_series_p2)    

    # Normalizo datos 
    
    df = pd.DataFrame(M, index=t, columns=nom_series) 
    
    df = df[(df >= -1000).all(axis=1)]
    
    stats = df.describe()
    stats = stats.transpose()

    M_max = np.tile(stats['max'].values,(len(M),1))
    M_min = np.tile(stats['min'].values,(len(M),1))
    
    M_n = (M - M_min )/(M_max - M_min)
        
    max_pot = stats.at['potSCADA_7', 'max']
    min_pot = stats.at['potSCADA_7', 'min']        


    
    # Busco secuencias de patrones que quiero calcular su pot
    
    pot = M[:,-1]
   
    #inicializo la pot igual a la real, luego relleno huecos
    pot_estimada = pot
    
    filt_pot = pot < -1
    k = 0
    delta = 5
    
    dt_ini_calc = datetime.datetime(2018, 5, 1)
    dt_fin_calc = datetime.datetime(2018, 5, 10)

    dt = t[1] - t[0]    
    k_ini_calc = round((dt_ini_calc - t[0])/dt)
    k_fin_calc = round((dt_fin_calc - t[0])/dt)
        
    k = k_ini_calc
    
    X_Pats_n = list()
    X_calc_n = list()
    kini_calc = list()
    
    while k <= k_fin_calc:
        
        if filt_pot[k]:
            # Encuentro RO
            kiniRO = k
            kini_calc.append(kiniRO)
            # Avanzo RO hasta salir 
            while (k <= k_fin_calc) and (filt_pot[k+1]):
                k = k + 1
           
            kfinRO = k                  
            #Agrego sequencia con RO patron
            x_pat_n = M_n[(kiniRO - delta):(kfinRO + delta),:]
            
            X_Pats_n.append(x_pat_n)
            
            x_calc_n = np.full(len(x_pat_n), False)
            x_calc_n[delta:-delta+1] = True
            X_calc_n.append(x_calc_n)
            print(f"cantidad de ROs = {len(X_Pats_n)}")
            k = k + 1
        else:
            k = k + 1
                
    print(f"{len(X_Pats_n)} RO encontradas en el periodo {dt_ini_calc} a {dt_fin_calc}")
    
    for i in range(len(X_Pats_n)):
        
        print(f"Calculando RO {i+1} de {len(X_Pats_n)}")
        
        X_n,y_n = seriesAS.split_sequences_patrones(M_n, X_Pats_n[i], X_calc_n[i])
            
        
        
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
        model = Sequential()
        model.add(Dense(n_features, input_dim=n_features, kernel_regularizer=l2(0.8), bias_regularizer=l2(0.01)))
        model.add(Dense(n_features*5, activation='tanh', kernel_regularizer=l2(0.8), bias_regularizer=l2(0.01)))
        model.add(Dense(n_features*5, activation='tanh', kernel_regularizer=l2(0.8), bias_regularizer=l2(0.01)))
        model.add(Dense(n_output, activation='tanh', kernel_regularizer=l2(0.8), bias_regularizer=l2(0.01)))
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # fit model
        history = model.fit(X_train_n, y_train_n, validation_data=(X_test_n, y_test_n), 
                            epochs=100, verbose=1, callbacks=[es],initial_epoch = 1)
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
        plt.show() 
        
        
        
        y_test_predict_n = model.predict(X_test_n) 
        y_train_predict_n = model.predict(X_train_n) 
        
        # desnormalizo 
        
        
        y_test_predict = y_test_predict_n * (max_pot-min_pot) + min_pot
        y_train_predict = y_train_predict_n * (max_pot-min_pot) + min_pot

        y_test = y_test_n * (max_pot-min_pot) + min_pot
        
        
        y_test_predict_acum = np.sum(y_test_predict, axis = 1)
        y_test_acum = np.sum(y_test, axis = 1)
        
        y_dif_acum = np.subtract(y_test_predict_acum,y_test_acum)        
        
        Error_medio = y_dif_acum.mean()
        
        MSE_test = np.square(y_dif_acum).mean()
                
        RMSE_test = MSE_test ** .5
        
        y_dif_acum_pu = np.divide(y_dif_acum,y_test_acum)

        Error_medio_pu = np.divide(y_dif_acum,y_test_acum).mean()

        MSE_test_pu = np.square(y_dif_acum_pu).mean()        
        RMSE_test_pu = MSE_test_pu ** .5
        
        print(f"MSE_test = {MSE_test} MW")
        print(f"RMSE_test = {RMSE_test} MW")
        print(f"EMed = {Error_medio} MW")
        print(f"MSE_test = {MSE_test_pu*100} %")
        print(f"RMSE_test = {RMSE_test_pu*100} %") 
        print(f"EMed = {Error_medio_pu*100} %")
        

        '''
        plt.figure()
        plt.hist(y_dif_acum_pu, bins=100, cumulative=True, density = True)  
        plt.xlim(-1,1)
        plt.xticks(np.arange(-1, 1, step=0.1))
        plt.ylim(0,1)
        plt.yticks(np.arange(0, 1, step=0.1))
        '''

    
        filt_pat = X_Pats_n[i] < -1000
        
        X_RO_n = X_Pats_n[i][~filt_pat].flatten()
        
        X_RO_n = np.asmatrix(X_RO_n)
        
        y_RO_n = model.predict(X_RO_n)
        
        y_RO = y_RO_n * (max_pot-min_pot) + min_pot
        
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

    meds.append(vel_GEN_7)
    meds.append(vel_SCADA_7)
    
    meds.append(vel_PRONOS_7)
    
    #meds.append(dir_pronos_5)
    #meds.append(dir_scada_5)
    

    graficas.clickplot(meds)
    plt.show()
    
    
    
    
    