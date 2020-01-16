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
    nom_series_p2 = ['velPRONOS','dirPRONOS','radPRONOS','potSCADA']
    vel_PRONOS_7 = parque2.medidores[0].get_medida('vel','pronos')
    
    seriesAS.gen_series_analisis_serial(parque1, parque2, nom_series_p1, nom_series_p2, 
                               True)
    
    t, M, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                           nom_series_p1, nom_series_p2)    
      
    # creo data frame con datos    
    df = pd.DataFrame(M, index=t, columns=nom_series) 
    

    # Filtro valores menores a cero
    df_ = df[(df >= 0).all(axis=1)]
    
    stats = df_.describe()
    stats = stats.transpose()
    stats

    df_norm = (df - stats['mean'])/stats['std']
        
    df_.corr(method='spearman')
    df_norm.corr(method='spearman')
    
    # choose a number of time steps
    n_steps = 1
    n_desf_pot = 0
    
    # convert into input/output
    #X, y = seriesAS.split_sequences(df_.values, n_steps)
    
    X, y, X_orig, y_orig = seriesAS.split_sequences_pot_input(df_norm.values, 
                                                           n_steps, n_desf_pot)
    
    
    #X_orig, y_orig = seriesAS.split_sequences(df.values, n_steps)
    
    print(X.shape, y.shape)
    print(X)
    
    # summarize the data
    #for i in range(len(X)):
    for i in range(len(X)-2,len(X)):
        print(X[i],'/n', y[i])
        
    # define model
    #defino el numero de entradas
    n_features = X.shape[2]
    #defino la red
    model = Sequential()
    model.add(Dense(10,kernel_initializer='normal', activation='linear'))
    model.add(LSTM(10, activation='linear', input_shape=(n_steps, n_features)))
    
    #model.add(Dense(10,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     

    # fit model
    model.fit(X, y, epochs=10)
    
    salida_modelo_TEST_norm = model.predict(X_orig)    
    
    
    plt.figure
    plt.plot(y_orig,salida_modelo_TEST_norm,'b,')
    #plt.plot(salida_obj,entrada_pot_12,'r.')
    plt.xlim((0,60))
    plt.ylim((0,60))
    plt.show
    

    std_pot = stats.at['potSCADA_7', 'std']
    mean_pot = stats.at['potSCADA_7', 'mean']        
    salida_modelo_TEST = salida_modelo_TEST_norm * std_pot + mean_pot

    
    #idx = (df >= 0).all(axis=1).to_numpy()
    for k in range(n_desf_pot + n_steps, len(df.index) - n_steps):
        df['potSCADA_7'][k] = salida_modelo_TEST[k - n_desf_pot - n_steps, 0]

    
    RMS = np.mean(np.subtract(y_orig, salida_modelo_TEST[:,0])** 2) ** .5
    print('RMS = ' + str(RMS))
        
    tipoDato = 'pot'
    nrep = filtros.Nrep(tipoDato)
    pCG_mod = datos.Medida('estimacion', df['potSCADA_7'].to_numpy(), df.index,
                           tipoDato,'pot_estimada', 0, 60, nrep) 
    
    pot_scada_mw = parque1.pot
    pot_scada_cg = parque2.pot
    cgm_cg = parque2.cgm
    
    meds = list()
    
    meds.append(pot_scada_cg)
    meds.append(pot_scada_mw)
    meds.append(cgm_cg)
    meds.append(pCG_mod)
    meds.append(vel_GEN_5)
    meds.append(vel_scada_5)
    meds.append(vel_PRONOS_7)
    
    meds.append(dir_pronos_5)
    meds.append(dir_scada_5)
    

    graficas.clickplot(meds)
    plt.show()
    
    
    
    
    