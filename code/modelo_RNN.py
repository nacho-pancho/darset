# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:30:25 2020

@author: fpalacio
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Set `PYTHONHASHSEED` envcondaironment variable at a fixed value
import os
# Seed value (can actually be different for each attribution step)
seed_value= 1231987

os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

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
        
    max_pot = stats.at['potSCADA_7', 'max']
    min_pot = stats.at['potSCADA_7', 'min']        

    return M_n, max_pot, min_pot  


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


def estimar_ro(train_pu, X_n, y_n, X_RO_n):

        
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
        k1 = 1
        k2 = 1.0

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
        
        print(model.summary())
            
        
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])     
        
        # verificado, respeta seed
        #w = model.get_weights()
        #print(w[0])

        #b = model.get_bias()
        #print(b[0])


        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        # fit model
        
        
        history = model.fit(X_train_n, y_train_n, validation_data=(X_test_n, y_test_n), 
                            epochs=100, verbose=1, callbacks=[es])
       
        # evaluate the model
        '''
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
        
        '''
       
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
    
   