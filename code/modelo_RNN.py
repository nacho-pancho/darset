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
import matplotlib
from sklearn.multioutput import MultiOutputRegressor
#matplotlib.use('Agg')

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


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import gc
import gaussianize as g
import Gaussianizacion as gauss


series_g = []

def my_loss(y_true, y_pred):
    
    custom_loss = K.square(K.mean(y_pred, axis=1)- K.mean(y_true, axis=1))
   
    return custom_loss


def normalizar_datos(M, F, t, nom_series, tipo_norm, carpeta):

    print('Normalizando datos')
    
    # Normalizo datos 
    if tipo_norm == 'Standard':

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

    elif tipo_norm == 'Gauss':

        max_pot = -99999999
        min_pot = max_pot
 
        '''        
        global series_g
        M_n = copy.copy(M) 

            
        for col in range(M.shape[1]):            
            idx_val = (F[:,col] == 0)         
            x = M[idx_val, col]
            out = g.Gaussianize(strategy='brute') 
            out.fit(x)
            x_n = np.squeeze(out.transform(x))
            M_n[idx_val, col] = x_n
            series_g.append(out)      
        '''
        
       
        gauss.GenerarYGuardarLentes( M, F, t, nom_series, 4, carpeta)
        M_n = gauss.GaussianisarSeries( M, F, t, nom_series, carpeta )
        print(M_n)
        
                
    else:
        raise Exception('Tipo de normalizacion ' + tipo_norm + ' no implementada.')
    
    return M_n, max_pot, min_pot


def desnormalizar_datos(datos, t, min_, max_, tipo_norm, nombres, carpeta):

    print('Desnormalizando datos')
    
    if tipo_norm == 'Standard':
        for kvect in range(len(datos)):         
            datos[kvect] = datos[kvect] * (max_- min_) + min_

    elif tipo_norm == 'Gauss':
        
        datos = gauss.DesGaussianizarSeries( datos, None, t, nombres, carpeta )
        
        '''
        for kvect in range(len(datos)):         
            datos[kvect] = np.squeeze(series_g[-1].inverse_transform(datos[kvect].T))
        '''


    return datos       

# Esta función crea RO y sus patrones para posteriormente ser calculados
def patrones_ro(delta, F, M_n, t, dt_ini_calc, dt_fin_calc):

    filt_pot = F[:,-1]
    
    #print(t)
    dt = t[1] - t[0]    

    k_ini_calc = [round((dt_ - t[0])/dt) for dt_ in dt_ini_calc]
    k_fin_calc = [round((dt_ - t[0])/dt) for dt_ in dt_fin_calc]
    
    Pats_Data_n = list()
    Pats_Filt = list()
    Pats_Calc = list()   

    dtini_ro = list()
    dtfin_ro = list()
    kini_ro = list()
    dt_ro = list()
    
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
                
                dt_RO = [t[0] + k * dt for k in range(kiniRO, kfinRO + 1)]
                dt_ro.append(dt_RO)
                
                #Agrego sequencia con RO patron
                pat_data_n = M_n[(kiniRO - delta):(kfinRO + delta),:]
                pat_filt = F[(kiniRO - delta):(kfinRO + delta),:]
            
                Pats_Data_n.append(pat_data_n)
                Pats_Filt.append(pat_filt)
                          
                x_calc_n = np.full(len(pat_data_n), False)
                x_calc_n[delta:-delta] = True
                Pats_Calc.append(x_calc_n)
    
                k = k + 1
            else:
                k = k + 1  
    
    
    return Pats_Data_n, Pats_Filt, Pats_Calc, dtini_ro, dtfin_ro, kini_ro, dt_ro


def estimar_ro(X_train_n, y_train_n, X_val_n, y_val_n, X_test_n, y_test_n,
               X_RO_n, carpeta_ro, k1, k2):

        
        print('Arranca NN')
    
        n_features = X_train_n.shape[1]
        n_output = y_train_n.shape[1]
        
        #defino la red
        
        l2_ = l2(0.000001)
        #l2_ = l2(0.001)

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
        
        
        model.add(Dense(int(k1*n_output), activation = 'tanh',
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
        
        
        history = model.fit(X_train_n, y_train_n, validation_data=(X_val_n, y_val_n), 
                            epochs=100, verbose=0, callbacks=[es])
        
        print(model.summary())
        NParametros = model.count_params()
        
        # evaluate the model
        
        _, train_acc = model.evaluate(X_train_n, y_train_n, verbose=0)
        _, test_acc = model.evaluate(X_test_n, y_test_n, verbose=0)

        '''
        print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))        
          
        
        # plot training history
        
        '''
        
        
        
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.grid()
        #plt.show() 
        
        plt.savefig(carpeta_ro + 'convergencia.png')
        
        
       
        y_test_predict_n = model.predict(X_test_n) 
        y_train_predict_n = model.predict(X_train_n)
        y_val_predict_n = model.predict(X_val_n)

        
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

        del model
        gc.collect()
        K.clear_session()
        
        return  (y_test_predict_n, y_train_predict_n, y_val_predict_n, 
                 np.squeeze(y_RO_predict_n), NParametros) 

def main_ro(flg_estimar_RO, parque1, parque2, nom_series_p1, nom_series_p2, dt_ini_calc,
            dt_fin_calc, delta_print_datos, meds_plot_p1, meds_plot_p2, 
            flg_print_datos = False, flg_recorte_SMEC = True, tipo_calc = 'NN',
            tipo_norm = 'Gauss'): 

    
    nid_parque = parque2.id   

    meds = meds_plot_p1 + meds_plot_p2
    
    if flg_estimar_RO:
            
        t, M, F, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                     nom_series_p1, nom_series_p2)    
        
        #pltxy.plot_meds(M,F,nom_series,'velGEN_9','potSCADA_57')

        
    
        # creo df donde voy a guardar los resultados de las RO
        columns_ro = ['dt_ini', 'dt_fin', 'largo','Estimacion [MWh]', 'Error_PE_70% [MWh]',
                      'Error_PE_30% [MWh]', 'Error_VE [MWh]', 'Delta Error VE - PE70 [MWh]',
                      'EG [MWh]', 'ENS VE [MWh]', 'ENS PE_70 [MWh]', 'k1_opt', 'k2_opt',
                       'error_pu_opt', 'std_pu_opt', 'b_v_pu_opt', 'NDatosTrain', 
                       'NDatosTest', 'NDatosVal', 'NInput', 'NOutput',
                       'NParametros']
            
        df_ro = pd.DataFrame(columns=columns_ro)    
    

        carpeta_res = archivos.path_carpeta_resultados(nid_parque, tipo_calc, 
                                                       tipo_norm)
        carpeta_lentes = archivos.path_carpeta_lentes(carpeta_res)
    
        # Normalizo datos 
        
        M_n, max_pot, min_pot = normalizar_datos(M, F, t, nom_series, tipo_norm,
                                                 carpeta_lentes)
                           
        pot = M[:,-1]
       
        # inicializo la pot igual a la real, luego relleno huecos
        pot_estimada = np.zeros(len(pot))
        pot_estimada_PE70 = np.zeros(len(pot))

        
        # Busco secuencias de patrones que quiero calcular su pot
        
        delta = max(int(10*(10/archivos.TS_MIN)), 1) # agrega delta datos 10min antes y despues de las RO encontradas
        Pats_Data_n, Pats_Filt, Pats_Calc, dtini_ro, dtfin_ro, kini_ro, dt_RO = \
            patrones_ro(delta, F, M_n, t, dt_ini_calc, dt_fin_calc)       
        n_ro = len(Pats_Data_n)
        largos_ro = np.array(dtfin_ro) - np.array(dtini_ro)
        indices = np.argsort(-1*largos_ro)
        diome = int(n_ro/2)
        
        print(indices[0],largos_ro[indices[0]])
        print(indices[diome],largos_ro[indices[diome]])
        print(indices[-1],largos_ro[indices[-1]])

        #ros = list()        
        #ros = indices[1:2]
        ros = indices
        #ros = range(0, min(2,n_ro))
        #ros = [4]
     

        
        for kRO in ros:
            
            dt_ro = dt_RO[kRO]
            
            carpeta_ro = archivos.path_ro(kRO + 1, carpeta_res)
            
            
            print(f"Calculando RO {kRO+1} de {len(Pats_Data_n)}")
            
            X_n, y_n, dt = seriesAS.split_sequences_patrones(F, M_n, t,  Pats_Data_n[kRO],
                                                 Pats_Filt[kRO], Pats_Calc[kRO])
    
            kini_RO = kini_ro[kRO] 
            X_RO_n = Pats_Data_n[kRO][~Pats_Filt[kRO]].flatten()
            X_RO_n = np.asmatrix(X_RO_n)              

            
            L = largos_ro[kRO]
            print(f"ro: {kRO} L: {L}")
            
            # separo datos de entrenamiento, validación y testeo 
            X_train_n, X_test_n, X_val_n, y_train_n, y_test_n, y_val_n, dt_train,\
              dt_test, dt_val = train_test_val_split(X_n, y_n, dt, 0.6, 0.2, 0.2, 42)

            [y_test, y_train, y_val] = \
                desnormalizar_datos([y_test_n, y_train_n, y_val_n], 
                                    [dt_test, dt_train, dt_val],
                                    min_pot, max_pot, tipo_norm, [nom_series[-1]],
                                    carpeta_lentes)              

            # calibro y calculo para la RO y datos test y entrenamiento                        
            if tipo_calc == 'NN':
                k1, k2, b_v, error_pu, std_pu, y_RO_e, y_test_e, y_train_e,
                y_val_e, NParametros = \
                    estimar_ro_NN(X_train_n, y_train_n, y_train, dt_train, X_val_n,
                                    y_val_n, y_val, dt_val, X_test_n, y_test_n, y_test, dt_test,
                                    X_RO_n, dt_ro, carpeta_ro, min_pot, max_pot,
                                    tipo_norm, nom_series, carpeta_lentes)                     
    
            elif tipo_calc == 'MVLR':                
                y_RO_n_e, y_test_n_e, y_train_n_e, y_val_n_e, NParametros = \
                    estimar_ro_mvlr(X_train_n, y_train_n, X_test_n, y_test_n, 
                                    X_val_n, y_val_n, X_RO_n, carpeta_ro, 
                                    min_pot, max_pot, tipo_norm)

            elif tipo_calc == 'MVLR_R':                
                y_RO_n_e, y_test_n_e, y_train_n_e, y_val_n_e, NParametros = \
                    estimar_ro_mvlr_ridge(X_train_n, y_train_n, X_test_n, y_test_n, 
                                    X_val_n, y_val_n, X_RO_n, carpeta_ro, 
                                    min_pot, max_pot, tipo_norm)
            elif tipo_calc == 'MVLR_L':                
                y_RO_n_e, y_test_n_e, y_train_n_e, y_val_n_e, NParametros = \
                    estimar_ro_mvlr_lasso(X_train_n, y_train_n, X_test_n, y_test_n, 
                                    X_val_n, y_val_n, X_RO_n, carpeta_ro, 
                                    min_pot, max_pot, tipo_norm)                    
            else:
                raise Exception('Tipo de cálculo ' + tipo_calc + ' no implementado.')
            
            
            if tipo_calc != 'NN':
                k1 = -1
                k2 = -1             
                datos_norm = [y_test_n_e, y_train_n_e, y_val_n_e, y_RO_n_e]
                dt = [dt_test, dt_train, dt_val, dt_RO]
                [y_test_e, y_train_e, y_val_e, y_RO_e] = \
                    desnormalizar_datos(datos_norm, dt, min_pot, max_pot, tipo_norm,
                                        [nom_series[-1]], carpeta_lentes)                
                
                [error_pu, std_pu, b_v] = \
                    errores_modelo(y_train, y_train_e, y_test, y_test_e, y_val, y_val_e)            
            
            
            pot_estimada[kini_RO:kini_RO+y_RO_e.size] = y_RO_e            
            
            y_RO_gen = pot[kini_RO:kini_RO+y_RO_e.size]
                                   
            # estimo la potencia con probabilidad de excedencia 70 %            
            pPE70, ENS_VE, delta_70, E_est_PE70, ENS_PE_70, E_dif_PE30, E_dif_PE70, E_dif_VE = \
                estimar_pot_PE(y_test_e, y_train_e, y_test, y_train, y_RO_e,
                               y_RO_gen, carpeta_ro, parque2.PAutorizada)

            pot_estimada_PE70[kini_RO:kini_RO+y_RO_e.size] = pPE70

            calculos_ro = [dtini_ro[kRO], dtfin_ro[kRO], dtfin_ro[kRO] - dtini_ro[kRO],
                           np.sum(y_RO_e)/6, E_dif_PE70, E_dif_PE30, E_dif_VE,
                           delta_70, np.sum(y_RO_gen)/6, ENS_VE, ENS_PE_70, k1, k2,
                           error_pu, std_pu, b_v, len(X_train_n), len(X_test_n), 
                           len(X_val_n), len(X_train_n[0]), len(y_train[0]),
                           NParametros]                     

            archi_ro = carpeta_ro + 'resumen.txt'
            s = pd.Series(calculos_ro, index=columns_ro)
            s.to_csv(archi_ro, index=True, sep='\t') 
            
            df_ro = df_ro.append(s, ignore_index=True) 
            
            # ejemplos de cálculo
            y_e_ej, y_r_ej, t_ej = \
                ejemplos_modelo_test (y_test_e, y_test, dt_test, y_RO_e, 300, 3)           
            
            # grafico los ejemplos
            
            graficar_ejemplos(y_e_ej, y_r_ej, t_ej, meds, parque2,
                              delta_print_datos, carpeta_ro)
            
        # guardo resumen RO
        
        df_ro.to_csv( carpeta_res + 'resumen.txt', index=True, sep='\t',
                     float_format='%.4f') 

    
        # grafico error_pu y std_pu en función del largo de la ro
        df_ro['largo'] = df_ro['largo'].dt.total_seconds()/3600
        ax = df_ro.plot(kind='scatter', x='largo', y='std_pu_opt', 
                      color='DarkBlue', style='.', label='std_pu')
        
        df_ro.plot(kind='scatter', x='largo', y='error_pu_opt', secondary_y=True,
            color='Red', style='.', label='error_pu', ax=ax);
        #plt.show()
        plt.savefig(carpeta_res + 'desempenho.png', dpi=300)

                   
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
        archivos.generar_ens_dte(pot_estimada_PE70/6, pot/6, t, carpeta_res)
        # imprimo la ens topeada para que pinyectada + pnosuministrada < Ptope = Pautorizada
        if flg_recorte_SMEC:
            archivos.generar_ens_topeada(nid_parque, parque2.PAutorizada)
   
 

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
            carpeta_ro = archivos.path_ro(kRO + 1, carpeta_res)            

            plt.savefig(carpeta_ro + 'datos.png', dpi=300)
    elif flg_print_datos:
        for kcalc in range(len(dt_ini_calc)):           
            dtini_w = dt_ini_calc[kcalc] - datetime.timedelta(minutes=delta_print_datos)
            dtfin_w = dt_fin_calc[kcalc] + datetime.timedelta(minutes=delta_print_datos)
            
            graficas.window = [dtini_w, dtfin_w]
            
            graficas.clickplot_redraw()

            carpeta_datos = archivos.path_carpeta_datos(nid_parque) 
            plt.savefig(carpeta_datos + str(kcalc) + '.png' )
        

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


def ejemplos_modelo_test (y_e, y_r, t, y_RO_e, NDatos_delta, NEjemplos):

    delta_datos = math.trunc(NDatos_delta/2)
    
    y_e_acum = np.sum(y_e, axis = 1)
    y_e_acum = y_e_acum/6

    y_r_acum = np.sum(y_r, axis = 1)
    y_r_acum = y_r_acum/6
        
    y_e_RO_acum = np.sum(y_RO_e)/6
       
    # ordeno los datos según la energía estimada
    sort = np.argsort(y_e_acum)

    # ordeno el resto de los datos para ser coherentes
    y_e_acum_sort = np.array(y_e_acum)[sort]
    y_r_acum_sort = np.array(y_r_acum)[sort]
    t_sort = np.array(t)[sort, :]
    y_e_sort = np.array(y_e)[sort, :]
    y_r_sort = np.array(y_r)[sort, :]


    # encuentro el índice mas cercano a la estimación 
    idx_RO = np.searchsorted(y_e_acum_sort, y_e_RO_acum, side="left")

    # ahora selecciono un intervalo de datos en torno a la energía estimada
    # durante la RO    
    idx_izq = max(0, idx_RO - delta_datos) 
    idx_der = min(len(y_e_acum_sort), idx_RO + delta_datos)
        
    y_e_delta = y_e_sort[idx_izq:idx_der][:]
    y_r_delta = y_r_sort[idx_izq:idx_der][:]
    t_delta = t_sort[idx_izq:idx_der][:]
    y_e_acum_delta = y_e_acum_sort[idx_izq:idx_der][:]
    y_r_acum_delta = y_r_acum_sort[idx_izq:idx_der][:]
    
    #y_difPUABS_delta = np.subtract(y_e_acum_delta, y_r_acum_delta) 
    #y_difPUABS_delta = np.divide(y_difPUABS_delta, y_r_acum_delta)
    #y_difPUABS_delta = np.absolute(y_difPUABS_delta)
    
    y_difPUABS_delta = np.subtract(y_e_delta, y_r_delta) 
    y_difPUABS_delta = np.square(y_difPUABS_delta)
    #print(y_difPUABS_delta.shape)
    y_difPUABS_delta = np.mean(y_difPUABS_delta, axis=1)
    y_difPUABS_delta = np.sqrt(y_difPUABS_delta)
    y_difPUABS_delta = np.divide(y_difPUABS_delta, np.mean(y_r_delta, axis=1))
    

    # ordeno los datos según el error en p.u absoluto
    sort = np.argsort(y_difPUABS_delta)
    
    plt.figure()
    plt.plot(y_difPUABS_delta[sort])
    
    y_e_def = y_e_delta[sort, :]
    y_r_def = y_r_delta[sort, :]
    t_def = t_delta[sort, :]
    
    idx = np.round(np.linspace(0, len(y_e_def[:, 0]) - 1, NEjemplos)).astype(int)
    
        
    return y_e_def[idx, :], y_r_def[idx, :], t_def[idx, :]


def graficar_ejemplos(y_e, y_r, t, meds, parque, delta_print_datos, carpeta_ro):
    

    for kej in range(len(y_e[:,0])):
        
        meds_ = copy.deepcopy(meds)
        
        pot_estim = datos.Medida('estimacion',y_e[kej,:], t[kej,:],
            'pot','pot_estimada', parque.pot.minval, parque.pot.maxval, 3)     
        
        pot_real = datos.Medida('estimacion',y_r[kej,:], t[kej,:],
            'pot','pot_real', parque.pot.minval, parque.pot.maxval, 3)     
        
        
        meds_.append(pot_estim)
        meds_.append(pot_real)
        
        graficas.clickplot(meds_)    
        
        dtini_w = t[kej,0] - datetime.timedelta(minutes=delta_print_datos)
        dtfin_w = t[kej,-1] + datetime.timedelta(minutes=delta_print_datos)    
        
        graficas.window = [dtini_w, dtfin_w]
        
        graficas.clickplot_redraw()
        
        
        plt.savefig(carpeta_ro + 'datos_ej_' + str(kej) + '.png', dpi=300)
        plt.close('all')
        

def estimar_ro_NN (X_train_n, y_train_n, y_train, dt_train, X_val_n, y_val_n, y_val, dt_val,
                   X_test_n, y_test_n, y_test, dt_test, X_RO_n, dt_RO, carpeta_ro, min_pot, 
                   max_pot, tipo_norm, nom_series, carpeta_lentes):

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
            
            y_test_n_e, y_train_n_e, y_val_n_e, y_RO_n_e, NParametros = \
                estimar_ro(X_train_n, y_train_n, X_val_n, y_val_n, X_test_n, 
                           y_test_n, X_RO_n, carpeta_ro, k1, k2)               
  
            datos_norm = [y_test_n_e, y_train_n_e, y_val_n_e, y_RO_n_e]
            dt = [dt_test, dt_train, dt_val, dt_RO] 
            [y_test_e, y_train_e, y_val_e, y_RO_e] = \
                desnormalizar_datos(datos_norm, dt, min_pot, max_pot, tipo_norm,
                                    [nom_series[-1]], carpeta_lentes)                
            
            [error_pu, std_pu, b_v] = \
                errores_modelo(y_train, y_train_e, y_test, y_test_e, y_val, y_val_e)
            
            
            df_iter_k.loc[k_idx] = [k1, k2, error_pu, std_pu, b_v]
            k_idx = k_idx + 1
            
            print(f"  k1: {k1}, k2: {k2}, b_v: {b_v}")
            
            if b_v < b_v_opt:
                k1_opt = k1
                k2_opt = k2
                b_v_opt = b_v
                error_pu_opt = error_pu
                std_pu_opt = std_pu                        
                y_RO_e_opt = np.array(y_RO_e, copy=True)
                y_test_e_opt = np.array(y_test_e, copy=True)
                y_train_e_opt = np.array(y_train_e, copy=True)
                y_val_e_opt = np.array(y_val_e, copy=True)
                NParametros_opt = NParametros                
    
    df_iter_k.to_csv(carpeta_ro + 'iter_k1k2.txt', index=True, sep='\t',
             float_format='%.2f')
        
    return (k1_opt, k2_opt, b_v_opt, error_pu_opt, std_pu_opt, y_RO_e_opt, 
        y_test_e_opt, y_train_e_opt, y_val_e_opt, NParametros_opt)
    

def estimar_ro_mvlr(X_train_n, y_train_n, X_test_n, y_test_n, X_val_n, y_val_n,
                    X_RO_n, carpeta_ro, min_pot, max_pot, tipo_norm):
        
    
    # Create linear regression object
    regr = linear_model.LinearRegression()

    X_tot_n = np.concatenate((X_train_n, X_val_n), axis=0)
    #print(X_tot_n.size())
    y_tot_n = np.concatenate((y_train_n, y_val_n), axis=0)

    # Train the model using the training sets
    regr.fit(X_tot_n, y_tot_n)
    
    # Make predictions using the testing, training, val and RO set
    y_test_n_e = regr.predict(X_test_n)
    y_train_n_e = regr.predict(X_train_n)
    y_val_n_e = regr.predict(X_val_n)
    y_RO_n_e = regr.predict(X_RO_n)
    
    NParametros = len(regr.coef_)
    
    return (y_RO_n_e, y_test_n_e, y_train_n_e, y_val_n_e, NParametros)


def estimar_ro_mvlr_ridge(X_train_n, y_train_n, X_test_n, y_test_n, X_val_n, y_val_n,
                    X_RO_n, carpeta_ro, min_pot, max_pot, tipo_norm):
    
    X_tot_n = np.concatenate((X_train_n, X_val_n), axis=0)
    #print(X_tot_n.size())
    y_tot_n = np.concatenate((y_train_n, y_val_n), axis=0)
    #print(y_tot_n.size())    
    regr = linear_model.RidgeCV(cv=3)
    
    wrapper = MultiOutputRegressor(regr)
    # fit the model on the whole dataset
    wrapper.fit(X_tot_n, y_tot_n)

    param_nz = np.empty((0))    
    for i in range(len(wrapper.estimators_)):
        coefs = wrapper.estimators_[i].coef_
        idx_coefs_nz = np.nonzero(coefs) 
        coefs_nz = coefs[idx_coefs_nz]
        #print(coefs_nz)
        param_nz = np.concatenate((param_nz, coefs_nz ))
      
    #print(param_nz)
    NParametros = len(param_nz)
    
    # Make predictions using the testing, training, val and RO set
    y_test_n_e = wrapper.predict(X_test_n)
    y_train_n_e = wrapper.predict(X_train_n)
    y_val_n_e = wrapper.predict(X_val_n)
    y_RO_n_e = wrapper.predict(X_RO_n)

    
    return (y_RO_n_e, y_test_n_e, y_train_n_e, y_val_n_e, NParametros)


def estimar_ro_mvlr_lasso(X_train_n, y_train_n, X_test_n, y_test_n, X_val_n, y_val_n,
                    X_RO_n, carpeta_ro, min_pot, max_pot, tipo_norm):

    
    X_tot_n = np.concatenate((X_train_n, X_val_n), axis=0)
    #print(X_tot_n.size())
    y_tot_n = np.concatenate((y_train_n, y_val_n), axis=0)
    #print(y_tot_n.size())    
    regr = linear_model.LassoCV(cv=3)
    
    wrapper = MultiOutputRegressor(regr)
    # fit the model on the whole dataset
    wrapper.fit(X_tot_n, y_tot_n)

    param_nz = np.empty((0))    
    for i in range(len(wrapper.estimators_)):
        coefs = wrapper.estimators_[i].coef_
        idx_coefs_nz = np.nonzero(coefs) 
        coefs_nz = coefs[idx_coefs_nz]
        #print(coefs_nz)
        param_nz = np.concatenate((param_nz, coefs_nz ))
      
    #print(param_nz)
    NParametros = len(param_nz)
    
    # Make predictions using the testing, training, val and RO set
    y_test_n_e = wrapper.predict(X_test_n)
    y_train_n_e = wrapper.predict(X_train_n)
    y_val_n_e = wrapper.predict(X_val_n)
    y_RO_n_e = wrapper.predict(X_RO_n)

    
    return (y_RO_n_e, y_test_n_e, y_train_n_e, y_val_n_e, NParametros)

def train_test_val_split(X_n, y_n, dt, train_pu, test_pu, val_pu, rs):
 
    resto_pu = val_pu + test_pu
    
    X_train_n, X_resto_n, y_train_n, y_resto_n, dt_train, dt_resto = \
        train_test_split(X_n, y_n, dt, test_size = resto_pu, random_state=rs)            
    
    X_test_n, X_val_n, y_test_n, y_val_n, dt_test, dt_val = \
        train_test_split(X_resto_n, y_resto_n, dt_resto,
                         test_size = val_pu/resto_pu, random_state=rs) 
        
    return (X_train_n, X_test_n, X_val_n, y_train_n, y_test_n, y_val_n, dt_train,
           dt_test, dt_val)
    
    
def errores_modelo(y_train, y_train_e, y_test, y_test_e, y_val, y_val_e):
    
    '''
    # concateno (train + test) salidas del modelo y datos reales 
    y_e_all = np.concatenate((y_test_e, y_train_e, y_val_e), axis=0)
    y_all = np.concatenate((y_test, y_train, y_val), axis=0)
    
    
    y_e_all = np.concatenate((y_val_e), axis=0)
    y_all = np.concatenate((y_val), axis=0)
    '''            
    
    y_e_all = y_test_e
    y_all = y_test 
    
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
    
    return (error_pu, std_pu, b_v)