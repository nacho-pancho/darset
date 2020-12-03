# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:51:27 2020

@author: jfpbf
"""

import numpy as np
from scipy.stats import norm
import archivos as archi
import fnmatch
import os
import copy
import gaussianize as g
import pandas as pd




def normalizar_datos(M, F, t, nom_series, tipo_norm, carpeta):

    print('Normalizando datos')
    
    # Normalizo datos 
    if tipo_norm == 'Standard':
        M_n = copy.deepcopy(M)
        for col in range(M.shape[1]):            
            idx_val = (F[:,col] == 0)         
            x = M[idx_val, col]
            min_x = min(x)
            max_x = max(x)
            M_n[idx_val, col] = (M_n[idx_val, col] - min_x)/(max_x - min_x) 
            if col == (M.shape[1] - 1):
                max_pot = max_x
                min_pot = min_x    



        '''        
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
        '''
        
        
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
        
       
        GenerarYGuardarLentes( M, F, t, nom_series, 4, carpeta)
        M_n = GaussianisarSeries( M, F, t, nom_series, carpeta )
        #print(M_n)
        
                
    else:
        raise Exception('Tipo de normalizacion ' + tipo_norm + ' no implementada.')
    
    return M_n, max_pot, min_pot


def desnormalizar_datos(datos, t, min_, max_, tipo_norm, nombres, carpeta):

    print('Desnormalizando datos')
    
    if tipo_norm == 'Standard':
        for kvect in range(len(datos)):         
            datos[kvect] = datos[kvect] * (max_- min_) + min_

    elif tipo_norm == 'Gauss':
        
        datos = DesGaussianizarSeries( datos, None, t, nombres, carpeta )
        
        '''
        for kvect in range(len(datos)):         
            datos[kvect] = np.squeeze(series_g[-1].inverse_transform(datos[kvect].T))
        '''


    return datos       





def cargar_lentes(carpeta_lentes, nombres):
    
    # cargo TODOS los lentes asociados a los nombres de las series
    lentes = {}
    for filename in os.listdir(carpeta_lentes):
        for nombre in nombres:            
            if filename.endswith('.npy') and fnmatch.fnmatch(filename, '*' + nombre + '*'):
                lentes[filename] = np.load(carpeta_lentes + filename)
    #print(lentes)            
        
    return lentes


def GenerarYGuardarLentes( Mr, Filt, t, nombres, hs_overlap, carpeta_lentes):

    print('Generando y guardando lentes.')
    horas = np.array([ti.hour for ti in t])

    for col in range(Mr.shape[1]):            
        idx_val = (Filt[:, col] == 0) 
        #print(idx_val)        
        m = Mr[idx_val, col]
        h = horas[idx_val]
        
        for h_lente in range(24):
            inth = []
            h1 = int(h_lente - hs_overlap);
            h2 = int(h_lente + hs_overlap);
            if h1<0:
                inth.extend(range(24 + h1, 23 + 1))
                inth.extend(range(h2 + 1))
            elif h2>23:
                inth.extend(range(h1, 23 + 1))
                inth.extend(range(0, h2 - 24 + 1))                
            else:
                inth.extend(range(h1, h2 + 1))
        
            mask = np.in1d(h, inth)
            #idxh = [horas.index(hi) for hi in inth]
            lente = np.sort(m[mask], axis=None) 
            archi_ = archi.archi_lente(nombres[col], h_lente)

            np.save(carpeta_lentes + archi_, lente)
        


def GaussianisarSeries( Mr, Filt, t, nombres, carpeta_lentes ):

    
    print('Gaussianizando Series')
    
    horas = np.array([ti.hour for ti in t])
    #print(horas)    
    lentes = cargar_lentes(carpeta_lentes, nombres)
        
    Mg = copy.deepcopy(Mr)
        
    for kSerie in range(Mr.shape[1]):
        idx_val = (Filt[:, kSerie] == 0)
        m = Mr[idx_val, kSerie]
        hs = horas[idx_val]
        archis_lentes = [ archi.archi_lente(nombres[kSerie], hi) for hi in hs ]

        paux_lst = []        
        for k in range(len(m)):              
            h = hs[k]               
            archi_lente = archis_lentes[k]
            lente = lentes[archi_lente]
            ni = len(lente);
            ii = np.amax(np.where(lente <= m[k])) # le saco el menor o igual xq me esta dando NaN

            #print(ii)            
            if (ii.size == 0):   # menor que el menor dato 
              paux = 0.5             
            elif (ii == ni-1):    # mayor o igual que el mayor dato
              paux = (ii - 0.5)
            else:
              paux = (ii - 0.5) * (m[k] - lente[ii]) + (ii - 1.5) * (lente[ii + 1] - m[k])
              paux = paux / (lente[ii+1]-lente[ii])

                       
            paux = max( paux, 0.5 )
            paux = paux / ni
            
            paux_lst.append(paux)
            
        Mg[idx_val, kSerie] = norm.ppf(paux_lst)
        
    
    print(Mg)

    return Mg


def DesGaussianizarSeries( Mg, Filt, t, nombres, carpeta_lentes ):

    print('DesGaussianizando Series')
      
    lentes = cargar_lentes(carpeta_lentes, nombres)
    
    Mr = copy.deepcopy(Mg)
    
    for k_lst in range(len(Mg)):

        nombre = nombres[min(k_lst, len(nombres) - 1)]

        for k_serie in range(len(Mg[k_lst])):

            if (Filt != None):
                idx_val = (Filt[k_lst][k_serie] == 0)
            else:
                idx_val = slice(len(Mg[k_lst][k_serie]))
               
            mg = norm.cdf(Mg[k_lst][k_serie][idx_val])         
            ts = t[k_lst][k_serie][idx_val]           
            archis_lentes = [ archi.archi_lente(nombre, ti.hour) for ti in ts ]
            
            if len(mg) > len(archis_lentes):
                print('Hay algo mal: len(mg) > len(archis_lentes)')
                print(ts)
                print(mg)
            
            m_lente = []                       
            for k in range(len(mg)):
                archi_lente = archis_lentes[k]
                lente = lentes[archi_lente]
                #kidx_lente = int((mg[k]) * len(lente) - 0.5)
                kidx_lente = int(mg[k] * (len(lente) - 1))
                m_lente.append(lente[kidx_lente])

            '''
            print('mr')
            print(m_lente)
            '''            
            
            Mr[k_lst][k_serie][idx_val] = m_lente   
       
    print('DesGaussianización Finalizada')
    return Mr



