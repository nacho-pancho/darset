# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:46:05 2020

@author: fpalacio
"""
import numpy as np
from scipy.stats import norm
import archivos as archi
import fnmatch
import os


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

    horas = np.array([ti.hour for ti in t])
    #print(horas)

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
        
    Mg = np.copy(Mr)
        
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
        
    
    #print(Mg)

    return Mg


def DesGaussianizarSeries( Mg, Filt, t, nombres, carpeta_lentes ):

    print('DesGaussianizando Series')
      
    lentes = cargar_lentes(carpeta_lentes, nombres)
    
    Mr = Mg
    
    for k_lst in range(len(Mg)):

        if len(nombres) == 1:
            nombre = nombres[-1]
        else:
            nombre = nombres[k_lst]

        for k_serie in range(len(Mg[k_lst])):

            if (Filt != None):
                idx_val = (Filt[k_lst][k_serie] == 0)
            else:
                idx_val = slice(len(Mg[k_lst][k_serie]))
               
            mg = norm.cdf(Mg[k_lst][k_serie][idx_val]) 
            ts = t[k_lst][k_serie][idx_val]
            hs = [ti.hour for ti in ts]
            
            #print(hs)
            
            archis_lentes = [ archi.archi_lente(nombre, hi) for hi in hs ] 

            m_lente = []
            #print(mg)
            for k in range(len(mg)):
                h = hs[k]               
                archi_lente = archis_lentes[k]
                lente = lentes[archi_lente]
                kidx_lente = int((mg[k]) * len(lente) - 0.5)
                m_lente.append(lente[kidx_lente])
            
            Mr[k_lst][k_serie][idx_val] = m_lente   

    print('DesGaussianización Finalizada')
    return Mr
    
    
    
    