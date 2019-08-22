# -*- coding: utf-8 -*-
"""
En este archivo se implementan lo distintos filtros, es decir,
los distintos métodos para detectar que una medida dada es anómala
o no.

Created on Thu May  2 14:55:56 2019

@author: fpalacio
"""
##############################################################################

import numpy
import datos as d
import numpy as np
import scipy.stats as r
import math as m

##############################################################################

def str_to_tipo(s):
    if s.find('Var(vel)') != -1:
        return None
    elif s.find('vel') != -1:
        return 'vel'
    elif s.find('dir') != -1:
        return 'dir'
    elif s.find('pot') != -1:
        return 'pot'
    elif s.find('tem') != -1:
        return 'tem'
    elif s.find('pre') != -1:
        return 'pre'
    elif s.find('hum') != -1:
        return 'hum'
    elif s.find('cgm') != -1:
        return 'cgm'    
    elif s.find('dis') != -1:
        return 'dis'
    elif s.find('corr_sp') != -1:
        return 'corr_sp'
    else:
        return None

##############################################################################
        
def min_max(tipo,PotAut):
    if tipo == 'vel':
        return [0,30]
    elif tipo == 'dir':
        return [0,360]
    elif tipo == 'pot':
        return [0,PotAut]
    elif tipo == 'tem':
        return [0,40]
    elif tipo == 'pre':
        return [0,1800]
    elif tipo == 'hum':
        return [0,100]
    elif tipo == 'cgm':
        return [0,PotAut]
    elif tipo == 'dis':
        return [0,1]
    elif tipo == 'corr_sp':
        return [-1,1]    

##############################################################################

def Nrep(tipo):
    if tipo == 'cgm' or tipo == 'dis':
        return None
    elif tipo == 'pot':
        return 100
    else:
        return 3

##############################################################################

def filtrar_rango(v,min_v,max_v):
    filtro = numpy.zeros(len(v), dtype=bool)
    for i in range(len(v)):
        if v[i]>max_v or v[i]<min_v:
            filtro[i] = True    
    return filtro 
        
##############################################################################

def filtrar_rep(v,filtro_huecos,nRep):
 
    filtro = numpy.zeros(len(v), dtype=bool)
    if nRep is None:
        return filtro
    
    k1 = 1
    cnt_total = 0
    cnt = 0
    buscando = True

    while buscando and (k1 < len(filtro_huecos)):
        if filtro_huecos[k1]:
            k1 = k1 + 1
        else:
            buscando = False
    
    if buscando:
        return filtro

    vant = v[k1]
    k = k1 + 1
    while k < len(v):
        if ((not filtro_huecos[k]) and (v[k] == vant)):
            cnt = cnt + 1
            if cnt >= nRep:
                while k1 <= k:
                    filtro[k1] = True
                    k1 = k1 + 1
        else:
            vant = v[k]
            k1 = k
            if cnt >= nRep:
                cnt_total = cnt_total + cnt
            cnt = 0
        
        k = k + 1

    return filtro     

##############################################################################
    
def corr_medidas(x,y,NDatosCorr,NDatosDesf):

    
    if ((x.tipo == 'dir') and (y.tipo == 'dir')):
        flg_dir_dir = True
    else:
        flg_dir_dir = False    
    
    filtro_x = x.filtrada()
    filtro_y = y.filtrada() 
    
    dtini_x = x.tiempo[0]
    dtini_y = y.tiempo[0]
    
    dif_dtini = dtini_y - dtini_x
    
    N10min_des_yx = round((dif_dtini.total_seconds())/600)

    filtro_y_des = np.ones(len(x.muestras),dtype=bool)

    y_m_des = np.zeros(len(x.muestras))
    
    for k in range(len(x.muestras)):
        if ((k-N10min_des_yx - NDatosDesf >= 0) and (k-N10min_des_yx - NDatosDesf < len(y.muestras))):
            y_m_des[k] = y.muestras[k - N10min_des_yx - NDatosDesf]
            filtro_y_des[k] = filtro_y[k - N10min_des_yx - NDatosDesf]
        else:
            y_m_des[k] = -99999999
            filtro_y_des[k] = 1
            
    filtro_total = filtro_x | filtro_y_des
           
    idx_mask = np.where(filtro_total < 1)
    
    x_m = x.muestras
    y_m = y_m_des
    
    
    if not flg_dir_dir:
        x_m_mask = x_m[idx_mask]
        y_m_mask = y_m[idx_mask]
        
        x_m_mask_u = r.rankdata(x_m_mask, "average")
        y_m_mask_u = r.rankdata(y_m_mask, "average")       
        
        x_m_mask_u = x_m_mask_u / np.max(x_m_mask_u)
        y_m_mask_u = y_m_mask_u / np.max(y_m_mask_u)
               
        x_m_u = np.zeros(len(x_m))
        y_m_u = np.zeros(len(y_m))
        
        x_m_u [idx_mask] = x_m_mask_u 
        y_m_u [idx_mask] = y_m_mask_u

                 
    idx_buff = np.zeros(NDatosCorr,dtype=int)
    corr = np.zeros(len(x.muestras))
    k_idx_buff = 0
    k = 0

    cualquiera = list()
    decorr = False
    while (k < len(x_m)):
        if not filtro_total[k]:
            if k_idx_buff < NDatosCorr:
                idx_buff[k_idx_buff] = k
                k_idx_buff = k_idx_buff + 1
                k = k + 1
                continue
            else:
                idx_buff_aux = idx_buff
                idx_buff[1:NDatosCorr-1] = idx_buff_aux[0:NDatosCorr-2]
                idx_buff[0] = k

            if flg_dir_dir:
                a1 = x_m[idx_buff]
                a2 = y_m[idx_buff]
                
                dif_ang_deg = np.add(a1,-a2)
                dif_ang_deg = np.abs(dif_ang_deg)

                idx_mayor180 = np.where(dif_ang_deg > 180)
                dif_ang_deg[idx_mayor180] = 360 - dif_ang_deg[idx_mayor180] 

                dang = np.mean(dif_ang_deg)/180
                corr[k] = 1 - dang
            else:     
                corr[k] = np.dot(x_m_u[idx_buff], y_m_u [idx_buff]) / \
                    (np.linalg.norm(x_m_u[idx_buff]) * np.linalg.norm(y_m_u[idx_buff]))
                if corr[k] < 0.7:
                    if not decorr:
                        cualquiera.append(k)
                        decorr = True
                else:
                    decorr = False
        else:
            corr[k] = corr[k-1]
            #print('(filtrado)')
        k = k + 1
    #print(cualquiera)
    #print('Episodios:',len(cualquiera))
    
    idx_datos_validos = np.where(filtro_total < 1) 
    corr_prom = corr[idx_datos_validos].mean()    
    print ('NDatosDesf: ',NDatosDesf,', corr = ',corr_prom)
    
    return d.Medida(corr,x.tiempo,'corr','corr_' + x.tipo + '_' + y.tipo,0.95,1.0,0),corr_prom
    

#tipo=str_to_tipo('vel')
#tipo2=str_to_tipo('dirgasdfgsdfg')
        