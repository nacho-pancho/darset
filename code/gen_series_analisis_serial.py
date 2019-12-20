# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:19:25 2019

@author: fpalacio
"""

import archivos
import numpy as np
import math



def gen_series_analisis_serial(parque1, parque2, nom_series_p1, nom_series_p2):
   
    M1,F1,nom1,t1 = parque1.exportar_medidas()
    M2,F2,nom2,t2 = parque2.exportar_medidas()
    
    nom_series_tot = nom_series_p1 + nom_series_p2

    M1_m = np.ma.array(M1, mask=F1.astype(int) , fill_value=-999999).filled()
    M2_m = np.ma.array(M2, mask=F2.astype(int), fill_value=-999999).filled()

    dt = t1[1] - t1[0]    
    tmin, tmax = max(t1[0],t2[0]), min(t1[-1],t2[-1])
    i_tmin1 = round((tmin - t1[0])/dt)
    i_tmin2 = round((tmin - t2[0])/dt)
    
    i_tmax1 = round((tmax - t1[0])/dt)
    i_tmax2 = round((tmax - t2[0])/dt)

    NDatos = i_tmax2 - i_tmin2 + 1
    
    M_tot_m = np.concatenate((M1_m[i_tmin1:i_tmax1,:],M2_m[i_tmin2:i_tmax2,:]),axis = 1)   
    
    archi = archivos.path(parque1.id) + 'archi_AS'
   
    f = open(archi,"w+")
    
    NSeries = len(nom_series_p1) + len(nom_series_p2)
    
    
    f.write(str(NSeries) +  '\n')
    f.write(str(tmin.year) + '\t' + str(tmin.month)  + '\t' + str(tmin.day) + \
         '\t' + str(tmin.hour) + '\t' + str(tmin.minute) + '\t' + str(tmin.second) + '\n')
    
    
    muestreo_hs = math.trunc(dt.total_seconds() + 0.01)/3600
    
    f.write(str(muestreo_hs) +  '\n')
    f.write(str(NDatos) +  '\n')
    f.write(str(1) +  '\n')
    
    
    str_noms = ''
    for i in range(len(nom_series_tot)):
        str_noms = str_noms + '\t' + nom_series_tot[i]        
    
    f.write( ' \t' + str_noms + '\n')
    
    for kdato in range(NDatos):
        f.write( str(kdato) +  '\t')
        for kcol in range(NSeries):
            f.write( str(M_tot_m[kdato,kcol]) +  '\t')        
            
        f.write('\n')
                 
    f.close()