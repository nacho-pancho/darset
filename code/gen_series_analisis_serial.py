# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:19:25 2019

@author: fpalacio
"""

import archivos
import numpy as np
import math

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def gen_series_analisis_serial(parque1, parque2, nom_series_p1, nom_series_p2, 
                               guardar = False):
   
    M1,F1,nom1,t1 = parque1.exportar_medidas()
    M2,F2,nom2,t2 = parque2.exportar_medidas()
    
    M1_ = np.zeros((M1.shape[0],len(nom_series_p1)))
    F1_ = np.zeros((F1.shape[0],len(nom_series_p1)))
    icol = 0
    for i in range(len(nom_series_p1)):
         x_col = nom1.index(nom_series_p1[i])
         M1_[:,icol] = M1[:,x_col]
         F1_[:,icol] = F1[:,x_col]
         icol = icol + 1
         
    M2_ = np.zeros((M2.shape[0], len(nom_series_p2)))
    F2_ = np.zeros((F2.shape[0], len(nom_series_p2)))
    icol = 0
    for i in range(len(nom_series_p2)):
         x_col = nom2.index(nom_series_p2[i])
         M2_[:,icol] = M2[:,x_col]
         F2_[:,icol] = F2[:,x_col]
         icol = icol + 1    

    nom_series_p1 = [s + '_' + str(parque1.id) for s in nom_series_p1]
    nom_series_p2 = [s + '_' + str(parque2.id) for s in nom_series_p2]
    nom_series_tot = nom_series_p1 + nom_series_p2

    
    M1_m = np.ma.array(M1_, mask=F1_.astype(int) , fill_value=-999999).filled()
    M2_m = np.ma.array(M2_, mask=F2_.astype(int), fill_value=-999999).filled()

    dt = t1[1] - t1[0]    
    tmin, tmax = max(t1[0],t2[0]), min(t1[-1],t2[-1])
    i_tmin1 = round((tmin - t1[0])/dt)
    i_tmin2 = round((tmin - t2[0])/dt)
    
    i_tmax1 = round((tmax - t1[0])/dt)
    i_tmax2 = round((tmax - t2[0])/dt)

    
    
    t_tot = t1[i_tmin1 : (i_tmax1+1)]
    
    M_tot_m = np.concatenate((M1_m[i_tmin1:i_tmax1+1,:],M2_m[i_tmin2:i_tmax2+1,:]),axis = 1)   

    if guardar:
        
        NDatos = i_tmax2 - i_tmin2 + 1
        
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
        
        f.write(str_noms + '\n')
        
        for kdato in range(NDatos):
            f.write( str(kdato) +  '\t')
            for kcol in range(NSeries):
                f.write( str(M_tot_m[kdato,kcol]) +  '\t')        
                
            f.write('\n')
                     
        f.close()
    
    
    return t_tot, M_tot_m, nom_series_tot