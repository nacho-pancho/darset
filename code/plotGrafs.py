# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:04:47 2019

@author: fpalacio
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datos as d
import numpy as np

def plotMedidas(medidas,plt_filtros,fecha_ini,fecha_fin,ruta,guardarFig=False):

    plt.close('all')   
    fig, axes = plt.subplots(figsize=(75,20),nrows=2, ncols=1)    
    plt.rc('font', size=40)

    plt.grid()
    
    for k in range(len(medidas)):   
        df_meds = pd.DataFrame(medidas[k].muestras, index=medidas[k].tiempo,columns=[medidas[k].nombre])
          
        df_meds_filt = df_meds[(df_meds.index >= fecha_ini) & (df_meds.index <= fecha_fin)]
               
        df_meds_filt.plot(ax=axes[0], linewidth=10)
        
        filtros, nombres = medidas[k].filtrosAsInt()
        
        df_filt = pd.DataFrame(filtros,index=medidas[k].tiempo,columns=nombres)
 
        df_filt_filt = df_filt[(df_filt.index >= fecha_ini) & (df_filt.index <= fecha_fin)] 
        
        df_filt_filt.plot(ax=axes[1], linewidth=10)

    axes[0].grid(True)
    axes[1].grid(True)
    plt.show() 

           
    if guardarFig:
        nombreFig = ''
        for k in range(len(medidas)):
            nombreFig = nombreFig + '_' + medidas[k].nombre    
        archi = ruta + nombreFig
        fig.savefig(archi,dpi=150)
    
    return  None
    