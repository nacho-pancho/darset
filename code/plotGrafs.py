# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:04:47 2019

@author: fpalacio
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datos as d

def plotMedidas(medidas,plt_filtros,fecha_ini,fecha_fin,guardarFig=False):

    plt.close('all') 

    ax = None  

    for k in range(len(medidas)):   
        df = pd.DataFrame(medidas[k].muestras, index=medidas[k].tiempo,columns=[medidas[k].nombre])
          
        df_filt = df[(df.index >= fecha_ini) & (df.index <= fecha_fin)]
        
        if ax == None:
            fig, ax = plt.subplots()
            plt.grid(True)
        
        df_filt.plot(ax=ax)
    
    plt.show() 
    
           
    if guardarFig:
        archi ='../data/modelado_ro/c5/fig.png'
        fig.savefig(archi,dpi=150)
    
    return  ax
    