# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:04:47 2019

@author: fpalacio
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datos as d

def plotMedidas(medidas,plt_filtros,ran,ax = None,guardar=False,nidCentral=0):

    #plt.close('all')    
    # grafico datos 




    dt = list( medidas.tiempo[i] for i in ran )
    muestras = medidas.muestras[ran]
    
    if ax == None:
        fig, ax = plt.subplots()

    ax.plot_date(dt, muestras)
   
    plt.show()
    
    if guardar:
        archi ='../data/modelado_ro/c5/SMEC'
        plt.savefig(archi,dpi=150)

    
    return  ax   
    
    '''
    plt.close('all')    
    # grafico datos 
    dt = list( medida.tiempo[i] for i in ran )
    df = pd.DataFrame(medida.muestras[ran], index=dt,columns=[medida.nombre])
    
    if ax == None:
        ax = df.plot(figsize=(16, 6), grid=True)
    else:
        df.plot(ax=ax2)
    '''