# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:49:07 2019

@author: fpalacio
"""
import datos
import filtros
import os
import numpy as np
import datetime
import copy
import math as m
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import scipy.signal as signal

RUTA_DATOS = '../data/'


def fechaNumtoDateTime(dt_num):
    dtini=datetime.datetime(1900, 1,1)
    dt = []
    for i in range(len(dt_num)):
        num= dt_num[i]
        dt_datetime=dtini + datetime.timedelta(seconds=((num-2)*24*3600))
        dt.append(dt_datetime)
    return dt

def fechaInitoDateTime(dt_ini,ndias,cant_min):
    dt = []
    muestras_por_dia = m.trunc((60*24) / cant_min + 0.00001)
    for dia in range(ndias):
        for muestra in range(muestras_por_dia):
            seg = dia*24*3600 + muestra * cant_min * 60
            dt_datetime=dt_ini + datetime.timedelta(seconds=seg)
            dt.append(dt_datetime)
    return dt

def archiSCADA(ncentral):
    return RUTA_DATOS +'modelado_ro/c'+ str(ncentral) +'/c'+str(ncentral)+'_series10min.sas'

def archiSMEC(ncentral):
    return RUTA_DATOS +'modelado_ro/c'+ str(ncentral) + '/medidasSMEC.txt'

def path(ncentral):
    return RUTA_DATOS +'modelado_ro/c'+ str(ncentral) + '/'



def leerArchiSCADA(nidCentral):    
    
    nidCentral=5
    archi_scada = archiSCADA(nidCentral)       
    
    f = open(archi_scada, 'r')
    
    # Leo datos de las estaciones
    
    line=f.readline()
    cols = line.split('\t')
    nSeries = int(cols[0])
    
    line=f.readline()
    cols = line.split('\t')
    meteo_utm_zona = cols[0]
    
    line=f.readline()
    cols = line.split('\t')
    meteo_utm_huso = int(cols[0])    
    
    line=f.readline()
    cols = line.split('\t')
    meteo_utm_xe = float(cols[0])

    line=f.readline()
    cols = line.split('\t')
    meteo_utm_yn = float(cols[0])
    

    
    line=f.readline()
    cols = line.split('\t')
    ident = cols[0]

    ubicacion = datos.Ubicacion(meteo_utm_zona,meteo_utm_huso,meteo_utm_xe,meteo_utm_yn,ident)
    
    line=f.readline()
    cols = line.split('\t')
    PAutorizada = float(cols[0])

    line=f.readline()
    tipos = line.split('\t')
    seg = np.arange(1,nSeries+1,1,dtype=np.int)
    tipos = [ tipos[i] for i in seg]
    
    f.close() 

    # Leo etiquetas de tiempo comunes a todos los datos
    data=np.loadtxt(archi_scada,skiprows=8)
    dt_num=data[:,0]
    tiempo=fechaNumtoDateTime(dt_num)
    
    # Leo medidas

    medidas = []
    for i in range(nSeries):

        tipoDato = filtros.str_to_tipo(tipos[i])
        if tipoDato == None:
            break
        meds = data[:,i+1]
        nombre = tipoDato +'_'+ ident
        minmax = filtros.min_max(tipoDato,PAutorizada)
        nrep = filtros.Nrep(tipoDato)
        
        med = datos.Medida(meds,tipoDato,nombre,minmax[0],minmax[1],nrep)
        
        if (tipoDato != 'pot' and tipoDato != 'cgm' and tipoDato != 'dis'):
            medidas.append(med)
        elif (tipoDato == 'pot'):
            pot=copy.copy(med) 
        elif (tipoDato == 'cgm'):
            cgm=copy.copy(med) 
        elif (tipoDato == 'dis'):
            dis=copy.copy(med) 
            

    
    Medidor = datos.Medidor(tiempo,medidas,ubicacion)
    
    parque = datos.Parque(Medidor,cgm,pot,dis)
    
    
    return parque


def leerArchiSMEC(nidCentral):
    archi_SMEC = archiSMEC(nidCentral)

    # Leo muestras (todas las celdas tienen que tener un valor)
    f = open(archi_SMEC, 'r')
    line=f.readline()
    line=f.readline()    
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split('\t')[1:-1])
        
    f.close()    

    muestras_mat = np.array(result)
    ndias,n15min = muestras_mat.shape
    muestras15min = muestras_mat.flatten().astype(float)

    # Leo fecha inicial
    f = open(archi_SMEC, 'r')
    line=f.readline()
    line=f.readline()
    line=f.readline()
    f.close() 
    cols = line.split('\t')    
    dtini_str = cols[0]
    dtini = datetime.datetime.strptime(dtini_str, '%d/%m/%Y') 

    dt_15min = fechaInitoDateTime(dtini,ndias,15)
    
    muestras10min = signal.resample_poly(muestras15min,up=15,down=10)
    dt_10min = fechaInitoDateTime(dtini,ndias,10)   
    
    ran = np.arange(70494,70700,dtype=int)
    ran2 = np.arange(m.trunc(70494*15/10),m.trunc(70700*15/10),dtype=int)
    
    # grafico datos 
    dt_15min = list( dt_15min[i] for i in ran )
    dt_10min = list( dt_10min[i] for i in ran2 )
    df = pd.DataFrame(muestras15min[ran], index=dt_15min)
    
    ax = df.plot(figsize=(16, 6), marker='o')

    df2 = pd.DataFrame(muestras10min[ran2], index=dt_10min)
    df2.plot(ax=ax)
    plt.show()

    archi = path(nidCentral) + "SMEC"  
    plt.savefig(archi,dpi=150)         

    
    '''
    dt_15min_plt = mdates.date2num(dt_15min)
#   plt.plot(dt_15min_plt, muestras)

    plt.close('all')
    plt.figure(1,figsize=(150,20))
    plt.grid(True)


    x = dt_15min_plt[70494:70700]
    y = muestras[70494:70700]
    
    dates = mdates.date2num(dt_15min)
    plt.plot_date(dates, y)
    
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(30))
    plt.plot(x,y)
    plt.gcf().autofmt_xdate()
    plt.show()

    plt.plot_date(dt_15min[70494:70700], muestras[70494:70700])
    plt.show()
    archi = path(nidCentral) + "SMEC"
    plt.savefig(archi,dpi=150)
    '''

    a=1

leerArchiSMEC(5) 
#parque = leerArchiSCADA(5)    
#♣fechaNumtoDateTime([42139])      