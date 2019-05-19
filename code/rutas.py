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


def fechaNumtoDateTime(dt_num):
    dtini=datetime.datetime(1900, 1,1)
    dt = []
    for i in range(len(dt_num)):
        num= dt_num[i]
        dt_datetime=dtini + datetime.timedelta(seconds=((num-2)*24*3600))
        dt.append(dt_datetime)
    return dt

def archiSCADA(ncentral):
    pathIni = os.getcwd()
    os.chdir("..")
    path = os.getcwd()
    os.chdir(pathIni)
    return path +'\modelado_ro\c'+ str(ncentral) +'\c'+str(ncentral)+'_series10min.sas'


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


parque = leerArchiSCADA(5)    
#♣fechaNumtoDateTime([42139])      