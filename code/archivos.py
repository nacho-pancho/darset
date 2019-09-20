# -*- coding: utf-8 -*-
"""
Funciones para lear archivos de datos y cargarlos en las distintas
estructuras y clases que se definen en este paquete.

@author: fpalacio
"""
import datos
import filtros
import os
import numpy as np
import datetime
import copy
import math as m
import pandas as pd
import scipy.signal as signal

##############################################################################

RUTA_DATOS = '../data/'

##############################################################################

def fechaNumtoDateTime(dt_num):
    dt = []
    for i in range(len(dt_num)):
        num= dt_num[i]
        dt_datetime=NumtoDateTime(num)
        dt.append(dt_datetime)
    return dt

##############################################################################
    

def NumtoDateTime(num):
    dtini=datetime.datetime(1900, 1,1)
    dt_datetime=dtini + datetime.timedelta(seconds=((num-2)*24*3600))
    return dt_datetime

##############################################################################

def fechaInitoDateTime(dt_ini,ndias,cant_min):
    dt = []
    muestras_por_dia = m.trunc((60*24) / cant_min + 0.00001)
    for dia in range(ndias):
        for muestra in range(muestras_por_dia):
            seg = dia*24*3600 + muestra * cant_min * 60
            dt_datetime=dt_ini + datetime.timedelta(seconds=seg)
            dt.append(dt_datetime)
    return dt

##############################################################################

def archiSCADA(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_series10min.sas'

##############################################################################
    

##############################################################################

def archiGEN(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_series10minGen.sas'

##############################################################################

def archiPRONOS(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_series60min_pronos.txt'

##############################################################################

def archiSMEC(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) + '/medidasSMEC.txt'

##############################################################################

def path(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) + '/'

##############################################################################
    

def leerArchi(nidCentral,tipoArchi):    
    print(f"Leyendo archivo de central {nidCentral}")

    if tipoArchi == 'scada':
        archi = archiSCADA(nidCentral)
    elif tipoArchi == 'gen':
        archi = archiGEN(nidCentral)
    else:
        return None
    
    f = open(archi, 'r')
    
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
    data=np.loadtxt(archi,skiprows=8)
    dt_num=data[:,0]
    tiempo=fechaNumtoDateTime(dt_num)
    #
    # verificamos que no haya fechas repetidas
    #
    
    dt = list()
    for i in range(len(tiempo)-1):
        dt.append(tiempo[i+1]-tiempo[i])
    dtmin,dtmed,dtmax = np.min(dt),np.median(dt),np.max(dt)
    print(f"dt: min{dtmin} med={dtmed} max={dtmax}")
    #dt.append(datetime.timedelta(dt[-1]))
    dtposta = datetime.timedelta(minutes=10)
    dtcero = datetime.timedelta(0)
    if dtmin == dtcero: # 
        trep = tiempo[dt == dtcero]
        print(f"ERROR: tiempos repetidos {trep}")
    elif np.abs(dtmax) > 1.001*dtposta:
        print(f"ERROR: tiempos faltantes!")
    
    
    # Leo medidas

    medidas = []
    for i in range(nSeries):

        tipoDato = filtros.str_to_tipo(tipos[i])
        if tipoDato == None:
            break
        meds = data[:,i+1]
    
        
        nombre = tipoDato + ident
        minmax = filtros.min_max(tipoDato,PAutorizada)
        nrep = filtros.Nrep(tipoDato)
        
        med = datos.Medida(meds,tiempo,tipoDato,nombre,minmax[0],minmax[1],nrep,datetime.timedelta(minutes=10))
 
        dtini_filt = datetime.datetime.strptime('2015-10-01 00:00:0.0', '%Y-%m-%d %H:%M:%S.%f')
        
        dtfin_filt = datetime.datetime.strptime('2015-10-02 00:09:0.0', '%Y-%m-%d %H:%M:%S.%f')
       
        
        med_filt = med.getmuestras_dt(dtini_filt,dtfin_filt)
       
        if (tipoDato != 'pot' and tipoDato != 'cgm' and tipoDato != 'dis'):
            medidas.append(med)
        elif (tipoDato == 'pot'):
            pot=copy.copy(med) 
        elif (tipoDato == 'cgm'):
            cgm=copy.copy(med) 
        elif (tipoDato == 'dis'):
            dis=copy.copy(med) 

    Medidor = datos.Medidor(ident,medidas,ubicacion)
    
    parque = datos.Parque(Medidor,cgm,pot,dis)
    
    return parque

##############################################################################

def leerArchiSMEC(nidCentral):
    print(f"Leyendo archivo SMEC  para la central {nidCentral}")
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
    muestras15min = muestras_mat.flatten().astype(float)*4

    # Leo fecha inicial
    f = open(archi_SMEC, 'r')
    line=f.readline()
    line=f.readline()
    line=f.readline()
    f.close() 
    cols = line.split('\t')    
    dtini_str = cols[0]
    dtini = datetime.datetime.strptime(dtini_str, '%d/%m/%Y') 

    delta_15min = datetime.timedelta(minutes=30) # sumo 30 min para que este en fase con SCADA
    dt_ini_corr = dtini + delta_15min
    dt_15min = fechaInitoDateTime(dt_ini_corr,ndias,15) 
    
    muestras10min = signal.resample_poly(muestras15min,up=15,down=10)
    dt_10min = fechaInitoDateTime(dt_ini_corr,ndias,10)   

    tipoDato = 'pot'
    minmax = filtros.min_max(tipoDato,50)
    nrep = filtros.Nrep(tipoDato)
  
    med_10min = datos.Medida(muestras10min,dt_10min,'pot','potSMEC10m',minmax[0],minmax[1],nrep)
    med_15min = datos.Medida(muestras15min,dt_15min,'pot','potSMEC15m',minmax[0],minmax[1],nrep)

    return med_10min, med_15min       

##############################################################################

def leerArchiPRONOS(nidCentral,muestreo_mins):    
    print(f"Leyendo archivo de pronósticos para la central {nidCentral}")
    archi_pronos = archiPRONOS(nidCentral)       
    
    f = open(archi_pronos, 'r')
    
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
        

    # Leo etiquetas de tiempo comunes a todos los datos
    data=np.loadtxt(archi_pronos,skiprows=8)

    line=f.readline()
    cols = line.split('\t')   
    f.close() 
    dtini_str = cols[0]
    dtini = NumtoDateTime(int(dtini_str))

    delta_30min = datetime.timedelta(minutes=30) # sumo 30 min para que este en fase con SCADA
    dt_ini_corr = dtini #+ delta_30min
    dt_10min = fechaInitoDateTime(dt_ini_corr,m.trunc(len(data[:,1])/24),muestreo_mins)       
    
    # Leo medidas
    medidas = []
    for i in range(nSeries):

        tipoDato = filtros.str_to_tipo(tipos[i])
        if tipoDato == None:
            break
        meds = data[:,i+1]
        nombre = tipoDato + ident
        minmax = filtros.min_max(tipoDato,PAutorizada)
        nrep = filtros.Nrep(tipoDato)
        
        if muestreo_mins != 60:      
            if tipoDato == 'vel':
                meds = signal.resample_poly(meds,up=60,down=10)
            else:            
                meds_sin = [m.sin(m.radians(k)) for k in meds ]
                meds_cos = [m.cos(m.radians(k)) for k in meds ]
                
                meds_sin_m = signal.resample_poly(meds_sin,up=60,down=muestreo_mins)
                meds_cos_m = signal.resample_poly(meds_cos,up=60,down=muestreo_mins)
                            
                meds_m = [m.atan2(s,c) for s,c in zip(meds_sin_m,meds_cos_m)]
                meds_m = [m.degrees(k) for k in meds_m]
                for k in range(len(meds_m)):
                    if meds_m[k] < 0 :
                        meds_m[k] = meds_m[k] + 360
                
                meds = np.asarray(meds_m) 
            
        med = datos.Medida(meds,dt_10min,tipoDato,nombre,minmax[0],minmax[1],nrep,datetime.timedelta(minutes=muestreo_mins))
        medidas.append(med)

    Medidor = datos.Medidor(ident,medidas,ubicacion)
           
    return Medidor

##############################################################################
