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
import pickle
import gzip

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
def NMuestras10minEntreDts(dt1,dt2):
    dif_dtini = dt2 - dt1
    return round((dif_dtini.total_seconds() + 1)/600)

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

def fechaInitoDateTimeN(dt_ini,Nmuestras10min):
    dt = []
    for k in range(Nmuestras10min):
        dt_k = dt_ini + k * datetime.timedelta(seconds=10*60)
        dt.append(dt_k)
    return dt

##############################################################################
    
def dt_to_dt10min(dt):
    dtdia = datetime.datetime(dt.year, dt.month, dt.day)
    dt10min = NMuestras10minEntreDts(dtdia,dt)*datetime.timedelta(seconds=10*60)
    dt10min = dt10min + dtdia
    return dt10min
#
#
# whatsapp BROU 092 001996
##############################################################################

def archiPICKLE(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'.pkl.gz'

##############################################################################

def archiFILTROS(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_filtros.pkl.gz'

##############################################################################

def archiSCADA(ncentral):
    return os.path.join(RUTA_DATOS,f'c{ncentral}/c{ncentral}_series10min.sas')

##############################################################################

def archiGEN(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_series10minGen.sas'

##############################################################################

def archiPRONOS(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_series60min_pronos.sas'

##############################################################################

def archiSMEC(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) + '/medidasSMEC.txt'

##############################################################################

def path(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) + '/'

##############################################################################
    
def leerCampo(file):
    line = file.readline().strip()
    cols = line.split()
    print('Campo',cols[1:],'Valor',cols[0])
    return cols[0]    

##############################################################################


def leerArchi(nidCentral,tipoArchi):    
    if tipoArchi == 'scada':
        archi = archiSCADA(nidCentral)
    elif tipoArchi == 'gen':
        archi = archiGEN(nidCentral)
    else:
        print(f"ERROR: tipo de archivo desconocido")
        return None

    print(f"LEYENDO ARCHIVO {tipoArchi} DE CENTRAL {nidCentral}: {archi}")

    if not os.path.exists(archi):
        print("ERROR: archivo no existente.")
        return None

    f = open(archi, 'r')
    print('LEYENDO ENCABEZADO:')
    line=f.readline().strip()
    cols = line.split()
    nSeries = int(cols[0])
    print('\tnum de series',nSeries)
    
    line=f.readline().strip()
    cols = line.split()
    meteo_utm_zona = cols[0]
    
    line=f.readline().strip()
    cols = line.split()
    meteo_utm_huso = int(cols[0])  
    
    line=f.readline().strip()
    cols = line.split()
    meteo_utm_xe = float(cols[0])

    line=f.readline().strip()
    cols = line.split()
    meteo_utm_yn = float(cols[0])
    print('\tzona horaria:',meteo_utm_zona,meteo_utm_huso,meteo_utm_xe,meteo_utm_yn)
    
    line=f.readline().strip()
    cols = line.split()
    ident = cols[0]

    ubicacion = datos.Ubicacion(meteo_utm_zona,meteo_utm_huso,meteo_utm_xe,meteo_utm_yn,ident)
    
    line=f.readline().strip()
    cols = line.split()
    PAutorizada = float(cols[0])
    print('\tpotencia autorizada:',PAutorizada)
    
    line=f.readline().strip()
    tipos = line.split()
    seg = np.arange(1,nSeries+1,1,dtype=np.int)
    #tipos = tipos[1:]
    print('\ttipos',tipos)    
    f.close() 
    
    print('LEYENDO DATOS')
    # Leo etiquetas de tiempo comunes a todos los datos
    data=np.loadtxt(archi,skiprows=8)
    dt_num=data[:,0]
    tiempo=fechaNumtoDateTime(dt_num)
    #
    # verificamos que no haya fechas repetidas
    #
    print('\tVerificando fechas repetidas')
    dt = list()
    for i in range(len(tiempo)-1):
        dt.append(tiempo[i+1]-tiempo[i])
    dtmin,dtmed,dtmax = np.min(dt),np.median(dt),np.max(dt)
    print(f"\tdt: min{dtmin} med={dtmed} max={dtmax}")
    dt.append(dt[-1])
    dtposta = datetime.timedelta(minutes=10)
    dtcero = datetime.timedelta(0)
    if dtmin == dtcero: # 
        trep = tiempo[dt == dtcero]
        print(f"ERROR: tiempos repetidos {trep}")
        return None
        exit
    elif dtmax > 1.1*dtposta:
        print(f"ERROR: tiempos faltantes!")
        return None
        exit
    
    print('\tconvirtiendo etiquetas de  tiempo a DateTime')
    dtini_10min = dt_to_dt10min(tiempo[0])
    tiempo = fechaInitoDateTimeN(dtini_10min,len(tiempo))
    
    print('\tLeyendo medidas')
    pot = None
    cgm = None
    medidas = []
    for i in range(nSeries):
        tipoDato = filtros.str_to_tipo(tipos[i])
        if tipoDato is None:
            continue
        meds = data[:,i+1]
        nombre = tipoDato + ident
        minmax = filtros.min_max(tipoDato,PAutorizada)
        nrep = filtros.Nrep(tipoDato)
        
        med = datos.Medida(tipoArchi,meds,tiempo,tipoDato,nombre,minmax[0],minmax[1],nrep)
        
        if tipoDato != 'pot' and tipoDato != 'cgm' and tipoDato != 'dis':
            medidas.append(med)
        elif tipoDato == 'pot':
            pot=copy.copy(med) 
        elif tipoDato == 'cgm':
            cgm=copy.copy(med) 
        elif tipoDato == 'dis':
            dis=copy.copy(med) 
    
    print('CREANDO MEDIDOR')
    Medidor = datos.Medidor(ident,medidas,ubicacion)
    print('CREANDO PARQUE')
    parque = datos.Parque(nidCentral,Medidor,cgm,pot)
    print('LECTURA TERMINADA\n')
    return parque

##############################################################################

def leerArchiSMEC(nidCentral):

    archi_SMEC = archiSMEC(nidCentral)

    if not os.path.exists(archi_SMEC):
        return None,None
        exit        

    print(f"Leyendo archivo SMEC  para la central {nidCentral}")

    # Leo muestras (todas las celdas tienen que tener un valor)
    f = open(archi_SMEC, 'r')
    line=f.readline()
    line=f.readline()    
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split()[1:-1])
        
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
    cols = line.split()    
    dtini_str = cols[0]
    dtini = datetime.datetime.strptime(dtini_str, '%d/%m/%Y') 

    delta_30min = datetime.timedelta(minutes=30) # sumo 30 min para que este en fase con SCADA
    dt_ini_corr = dtini + delta_30min
    dt_15min = fechaInitoDateTime(dt_ini_corr,ndias,15) 
    
    muestras10min = signal.resample_poly(muestras15min,up=15,down=10)
    dt_10min = fechaInitoDateTime(dt_ini_corr,ndias,10)   

    tipoDato = 'pot'
    minmax = filtros.min_max(tipoDato,50)
    nrep = filtros.Nrep(tipoDato)
  
    med_10min = datos.Medida('smec',muestras10min,dt_10min,'pot','potSMEC10m',minmax[0],minmax[1],nrep)
    med_15min = datos.Medida('smec',muestras15min,dt_15min,'pot','potSMEC15m',minmax[0],minmax[1],nrep)

    return med_10min, med_15min       

##############################################################################

def leerArchiPRONOS(nidCentral,muestreo_mins):    
    archi_pronos = archiPRONOS(nidCentral)       

    if not os.path.exists(archi_pronos):
        print(f"AVISO: no hay pronósticos para esta central. Archivo {archi_pronos} no encontrado.")
        return None
        exit

    print(f"Leyendo archivo de pronósticos para la central {nidCentral}: {archi_pronos}")
    
    f = open(archi_pronos, 'r')
    
    # Leo datos de las estaciones
    
    line=f.readline()
    print(line)
    cols = line.split()
    nSeries = int(cols[0])
    
    line=f.readline()
    cols = line.split()
    meteo_utm_zona = cols[0]
    
    line=f.readline()
    cols = line.split()
    meteo_utm_huso = int(cols[0])    
    
    line=f.readline()
    cols = line.split()
    meteo_utm_xe = float(cols[0])

    line=f.readline()
    cols = line.split()
    meteo_utm_yn = float(cols[0])
    
    line=f.readline()
    cols = line.split()
    ident = cols[0]

    ubicacion = datos.Ubicacion(meteo_utm_zona,meteo_utm_huso,meteo_utm_xe,meteo_utm_yn,ident)
    
    line=f.readline()
    cols = line.split()
    PAutorizada = float(cols[0])

    line=f.readline()
    tipos = line.split()
    tipos = [ tipos[i] for i in range(nSeries)]
        
    data=np.loadtxt(archi_pronos,skiprows=8)

    line=f.readline()
    cols = line.split()   
    f.close() 
    dt_ini_str = cols[0]
    dt_ini = NumtoDateTime(float(dt_ini_str))

    # Leo medidas
    medidas = []
    for i in range(nSeries):

        tipoDato = filtros.str_to_tipo(tipos[i])
        if tipoDato is None:
            break
        meds = data[:,i+1]
        nombre = tipoDato + ident
        minmax = filtros.min_max(tipoDato,PAutorizada)
        nrep = filtros.Nrep(tipoDato)
        
        if muestreo_mins != 60:      
            if tipoDato == 'dir':
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
            else:            
                meds = signal.resample_poly(meds,up=60,down=10)        
            
        dt_10min = fechaInitoDateTimeN ( dt_ini, len(meds))

        med = datos.Medida('pronos',meds,dt_10min,tipoDato,nombre,minmax[0],minmax[1],nrep)
        med.desfasar(-18) # los pronósticos vienen con GMT 0, nosotros tenemos GMT -3
        medidas.append(med)

    Medidor = datos.Medidor(ident,medidas,ubicacion)
           
    return Medidor

##############################################################################
    
def leerArchivosCentral (nidCentral):
    #
    # si existe, cargamos el objeto guardado
    #
    archip = archiPICKLE(nidCentral)
    if os.path.exists(archip):
        print(f'INFO: cargando datos de parque de {archip}')
        return cargarCentral(nidCentral)
    #
    # si no, generamos todo desde 0 en base a los distintos archivos
    # de texto que componen la info de la central.
    #
    parque = leerArchi(nidCentral,'scada')
    
    parqueGen = leerArchi(nidCentral,'gen')
    if parqueGen is not None:
        parque.medidores[0].agregar_meds(parqueGen.medidores[0].medidas)
    else:
        print("AVISO: No hay archivo GEN para esta central.")
    
    med_10min, med_15min = leerArchiSMEC(nidCentral)
    if med_10min is not None:
        parque.pot_SMEC = med_10min
    else:
        print("AVISO: No hay archivo SMEC para esta central.")

    medidor_pronos10min = leerArchiPRONOS(nidCentral,10)    
    if medidor_pronos10min is not None:
        parque.medidores[0].agregar_meds(medidor_pronos10min.medidas)
    else:
        print("AVISO: No hay archivo PRONOS para esta central.")
    #
    # si existe, cargamos el objeto guardado
    #
    return parque
        
##############################################################################

def guardarCentral(parque):
    archi_central = archiPICKLE(parque.id)
    with gzip.open(archi_central,'wb') as output:
        pickle.dump(parque,output,pickle.HIGHEST_PROTOCOL)

##############################################################################

def cargarCentral(id):
    archi_central = archiPICKLE(id)
    with gzip.open(archi_central,'r') as input:
        parque = pickle.load(input)
        return parque
