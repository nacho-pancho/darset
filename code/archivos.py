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
import re


##############################################################################
'''
if sys.platform.startswith('linux'):
    # cuando trabajamos en linux
    RUTA_DATOS = '../data/'
else:
    # cuando trabajamos en win
    RUTA_DATOS = 'Y:/modelado_ro_RNN/'
'''    

RUTA_DATOS = '../data/'
TS_MIN = 10

##############################################################################

def fechaNumtoDateTime(dt_num):
    dt = []
    for i in range(len(dt_num)):
        num= dt_num[i]
        dt_datetime=NumtoDateTime(num)
        dt.append(dt_datetime)
    return dt

##############################################################################
def NMuestrasTSEntreDts(dt1,dt2):
    dif_dtini = dt2 - dt1
    return int(m.ceil(dif_dtini/datetime.timedelta(minutes=TS_MIN)))

##############################################################################
def NumtoDateTime(num):  
    dtini=datetime.datetime(1900, 1, 1)
    dt_datetime = dtini + datetime.timedelta(days=num-2)   
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

def fechaInitoDateTimeN(dt_ini,NmuestrasTS):
    dt = []
    for k in range(NmuestrasTS):
        dt_k = dt_ini + k * datetime.timedelta(minutes=TS_MIN)
        dt.append(dt_k)
    return dt

##############################################################################
    
def dt_to_dtTS(dt):
    dtdia = datetime.datetime(dt.year, dt.month, dt.day)

    dtTS = NMuestrasTSEntreDts(dtdia,dt)*datetime.timedelta(minutes=TS_MIN)
    dtTS = dtTS + dtdia
    return dtTS

##############################################################################

def archiPICKLE(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'.pkl.gz'

##############################################################################

def archiFILTROS(ncentral):
    return RUTA_DATOS +'/c'+ str(ncentral) +'/c'+str(ncentral)+'_filtros.pkl.gz'

##############################################################################

def archiSCADA(ncentral):
    #f = os.path.join(RUTA_DATOS,f'c{ncentral}/archivos/')
    
    cnid = 'c'+ str(ncentral) 
    
    return RUTA_DATOS + cnid + '/archivos/' + cnid + '_series10min.sas'


##############################################################################

def archi_ro_pendientes(ncentral):
    #f = os.path.join(RUTA_DATOS,f'c{ncentral}/archivos/ro_pendientes_{ncentral}.txt')
    
    cnid = 'c'+ str(ncentral) 
    
    return RUTA_DATOS + cnid + '/archivos/ro_pendientes_' + str(ncentral) + '.txt'

##############################################################################

def archiGEN(ncentral):
    #return os.path.join(RUTA_DATOS,f'c{ncentral}/archivos/c{ncentral}_series10minGen.sas')
    cnid = 'c'+ str(ncentral) 
    
    return RUTA_DATOS + cnid + '/archivos/' + cnid + '_series10minGen.sas'    

##############################################################################

def archiPRONOS(ncentral):
    #return os.path.join(RUTA_DATOS,f'c{ncentral}/archivos/c{ncentral}_series60min_pronos.sas')
    cnid = 'c'+ str(ncentral) 
    
    return RUTA_DATOS + cnid + '/archivos/' + cnid + '_series60min_pronos.sas'  
    
##############################################################################

def archiSMEC(ncentral):
    #return os.path.join(RUTA_DATOS,f'c{ncentral}/archivos/medidasSMEC.txt')
    cnid = 'c'+ str(ncentral) 
    return RUTA_DATOS + cnid + '/archivos/medidasSMEC.txt'
##############################################################################

def path_central(ncentral):
    carpeta_central = RUTA_DATOS +'c'+ str(ncentral) + '/'
    if not os.path.exists(carpeta_central):
        os.makedirs(carpeta_central)
    return carpeta_central

##############################################################################

def path_ro (nro_ro, carpeta_res):  
    carpeta_ro = carpeta_res + str(nro_ro)  + '/'
    if not os.path.exists(carpeta_ro):
        os.makedirs(carpeta_ro)
    return carpeta_ro

##############################################################################    

def path_carpeta_datos(ncentral):
    carpeta_central = path_central(ncentral)
    carpeta_datos = carpeta_central + 'datos/'
    if not os.path.exists(carpeta_datos):
        os.mkdir(carpeta_datos)
    return carpeta_datos

##############################################################################

def path_carpeta_resultados(nidcentral, tipo_calc, tipo_norm):
    carpeta_central = path_central(nidcentral)
    carpeta_res = carpeta_central + 'resultados_' + tipo_calc + '_' + tipo_norm \
    + '_' + str(TS_MIN) + '/'
    if not os.path.exists(carpeta_res):
        os.mkdir(carpeta_res)
    return carpeta_res
##############################################################################

def path_carpeta_lentes(carpeta_res):
    
    carpeta_lentes = carpeta_res + 'lentes/'
    if not os.path.exists(carpeta_lentes):
        os.mkdir(carpeta_lentes)
    return carpeta_lentes    

##############################################################################

def archi_lente(nombre, khora):
    archi = 'l_' + nombre + '_h' + str(khora) + '.npy' 
    return archi 
##############################################################################
    
def leerCampo(file):
    line = file.readline().strip()
    cols = line.split()
    return cols[0]    

##############################################################################


def leerArchi(nidCentral,tipoArchi):    
    
    global TS_MIN
    
    if tipoArchi == 'scada':
        archi = archiSCADA(nidCentral)
    elif tipoArchi == 'gen':
        archi = archiGEN(nidCentral)
    else:
        print('ERROR: tipo de archivo desconocido')
        return None

    print('LEYENDO ARCHIVO ' + tipoArchi + ' DE CENTRAL ' + str(nidCentral) + ':' +archi)

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
    
    line = f.readline().strip()
    cols = line.split()
    PAutorizada = float(cols[0])
    print('\tpotencia autorizada:',PAutorizada)

    line=f.readline().strip()
    cols = line.split()
    cant_molinos = int(cols[0])
    print('\tcantidad molinos:',cant_molinos)
    
    line=f.readline().strip()
    tipos = line.split()
    seg = np.arange(1,nSeries+1,1,dtype=np.int)
    print('\ttipos',tipos)    
    f.close() 
    
    print('LEYENDO DATOS')
    # Leo etiquetas de tiempo comunes a todos los datos
    data=np.loadtxt(archi,skiprows=9)
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
    #print(f"\tdt: min{dtmin} med={dtmed} max={dtmax}")
    dt.append(dt[-1])
    dtposta = datetime.timedelta(minutes=TS_MIN)
    dtcero = datetime.timedelta(0)
    if dtmin == dtcero: # 
        trep = tiempo[dt == dtcero]
        print('ERROR: tiempos repetidos ' + str(trep))
        return None
        exit
    elif dtmax > 1.1*dtposta:
        print('ERROR: tiempos faltantes!')
        return None
        exit
    
    #print('\tconvirtiendo etiquetas de  tiempo a DateTime')

    
    print('\tLeyendo medidas')
    pot = None
    cgm = None
    dis = None
    medidas = []
    for i in range(nSeries):
        tipoDato = filtros.str_to_tipo(tipos[i])
        if tipoDato is None:
            continue
        meds = data[:, i+1]
        
        nombre = tipoDato + ident + '_' + str(nidCentral)
        minmax = filtros.min_max(tipoDato,PAutorizada,cant_molinos)
        nrep = filtros.Nrep(tipoDato)
        
        # Creo medida 10min auxiliar para filtrar medidas con su muestreo original
        if TS_MIN > 10:            
            TS_MIN_old = TS_MIN
            TS_MIN = 10
            dtini_TS = dt_to_dtTS(tiempo[0])
            tiempo = fechaInitoDateTimeN(dtini_TS, len(meds))        
            med = datos.Medida(tipoArchi,meds,tiempo,tipoDato,nombre,minmax[0],minmax[1],nrep)
            meds[med.filtrada()==1] = -99999999        
            TS_MIN = TS_MIN_old

        
        if TS_MIN != 10:
            if tipoDato == 'dir':
                meds_sin = [m.sin(m.radians(k)) for k in meds]
                meds_cos = [m.cos(m.radians(k)) for k in meds]

                if TS_MIN == 60:
                    meds_sin_m = average(meds_sin, 6)
                    meds_cos_m = average(meds_cos, 6)
                else:                    
                    meds_sin_m = signal.resample_poly(meds_sin, up=10, down=TS_MIN)
                    meds_cos_m = signal.resample_poly(meds_cos, up=10, down=TS_MIN)

                '''
                meds_sin_m = signal.resample_poly(meds_sin, up=10, down=TS_MIN)
                meds_cos_m = signal.resample_poly(meds_cos, up=10, down=TS_MIN)
                '''
                
                meds_m = [m.atan2(s, c) for s, c in zip(meds_sin_m, meds_cos_m)]
                meds_m = [m.degrees(k) for k in meds_m]
                for k in range(len(meds_m)):
                    if meds_m[k] < 0:
                        meds_m[k] = meds_m[k] + 360

                meds = np.asarray(meds_m)
            else:
                # asumo que está función promedia cuando pasa de 10 a 60min
                if TS_MIN == 60:
                    meds = average(meds, 6)
                    #meds = np.mean(meds.reshape(-1, 6), 1)
                else:                    
                    meds = signal.resample_poly(meds, up=10, down=TS_MIN) 
                
                
        dtini_TS = dt_to_dtTS(tiempo[0])
        tiempo = fechaInitoDateTimeN(dtini_TS, len(meds))
        med = datos.Medida(tipoArchi,meds,tiempo,tipoDato,nombre,minmax[0],minmax[1],nrep)
        
        if tipoDato != 'pot' and tipoDato != 'cgm' and tipoDato != 'dis':
            medidas.append(med)
        elif tipoDato == 'pot':
            pot = med #copy.copy(med) 
        elif tipoDato == 'cgm':
            cgm = med #copy.copy(med)
        elif tipoDato == 'dis':
            dis = med #copy.copy(med) 

        if tipoDato == 'vel':
            vel = med
            
        if (tipoDato == 'dir') and (vel != None):
            
            velx, vely, vel3x, vel3y, vel3 = velxy_from_veldir(vel, med, ident, nidCentral)    
            medidas.append(velx)
            medidas.append(vely)
            medidas.append(vel3x)
            medidas.append(vel3y)
            medidas.append(vel3)
            
            cosdir, sindir = cosin_from_dir(med, ident, nidCentral)
            medidas.append(cosdir)
            medidas.append(sindir)            

    
    print('CREANDO MEDIDOR')
    Medidor = datos.Medidor(ident,medidas,ubicacion)
    print('CREANDO PARQUE')
    parque = datos.Parque(nidCentral,Medidor,cgm,pot,dis,PAutorizada,cant_molinos)
    print('LECTURA TERMINADA\n')
    return parque

##############################################################################

def leerArchiSMEC(nidCentral):

    archi_SMEC = archiSMEC(nidCentral)

    if not os.path.exists(archi_SMEC):
        return None
        exit        

    print('Leyendo archivo SMEC para la central' +  str(nidCentral))

    # Leo muestras (todas las celdas tienen que tener un valor)
    f = open(archi_SMEC, 'r')
    line=f.readline()
    line=f.readline()    
    lines=f.readlines()
    line_ini = lines[0].split()[1:-1]
    
    if len(line_ini) > 24:
      flg_15min = True
      up_ = 15
    else:
      flg_15min = False
      up_ = 60

    result=[]
    for x in lines:
        result.append(x.split()[1:-1]) 
    f.close()

    #print(result)   
    muestras_mat = np.array(result)
    #print(muestras_mat.shape)
    ndias, nmuestras = muestras_mat.shape
    
    muestras_flat = muestras_mat.flatten().astype(float)
    if flg_15min:
        muestras_flat = muestras_flat *  4    
    
    muestrasTS = signal.resample_poly(muestras_flat,up=up_,down=TS_MIN)

    # Leo fecha inicial
    f = open(archi_SMEC, 'r')
    line=f.readline()
    line=f.readline()
    line=f.readline()
    f.close() 
    cols = line.split()    
    dtini_str = cols[0]
    dtini = datetime.datetime.strptime(dtini_str, '%d/%m/%Y') 

    dt_ini_corr = dtini + datetime.timedelta(minutes=30) # sumo 30 min para que este en fase con SCADA
    dt_TS = fechaInitoDateTime(dt_ini_corr,ndias,TS_MIN)
    
    tipoDato = 'pot'
    minmax = filtros.min_max(tipoDato,50,100)
    nrep = filtros.Nrep(tipoDato)
    
    med_TS = datos.Medida('smec',muestrasTS,dt_TS,'pot','potSMEC',minmax[0],minmax[1],nrep)
    
    if (TS_MIN == 10):
        med_TS.desfasar(-1)
    
    return med_TS 

##############################################################################

def leerArchiPRONOS(nidCentral):    
    archi_pronos = archiPRONOS(nidCentral)       

    if not os.path.exists(archi_pronos):
        print('AVISO: no hay pronósticos para esta central. Archivo ' + archi_pronos + 'no encontrado.')
        return None
        exit

    print('Leyendo archivo de pronósticos para la central ' + str(nidCentral) + ':' + str(archi_pronos))
    
    f = open(archi_pronos, 'r')
    
    # Leo datos de las estaciones
    
    line=f.readline()
    #print(line)
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
    vel = None
    for i in range(nSeries):

        tipoDato = filtros.str_to_tipo(tipos[i])
        #print(tipoDato)
        if tipoDato is None:
            break
        meds = data[:,i+1]
        nombre = tipoDato + ident + '_' + str(nidCentral)
        minmax = filtros.min_max(tipoDato,PAutorizada,1000)
        nrep = filtros.Nrep(tipoDato)
        
        if TS_MIN != 60:      
            if tipoDato == 'dir':
                meds = resample_poly_dir(meds, 60, TS_MIN)
            else:            
                meds = signal.resample_poly(meds,up=60,down=TS_MIN)
                
            
        dt_TS = fechaInitoDateTimeN ( dt_ini, len(meds))
        med = datos.Medida('pronos',meds,dt_TS,tipoDato,nombre,minmax[0],minmax[1],nrep)
        #med.desfasar(-18*10/TS_MIN) # los pronósticos vienen con GMT 0, nosotros tenemos GMT -3
        medidas.append(med)
        
        if tipoDato == 'vel':
            vel = med
            
        if (tipoDato == 'dir') and (vel != None):
            
            velx, vely, vel3x, vel3y, vel3  = velxy_from_veldir(vel, med, ident, nidCentral)               
            medidas.append(velx)
            medidas.append(vely)
            medidas.append(vel3x)
            medidas.append(vel3y)
            medidas.append(vel3)            
            
            cosdir, sindir = cosin_from_dir(med, ident, nidCentral)
            medidas.append(cosdir)
            medidas.append(sindir)                
       
        #med.desfasar(-18) # los pronósticos vienen con GMT 0, nosotros tenemos GMT -3


    Medidor = datos.Medidor(ident,medidas,ubicacion)
           
    return Medidor

##############################################################################

def velxy_from_veldir(vel, dir_, ident, nidCentral):

    proc = vel.procedencia
    velx = vel.muestras * [m.cos(m.radians(k)) for k in dir_.muestras]
    vely = vel.muestras * [m.sin(m.radians(k)) for k in dir_.muestras]

    vel3 = (vel.muestras ** (3))
    vel3x = vel3 * [m.cos(m.radians(k)) for k in dir_.muestras]
    vel3y = vel3 * [m.sin(m.radians(k)) for k in dir_.muestras]


    velx = np.where(dir_.muestras > -1, velx, -99999999)
    vely = np.where(dir_.muestras > -1, vely, -99999999)
    
    vel3x = np.where(dir_.muestras > -1, vel3x, -99999999)
    vel3y = np.where(dir_.muestras > -1, vel3y, -99999999)    

    med_velx = datos.Medida(proc, velx, vel.tiempo,'vel','velx' + ident + '_' + str(nidCentral),
                       -vel.maxval, vel.maxval, vel.nrep)
    
    med_vely = datos.Medida(proc,vely, vel.tiempo,'vel','vely' + ident + '_' + str(nidCentral),
                       -vel.maxval, vel.maxval, vel.nrep)

    med_vel3x = datos.Medida(proc, vel3x, vel.tiempo,'vel','vel3x' + ident + '_' + str(nidCentral),
                       (-vel.maxval)** (3/2), vel.maxval, vel.nrep)
    
    med_vel3y = datos.Medida(proc, vel3y, vel.tiempo,'vel','vel3y' + ident + '_' + str(nidCentral),
                       -vel.maxval, vel.maxval, vel.nrep)        

    med_vel3 = datos.Medida(proc, vel3, vel.tiempo,'vel','vel3' + ident + '_' + str(nidCentral),
                       0, (vel.maxval)**3, vel.nrep)        
    
    
    return med_velx, med_vely, med_vel3x, med_vel3y, med_vel3

##############################################################################

def cosin_from_dir(dir_, ident, nid):

    proc = dir_.procedencia
    
    tipomed_cos = 'cosdir'
    tipomed_sin = 'sindir'
    
    min_, max_ = filtros.min_max(tipomed_cos,1,1)
    
    cos = [m.cos(m.radians(k)) for k in dir_.muestras]
    sin = [m.sin(m.radians(k)) for k in dir_.muestras]
    
    cos = np.where(dir_.muestras > -1, cos, -99999999)
    sin = np.where(dir_.muestras > -1, sin, -99999999)
    
    
    
    med_cos = datos.Medida(proc,cos,dir_.tiempo,tipomed_cos,tipomed_cos + ident + '_' + str(nid),
                       min_,max_,2)
    
    med_sin = datos.Medida(proc,sin,dir_.tiempo,tipomed_sin,tipomed_sin + ident + '_' + str(nid),
                       min_,max_,2)

    return med_cos, med_sin    

##############################################################################
 
def leerArchivosCentral (nidCentral):
    #
    # si existe, cargamos el objeto guardado
    #
    archip = archiPICKLE(nidCentral)
    if os.path.exists(archip):
        print('INFO: cargando datos de parque de ' + archip)
        return cargarCentral(nidCentral)
    #
    # si no, generamos todo desde 0 en base a los distintos archivos
    # de texto que componen la info de la central.
    #
    parque = leerArchi(nidCentral,'scada')
    
    parqueGen = leerArchi(nidCentral,'gen')
    if parqueGen is not None:
        parque.medidores[0].agregar_meds(parqueGen.medidores[0]._medidas)
        parque.pot_GEN = copy.deepcopy(parqueGen.pot)
    else:
        print("AVISO: No hay archivo GEN para esta central.")
    
    med_TS = leerArchiSMEC(nidCentral)
    if med_TS is not None:
        parque.pot_SMEC = med_TS
    else:
        print("AVISO: No hay archivo SMEC para esta central.")

    medidor_pronosTS = leerArchiPRONOS(nidCentral)
    if medidor_pronosTS is not None:
        parque.medidores[0].agregar_meds(medidor_pronosTS._medidas)
    else:
        print("AVISO: No hay archivo PRONOS para esta central.")

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
    
##############################################################################

def leer_ro_pendientes(nidcentral):

    archi_ro = archi_ro_pendientes(nidcentral)       

    if not os.path.exists(archi_ro):
        print('AVISO: no hay ro pendientes para esta central. Archivo ' + archi_ro + ' no encontrado.')
        return None
        exit

    print('Leyendo archivo de ro faltantes para la central ' + str(nidcentral) + ':' + archi_ro)
    
    f = open(archi_ro, 'r')
    
    # Leo datos
    lines = f.readlines()
    
    date_ini = []
    date_fin = []
    
    for k in range(len(lines)):
        line = lines[k]
        cols = re.split('\t|\n', line)
        
        fmt = '%d-%m-%y %H:%M'
        dtiniRO = datetime.datetime.strptime(cols[1], fmt )
        dtfinRO = datetime.datetime.strptime(cols[2], fmt)
              
        date_ini.append(dtiniRO)
        date_fin.append(dtfinRO)
   
    return date_ini, date_fin

def generar_ens_dte(pot_estim, pot_gen, dt, carpeta, nid_dbSMEC = 999):

    dif_pot_ = pot_estim - pot_gen
    
    dif_pot = np.where((pot_estim > 0) & (dif_pot_ >= 0), dif_pot_, 0)
    
    
    #d = {'ens': dif_pot, 'pot_estim': pot_estim, 'pot_gen': pot_gen}
    d = {'ENS_MWh': dif_pot}
    
    df = pd.DataFrame(data=d, index=dt)
    df.index.name = 'Fecha'
    
    df.to_csv(carpeta + 'RO_DTE_' + str(nid_dbSMEC) + '_3.txt', index=True, sep='\t', 
              float_format='%.4f', date_format='%d-%m-%Y %H:%M')
    
    df_desf = df.shift(periods=-1, fill_value=0)    
    df_h = df_desf.resample('H').sum()
    
    #print(df_h)    
    df_h['hora'] = pd.Series(df_h.index.hour, index=df_h.index)
    
    dia = df_h.index.day
    mes = df_h.index.month
    anio = df_h.index.year
    
    dia_dt = [datetime.datetime(anio, mes, dia) for anio, mes, dia
              in zip(anio, mes, dia)]
    
    df_h['dia'] = pd.Series(dia_dt, index=df_h.index)
    
    
    df_h.set_index(['dia', 'hora'], inplace=True)
    df_h = df_h.unstack('hora') 
    #print(df_h)
    
    df_h.to_csv(carpeta + 'ens_DTE.txt', index=True, sep='\t', 
                float_format='%.4f', date_format='%d-%m-%Y')
    
    
def generar_ens_topeada(nidCentral, Ptope):
    
    archi_gen = path_central(nidCentral) + 'archivos/medidasSMEC.txt'
    f = lambda s: datetime.datetime.strptime(s,'%d/%m/%Y')
    gen = pd.read_csv(archi_gen, index_col = 0, skiprows=1, sep = '\t',
                      date_parser = f) 
    gen = gen.drop(['Acumulado'], axis=1)

    f = lambda s: datetime.datetime.strptime(s,'%d-%m-%Y')
    carpeta_res = path_carpeta_resultados(nidCentral) 
    ens = pd.read_csv(carpeta_res + 'ens_DTE.txt', 
                      index_col = 0, skiprows=2, sep = '\t', date_parser = f)    

    m = (ens >= 0)

    ens = ens.where(m, 0)
    
    #print(ens)
    
    ens.columns = gen.columns
    gen.index.rename('dia', inplace = True)
    
    suma = gen.add(ens, fill_value=0)
    suma = Ptope - suma
    #print(suma)
    
    m = (suma < 0)
    recorte = suma.where(m, 0)
    
    print(f"recorte/ens_ini [%]= {-recorte.values.sum()/ens.values.sum()*100}")
    
    
    ens_recorte = ens.add(recorte, fill_value=0)
    #print(ens_recorte)
    
    m = (ens_recorte > 0)
    ens_recorte = ens_recorte.where(m, 0)
   
    ens_recorte.to_csv(carpeta_res + 'ens_DTE_topeo_iny.txt', index=True, sep='\t', 
                float_format='%.4f', date_format='%d-%m-%Y')
    
    
    
def resample_poly_dir(med, up_, down_):
    
    meds_sin = [m.sin(m.radians(k)) for k in med]
    meds_cos = [m.cos(m.radians(k)) for k in med]
    
    meds_sin_m = signal.resample_poly(meds_sin,up=up_,down=down_)
    meds_cos_m = signal.resample_poly(meds_cos,up=up_,down=down_)
                
    meds_m = [m.atan2(s,c) for s,c in zip(meds_sin_m, meds_cos_m)]
    meds_m = [m.degrees(k) for k in meds_m]
    
    for k in range(len(meds_m)):
        if meds_m[k] < 0 :
            meds_m[k] = meds_m[k] + 360
    
    return np.asarray(meds_m)        

def average(arr, n):
    end = n * int(len(arr)/n)
    arr_np = np.array(arr)
    return np.mean(arr_np[:end].reshape(-1, n), 1)    