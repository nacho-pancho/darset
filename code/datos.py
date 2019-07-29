# -*- coding: utf-8 -*-
"""
Este módulo incluye las clases que representan datos del sistema
a distintos niveles de agregación.

La estructura general es la siguiente:

* El Sistema está compuesto por un número de Plantas.

* Una Planta reune la salida de un número de Medidores, más un conjunto
  de Medidas propias, como ser la consigna, la generación del parque
  medida por los equipos del parque, y la generación del parque medida 
  de manera externa por ADME (SMEC)

* Un Medidor reune un conjunto de Medidas, por ejemplo velocidad del viento,
  temperatura, radiación solar, etc.
  
* Una Medida representa la serie temporal univariada de valores de una variable
  a lo largo de un tiempo determinado.

Según su nivel de agregación, los distintos objetos de datos son capaces
de realizar ciertos chequeos sobre los datos. El resultado de lo chequeos
se denomina 'filtro', y tiene un valor '1' cuando se detecta una anomalía,
y '0' en condiciones normales.

Por ejemplo, una Medida por sí misma puede determinar si ella está fuera de rango
o atascada. Estos son dos filtros incorporados por defecto en la clase 'Medida'.
Es posible agregar otros filtros a una Medida de ser necesario.

Un Parque puede medir decorrelaciones entre la potencia reportada por el parque
y la medida de velocidad del viento de sus medidores. También puede detectar si acaso una medida
está desfasada respecto a las demás.

El Sistema en su conjunto puede detectar comportamientos anómalos entre medidas 
tomadas en distintos parques.

Created on Thu May  2 16:07:55 2019
@author: fpalacio
"""

import numpy as np
import filtros as f
import scipy
import datos as d
import scipy.stats as r
import matplotlib.pyplot as plt
from windrose import WindroseAxes

class Ubicacion(object):
    '''
    Ubicacion fisica en estandar UTM
    '''    
    def __init__(self,zona,huso,x,y,ident):
        '''
        inicializa en base a atributos
        '''
        self.zona = zona
        self.huso = huso
        self.x = x
        self.y = y
        self.ident = ident

#    def __init__(self,texto):
        '''
        inicializa la ubicacion en base a un texto en formato estandar
        '''
#        self.zona = 0
        
 # nacho fing eduy, ignacio.ramirez gmail, ignacio.ramirez.iie 

class Medida(object):
    '''
    Representa una serie de tiempo asociada a una medida
    de tipo particular, por ejemplo meteorologica o de potencia
    '''
    def __init__(self,muestras,tiempo,tipo,nombre,minval,maxval,nrep):
        self.muestras = muestras
        self.tiempo = tiempo
        self.tipo = tipo # vel,dir, rad, temp, etc
        self.nombre = nombre #vel_scada, vel_dir, vel_otrogen,etc
        self.minval = minval
        self.maxval = maxval
        self.nrep = nrep
        self.filtros = dict()
        self.agregar_filtro('fuera_de_rango',f.filtrar_rango
                            (self.muestras,self.minval,self.maxval))
        if self.tipo != 'corr':        
            self.agregar_filtro('trancada',f.filtrar_rep
                                (self.muestras,self.get_filtro('fuera_de_rango'),self.nrep))

    def agregar_filtro(self,nombre_f,filt):
        print ('filtro',filt)
        self.filtros[self.nombre + '_' + nombre_f] = filt.astype(np.uint8)
        
    def get_filtro(self,nombre_f):
        return self.filtros[self.nombre + '_' + nombre_f]
    
    def get_filtros(self):
        return self.filtros
 
    def filtrada(self):
        filtrada = np.zeros(len(self.muestras),dtype=bool)
        for f in self.filtros.values():
            filtrada = filtrada | f
        return filtrada    
    
class Medidor(object):
    '''
    Genera una o varias Medidas en determinados
    instantes de tiempo
    @see Medida
    '''

    def __init__(self, nombre, medidas, ubicacion):
        self.nombre = nombre
        self.medidas = medidas
        self.ubicacion = ubicacion 
        plt.figure()
        self.plot_rosa_vientos()

    def get_medida(self,t):
        for m in self.medidas:
            if m.tipo == t:
                return m
        print(f"AVISO: medida de tipo {t} no encontrada.")
        return None
    
    def plot_rosa_vientos(self):
        
        vel = self.get_medida('vel')
        dir_ = self.get_medida('dir')
        
        filtro = vel.filtrada() | dir_.filtrada()
        
        wd = dir_.muestras[filtro < 1]
        ws = vel.muestras[filtro < 1]
        ax = WindroseAxes.from_ax()
        ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
        ax.set_legend()
        
        
class Parque(object):
    '''
    Representa un parque de generación de energía
    Tiene asociadas medidas de potencia y uno o
    mas medidores.
    '''
    def __init__(self,medidores,cgm,pot,dis):
        if isinstance(medidores,list):
            self.medidores = medidores
        else:
            self.medidores = list()
            self.medidores.append(medidores)
        self.cgm = cgm
        self.pot = pot
        self.pot_SMEC = None
        self.dis = dis
        self.decorr = None
        self._filtro_cgm = None
        self._filtro_potBaja = None

    def filtro_cgm(self):
        if self._filtro_cgm == None:
            self._filtro_cgm = np.abs(self.pot.muestras - self.cgm.muestras) < (self.cgm.maxval * 0.1)                 
        return self._filtro_cgm

    def filtro_potBaja(self):
        if self._filtro_potBaja == None:
            self._filtro_potBaja = np.abs(self.pot.muestras) < (self.cgm.maxval * 0.05)                 
        return self._filtro_potBaja    
        
    def decorrelacion(self):
        '''
        Arma un filtro para cada medidor
        donde 1 en cada filtro indica que en ese momento
        el medidor corresp. esta decorrelacionado con 
        la potencia reportada por el parque
        '''
        if self.decorr is None:
            self.decorr = dict()
            for med in self.medidores:
                self.decorr[med.nombre] = self.decorrelacion_medidor(med)
        return self.decorr
            
    def decorrelacion_medidor (self,med):
        
        vel = med.get_medida('vel')
        
        filtro_cons = self.filtro_cgm()
        filtro_vel = vel.filtrada()
        filtro_pot = self.pot.filtrada()
        filtro_potBaja = self.filtro_potBaja()
        filtro_total = filtro_cons | filtro_vel | filtro_pot | filtro_potBaja
        
        NDatosCorr = 12     

        idx_mask = np.where(filtro_total < 1)
        
        vel_m = vel.muestras
        pot_m = self.pot.muestras
        
        vel_m_mask = vel_m[idx_mask]
        pot_m_mask = pot_m[idx_mask]
        
        plt.figure()
        plt.scatter(vel_m[idx_mask], pot_m[idx_mask])

        vel_m_mask_u = r.rankdata(vel_m_mask, "average")#/len(vel_m_mask)
        pot_m_mask_u = r.rankdata(pot_m_mask, "average")#/len(pot_m_mask)        
        
        vel_m_mask_u = vel_m_mask_u / np.max(vel_m_mask_u)
        pot_m_mask_u = pot_m_mask_u / np.max(pot_m_mask_u)
        
        plt.figure()        
        vel_m_u = np.zeros(len(vel_m))
        pot_m_u = np.zeros(len(pot_m))
        
        vel_m_u [idx_mask] = vel_m_mask_u 
        pot_m_u [idx_mask] = pot_m_mask_u
        plt.scatter(vel_m_mask_u, pot_m_mask_u)
        
        vel_m_g = np.zeros(len(vel_m))
        pot_m_g = np.zeros(len(pot_m))
        
        vel_m_g [idx_mask] = r.norm.ppf(vel_m_mask_u) 
        pot_m_g [idx_mask] = r.norm.ppf(pot_m_mask_u)

        plt.figure()        
        plt.scatter(vel_m_g [idx_mask], pot_m_g [idx_mask])
         
        idx_buff = np.zeros(NDatosCorr,dtype=int)
        corr = np.zeros(len(vel.muestras))
        k_idx_buff = 0
        k = 0

        cualquiera = list()
        decorr = False
        while k < len(vel_m_g):
            if not filtro_total[k]:
                if k_idx_buff < NDatosCorr:
                    idx_buff[k_idx_buff] = k
                    k_idx_buff = k_idx_buff + 1
                    k = k + 1
                    continue
                else:
                    idx_buff_aux = idx_buff
                    idx_buff[1:NDatosCorr-1] = idx_buff_aux[0:NDatosCorr-2]
                    idx_buff[0] = k
                #print(idx_buff)
                #corr[k] = scipy.stats.pearsonr(vel_m_g[idx_buff], pot_m_g[idx_buff])[0]
                corr[k] = np.dot(vel_m_u[idx_buff], pot_m_u [idx_buff]) /(np.linalg.norm(vel_m_u[idx_buff]) * np.linalg.norm(pot_m_u[idx_buff]))
                #print(k,corr[k])
                if corr[k] < 0.7:
                    print(k,corr[k])
                    if not decorr:
                        cualquiera.append(k)
                        decorr = True
                else:
                    decorr = False
            else:
                corr[k] = corr[k-1]
                #print('(filtrado)')
            k = k + 1
        print(cualquiera)
        print('Episodios:',len(cualquiera))
        return Medida(corr,vel.tiempo,'corr','corr_vel_pot',0.7,1.0,0)
    
    def deriva (self):
        return None
    
    def correlacion_dir(self,medidores):
        dir_ref = medidores[0].get_medida('dir')
        dir_2 = medidores[1].get_medida('dir')
        
        dir_dif  = dir_2 - dir_ref
        
        return Medida(np.cos(dir_dif),dir_ref.tiempo,'corr','corr_dir_dir',0.7,1.0,0)
        
        
        
    
    
class Concentrador(object):
    '''
    Reune (concentra) la informacion de varios parques.
    Sirve para tener un panorama general, y para detectar
    anomalias a escala global, como derivas.
    '''
    def __init__(self,parques):
        self.parques = parques

    def deriva (self):
        return None
        
    @staticmethod
    def geteventos(filtro, margen=0):
        '''
        devuelve una lista de rangos de a forma
        np.arange(a,b) -> a,.....,b-1
        correspondientes a lapsos de tiempo durante
        los que se detectó alguna anomalía.
        los rangos pueden ser "engorfados" por un margen dado 
        (por defecto 0) para darle contexto a un graficado posterior.
        '''
        N = len(filtro)
        f0 = filtro.astype(filtro,np.int8)
    
#pepe = Ubicacion("loma del orto")
