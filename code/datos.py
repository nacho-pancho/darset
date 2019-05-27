# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:07:55 2019

@author: fpalacio
"""

import numpy as np
import filtros as f

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
        self._fr = None
        self._tranc = None
        self.fuera_de_rango()
        self.trancada()

        
    def fuera_de_rango(self):
        '''
        devuelve una lista de verdadero/falso segun la medida
        esté fuera de rango o no
        '''
        if self._fr is None:
            self._fr  = f.filtrar_rango(self.muestras,self.minval,self.maxval)
        return self._fr    
    
    def trancada(self):
        '''
        devuelve una lista de verdadero/falso según se detecte
        que la medida está trancada en ciertos intervalos de tiempo
        '''
        if self._tranc is None:
            if self._fr is None:
                self.fuera_de_rango()
            self._tranc,cnt = f.filtrar_rep(self.muestras,self._fr,self.nrep)
        return self._tranc
    
    def filtrosAsInt(self):
        
        filtros = np.array((self._fr, self._tranc)).T.astype(int)
        nombres = [self.nombre + '_filtro_fr',self.nombre + '_filtro_tranc']
        
        return filtros, nombres

class Medidor(object):
    '''
    Genera una o varias Medidas en determinados
    instantes de tiempo
    @see Medida
    '''

    def __init__(self, medidas, ubicacion):
        self.medidas = medidas
        self.ubicacion = ubicacion
    
    def decorrelacion(self):
        '''
        devuelve una matriz de mxn con m la cantidad de medidas
        y n la cantidad de muestras(comun a todas las medidas).
        True en la posicion [i,j] indica que la medida i
        está decorrelacionada del resto en el tiempo j.
        '''
        m = len(self.medidas)
        n = len(self.tiempo)
        trancadas = np.zeros((m,n))
        for i in len(self.medidas):
            trancadas[i,:] = self.medidas[i].trancada()
        # FALTA TERMINAR
        return 
    
        
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
        self.dis = dis
        
    def descorrelacion (self):
        return None
    
    def deriva (self):
        return None
    
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
