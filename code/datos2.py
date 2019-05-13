# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:07:55 2019

@author: fpalacio
"""

import numpy as np

class Ubicacion(object):
    '''
    Ubicacion fisica en estandar UTM
    '''
    
    def __init__(self,zona,huso,x,y):
        '''
        inicializa en base a atributos
        '''
        self.zona = zona
        self.huso = huso
        self.x = x
        self.y = y

    def __init__(self,texto):
        '''
        inicializa la ubicacion en base a un texto en formato estandar
        '''
        self.zona = 0
        
 # nacho fing eduy, ignacio.ramirez gmail, ignacio.ramirez.iie 

class Medida(object):
    '''
    Representa una serie de tiempo asociada a una medida
    de tipo particular, por ejemplo meteorologica o de potencia
    '''
    def __init__(self,muestras,tipo,nombre,minval,maxval):
        self.muestras = muestras
        self.tipo = tipo # vel,dir, rad, temp, etc
        self.nombre = nombre #vel_scada, vel_dir, vel_otrogen,etc
        self.minval = minval
        self.maxval = maxval
        self._tranc = None
        self._fr = None
        
    def fuera_de_rango(self):
        '''
        devuelve una lista de verdadero/falso segun la medida
        esté fuera de rango o no
        '''
        if self._fr is None:
            self._fr = (self.muestras > maxval) or (self.muestras < minval)
        return self._fr    
    
    def trancada(self):
        '''
        devuelve una lista de verdadero/falso según se detecte
        que la medida está trancada en ciertos intervalos de tiempo
        '''
        if self._tranc is None:
            self._tranc = np.zeros(len(self.muestras))
        return self._tranc


class Medidor(object):
    '''
    Genera una o varias Medidas en determinados
    instantes de tiempo
    \see Medida
    '''
    def __init__(self,tiempo, medidas, ubicacion):
        self.tiempo = tiempo
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
        for i in len(medidas):
            trancacas[i,:] = self.medidas[i].trancada()
            
        return 
    
        
class Parque(object):
    '''
    Representa un parque de generación de energía
    Tiene asociadas medidas de potencia y uno o
    mas medidores.
    '''
    def __init__(self,medidores,cgm,pot):
        self.medidores = medidores
        self.cgm = cgm
        self.pot = pot
        
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
        
pepe = Ubicacion("loma del orto")
