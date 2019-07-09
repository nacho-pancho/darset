# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:07:55 2019

@author: fpalacio
"""

import numpy as np
import filtros as f
import scipy

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

    def filtrada(self):
        return self.trancada() | self.fuera_de_rango()
    
class Medidor(object):
    '''
    Genera una o varias Medidas en determinados
    instantes de tiempo
    @see Medida
    '''

    def __init__(self, medidas, ubicacion):
        self.medidas = medidas
        self.ubicacion = ubicacion    

    def get_medida(self,t):
        for m in self.medidas:
            if m.tipo == t:
                return m
        print(f"AVISO: medida de tipo {t} no encontrada.")
        return None
    
        
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
        self.filtro_vel_pot = None
        self._filtro_cgm = None
        
    def filtro_cgm(self):
        if self._filtro_cgm == None:
            self._filtro_cgm = np.abs(self.pot.muestras - self.cgm.muestras) < (self.cgm.maxval * 0.1)                 
        return self._filtro_cgm
        
    def decorrelacion(self):
        '''
        Arma un filtro para cada medidor
        donde 1 en cada filtro indica que en ese momento
        el medidor corresp. esta decorrelacionado con 
        la potencia reportada por el parque
        '''
        self.decorr = list()
        for med in self.medidores:
            self.decorr.append( self.decorrelacion_medidor(med) )
        
        return self.decorr
            
    def decorrelacion_medidor (self,med):
        
        vel = med.get_medida('vel')
        
        filtro_cons = self.filtro_cgm()
        filtro_vel = vel.filtrada()
        filtro_pot = self.pot.filtrada()
        filtro_total = filtro_cons | filtro_vel | filtro_pot
        
        NDatosCorr = 30 
        

        idx_buff = np.zeros(NDatosCorr,dtype=int)
        corr = np.zeros(len(vel.muestras))
        k_idx_buff = 0
        k = 0
        vel_m = vel.muestras
        pot_m = self.pot.muestras
        
        while k < len(vel_m):
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
                corr[k] = scipy.stats.spearmanr(vel_m[idx_buff], pot_m[idx_buff])[0]
                print(k,corr[k])
            else:
                corr[k] = corr[k-1]
                print('(filtrado)')
            k = k + 1
                
        #ahora tengo que ver si corr cambia mucho poner filtro = 1
        
        corr = corr        
        return corr
              
    
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
