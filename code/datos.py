# -*- coding: utf-8 -*-
"""
Este módulo incluye las clases que representan datos del sistema
a distintos niveles de agregación.

La estructura general es la siguiente:

* El SISTEMA está compuesto por un número de PLANTAS.

* Una PLANTA reune la salida de un número de MEDIDORES, más un conjunto
  de MEDIDAS propias, como ser la consigna, la generación del parque
  medida por los equipos del parque, y la generación del parque medida 
  de manera externa por ADME (SMEC)

* Un MEDIDOR reune un conjunto de MEDIDAS, por ejemplo velocidad del viento,
  temperatura, radiación solar, etc.
  
* Una MEDIDA representa la serie temporal univariada de valores de una variable
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
##############################################################################

import numpy as np
import filtros as f
import datos as d
import scipy.stats as r
import archivos as arch
import copy

##############################################################################

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

##############################################################################

class Medida(object):
    '''
    Representa una serie de tiempo asociada a una medida
    de tipo particular, por ejemplo meteorologica o de potencia
    '''
    def __init__(self,procedencia,muestras,tiempo,tipo,nombre,minval,maxval,nrep):
        self.procedencia = procedencia
        self.muestras = muestras
        self.tiempo = tiempo
        self.tipo = tipo # vel,dir, rad, temp, etc
        self.nombre = nombre #vel_scada, vel_dir, vel_otrogen,etc
        self.minval = minval
        self.maxval = maxval
        self.nrep = nrep
        self.filtros = dict()
        self.calcular_filtros()
        


    def calcular_filtros(self):
        if (self.tipo != 'Ndesf_opt_k'):        
            self.agregar_filtro('fuera_de_rango',f.filtrar_rango
                            (self.muestras,self.minval,self.maxval))
        if (self.tipo != 'corr') and (self.tipo != 'Ndesf_opt_k'):        
            self.agregar_filtro('trancada',f.filtrar_rep
                                (self.muestras,self.get_filtro('fuera_de_rango'),self.nrep))

    def agregar_filtro(self,nombre_f,filt):
        self.filtros[self.nombre + '_' + nombre_f] = filt.astype(np.uint8)

    
        
    def get_filtro(self,nombre_f):
        return self.filtros[self.nombre + '_' + nombre_f]


    
    def get_filtros(self):
        return self.filtros
    
    
    def reset_filtros(self,NMuestras):
        self.filtros.clear()

    def reset_muestrasYTiempo(self,dtini,NMuestras):
        self.tiempo = arch.fechaInitoDateTimeN(dtini,NMuestras)
        self.muestras = np.full(NMuestras,1.5 * self.maxval)
    
    def reset_med (self,dtini,NMuestras):
        self.reset_filtros(NMuestras)
        self.reset_muestrasYTiempo(dtini,NMuestras)
        
 
    def filtrada(self):
        filtrada = np.zeros(len(self.muestras),dtype=bool)
        for f in self.filtros.values():
            filtrada = filtrada | f
        return filtrada    


    def desfasar(self,Ndesf):
        dt_desf = (self.tiempo[1] - self.tiempo[0]) * Ndesf
        self.tiempo = [dt +  dt_desf for dt in self.tiempo] 
        return None

    def desfasar_dinamico (self,Ndesf):
        
        med_old = copy.deepcopy(self)
        
        m_old = med_old.muestras
        f_old = med_old.get_filtros()
        dtini_old = med_old.tiempo[0]

        # Redimensiono muestras y filtros
        dt_m = (self.tiempo[1] - self.tiempo[0])
        dt_new = [self.tiempo[k] + Ndesf[k] * dt_m for k in range(len(med_old.tiempo))]
        dtmin,dtmax = np.min(dt_new),np.max(dt_new)
        
        NMuestras = arch.NMuestras10minEntreDts(dtmin,dtmax) + 1

        self.reset_med(dtmin,NMuestras)
    
        N10min_des = arch.NMuestras10minEntreDts(dtini_old,dtmin)
        
        k_new = [k_old + Ndesf[k_old] - N10min_des for k_old in range(len(m_old))]

        self.muestras[k_new] = m_old
        
        self.calcular_filtros()
        
        
        #del  med_old
        return None
        
  
##############################################################################
    
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

    def get_medida(self,tipo,proc):
        for m in self.medidas:
            if (m.tipo == tipo) and (m.procedencia == proc):
                return m
        print(f"AVISO: medida de tipo {tipo} y procedencia {proc} no encontrada.")
        return None

    def agregar_meds(self,meds):
        for m in meds:
           self.medidas.append(m)
        

    def calcular_filtros(self):
        for med in self.medidas:
            tipo_m = med.tipo
            proc_m = med.procedencia
            if (proc_m != 'pronos'):
                if tipo_m in ('vel','dir','rad','temp'):
                    med_ref = self.get_medida(tipo_m,'pronos')
                    f.corr_medidas(med_ref,med,6,0,True)
                    
    def desfasar_meds(self):
        for med in self.medidas:
            if med.procedencia == 'pronos':
                if med.tipo == 'rad':
                    med.desfasar(-18)
    
            
            
            

        
##############################################################################

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


    def calc_corr_medidor(self,med):
        return None
        

    def calcular_filtros(self):
        
        '''
        Primero tengo que acomodar las series que estuvieran desfasadas
        '''
        
        self.pot_SMEC.desfasar(-1)
        
        for med in self.medidores:
            med.desfasar_meds()        
        
        

        '''
        Calcular los filtros de los medidores
        '''
        for med in self.medidores:
            med.calcular_filtros()

        '''
        Calcular los filtros del parque
        '''        
        
        return None
        
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
        
        #plt.figure()
        #plt.scatter(vel_m[idx_mask], pot_m[idx_mask])

        vel_m_mask_u = r.rankdata(vel_m_mask, "average")#/len(vel_m_mask)
        pot_m_mask_u = r.rankdata(pot_m_mask, "average")#/len(pot_m_mask)        
        
        vel_m_mask_u = vel_m_mask_u / np.max(vel_m_mask_u)
        pot_m_mask_u = pot_m_mask_u / np.max(pot_m_mask_u)
        
        #plt.figure()        
        vel_m_u = np.zeros(len(vel_m))
        pot_m_u = np.zeros(len(pot_m))
        
        vel_m_u [idx_mask] = vel_m_mask_u 
        pot_m_u [idx_mask] = pot_m_mask_u
        #plt.scatter(vel_m_mask_u, pot_m_mask_u)
        
        vel_m_g = np.zeros(len(vel_m))
        pot_m_g = np.zeros(len(pot_m))
        
        vel_m_g [idx_mask] = r.norm.ppf(vel_m_mask_u) 
        pot_m_g [idx_mask] = r.norm.ppf(pot_m_mask_u)

        #plt.figure()        
        #plt.scatter(vel_m_g [idx_mask], pot_m_g [idx_mask])
         
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
                corr[k] = np.dot(vel_m_u[idx_buff], pot_m_u [idx_buff]) /(np.linalg.norm(vel_m_u[idx_buff]) * np.linalg.norm(pot_m_u[idx_buff]))
                if corr[k] < 0.7:
                    if not decorr:
                        cualquiera.append(k)
                        decorr = True
                else:
                    decorr = False
            else:
                corr[k] = corr[k-1]
            k = k + 1
        return Medida(corr,list(vel.tiempo),'corr','corr_vel_pot',0.7,1.0,0)


    
    def deriva (self):
        return None


    
    def correlacion_dir(self,medidores):
        dir_ref = medidores[0].get_medida('dir')
        dir_2 = medidores[1].get_medida('dir')
        
        dir_dif  = dir_2 - dir_ref
        
        return Medida(np.cos(dir_dif),dir_ref.tiempo,'corr','corr_dir_dir',0.7,1.0,0)
        
##############################################################################
            
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
    
##############################################################################
