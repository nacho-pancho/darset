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

FUERA_DE_RANGO = -1e20

##############################################################################

class Ubicacion(object):
    """
    Ubicacion fisica en estandar UTM
    """
    def __init__(self,zona,huso,x,y,ident):
        """
        inicializa en base a atributos
        """
        self.zona = zona
        self.huso = huso
        self.x = x
        self.y = y
        self.ident = ident

##############################################################################

import datetime

class Medida(object):
    """
    Representa una serie de tiempo asociada a una medida
    de tipo particular, por ejemplo meteorologica o de potencia
    """

    def __init__(self,procedencia,muestras,tiempo,tipo,nombre,minval,maxval,nrep):
        t0 = tiempo[0]
        # print(f"Medida {procedencia} de tipo {tipo}, nombre {nombre}, comenzando en tiempo {t0}")
        self.procedencia = procedencia
        self.muestras = muestras
        self.tiempo = tiempo
        self.tipo = tipo # vel,dir, rad, temp, etc
        self.nombre = nombre #vel_scada, vel_dir, vel_otrogen,etc
        self.minval = minval
        self.maxval = maxval
        self.nrep = nrep
        self._filtros = None


    def get_tiempo(self):
        """
        Devuelve el conjunto de tiempos en lo que s emidió
        """
        return self.tiempo


    def registrar(self,nuevos_tiempos):
        """
        Reubica las muestras y el tiempo de la serie al rango de tiempos especificado
        Asumimos que el tiempo inicial ya está alineado a 10 minutos
        """
        dtd10 = datetime.timedelta(minutes=10)
        tini1 = nuevos_tiempos[0]
        n0 = len(self.tiempo)
        tini0 = self.tiempo[0]
        offset = int((tini0-tini1)/dtd10)

        self.tiempo = nuevos_tiempos
        muestras_viejas = self.muestras
        self.muestras = np.ones(len(nuevos_tiempos))* FUERA_DE_RANGO
        self.muestras[offset:(offset + n0)] = muestras_viejas

    def calcular_filtros(self):
        """
        Calcula todos los filtros disponibles para esta medida
        """
        self._filtros = dict()
        if self.tipo != 'Ndesf_opt_k':
            filtro = f.filtrar_rango(self.muestras, self.minval, self.maxval)
            self.agregar_filtro('fuera_de_rango',filtro)
        if (self.tipo != 'corr') and (self.tipo != 'Ndesf_opt_k'):
            filtro = f.filtrar_rep(self.muestras,self.get_filtro('fuera_de_rango'),self.nrep)
            self.agregar_filtro('trancada',filtro)


    def agregar_filtro(self,nombre_f,filt):
        """
        Agrega un nuevo tipo de filtro a la medida
        """
        self._filtros[self.nombre + '_' + nombre_f] = filt.astype(np.uint8)


    def get_filtro(self,nombre_f):
        """
        Obtiene el filtro de nombre especificado
        """
        filtros = self.get_filtros()
        return filtros[self.nombre + '_' + nombre_f]


    def get_filtros(self):
        """
        Devuelve una lista con todos los filtros
        """
        if self._filtros is None:
            self.calcular_filtros()
        return self._filtros
    
    
#    def reset_filtros(self,NMuestras):
#        """
#        Borra todos los filtros
#        """
#        self.filtros.clear()#
#
#    def reset_muestrasYTiempo(self,dtini,NMuestras):
#        self.tiempo = arch.fechaInitoDateTimeN(dtini,NMuestras)
#        self.muestras = np.full(NMuestras,1.5 * self.maxval)
#
#    def reset_med (self,dtini,NMuestras):
#        self.reset_filtros(NMuestras)
#        self.reset_muestrasYTiempo(dtini,NMuestras)
#
 
    def filtrada(self):
        filtrada = np.zeros(len(self.muestras),dtype=bool)
        for f in self.get_filtros().values():
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
        
        return None
        
  
##############################################################################
    
class Medidor(object):
    """
    Genera una o varias Medid/as en determinados
    instantes de tiempo
    @see Medida
    """

    def __init__(self, nombre, medidas, ubicacion):
        nm = len(medidas)
        print(f"Medidor {nombre} con {nm} medidas ubicado en {ubicacion}")
        self.nombre = nombre
        self._medidas = medidas
        self.ubicacion = ubicacion
        self._filtros = None

    def get_medida(self,tipo,proc):
        for m in self._medidas:
            if (m.tipo == tipo) and (m.procedencia == proc):
                return m
        print(f"AVISO: medida de tipo {tipo} y procedencia {proc} no encontrada.")
        return None


    def get_medidas(self):
        return self._medidas


    def agregar_meds(self,meds):
        for m in meds:
           self._medidas.append(m)

    def get_tiempo(self):
        """
        Devuelve el período máximo que cubre los tiempos de medición de todas las medidas
        en este medidor.
        """
        t = self._medidas[0].get_tiempo()
        tmin = t[0]
        tmax = t[-1]
        for i in range(1, len(self._medidas)):
            t = self._medidas[i].get_tiempo()
            t0 = t[0]
            t1 = t[-1]
            nom = self._medidas[i].nombre
            print(f"{nom} {t0} {t1}")
            if t0 < tmin:
                tmin = t0
            if t1 > tmax:
                tmax = t1
        dtd10 = datetime.timedelta(minutes=10)
        n =int( np.ceil( (tmax-tmin)/dtd10 ) )+ 1
        tiempo = [tmin+dtd10*i for i in range(n)]
        return tiempo

    def registrar(self,periodo):
        if periodo is None:
            periodo = self.get_tiempo()

        for m in self._medidas:
            m.registrar(periodo)


    def get_filtros(self):
        if self._filtros is None:
            self.calcular_filtros()
        return self._filtros


    def calcular_filtros(self):
        self._filtros = dict()
        for med in self._medidas:
            self._filtros.update(med.get_filtros())
            tipo_m = med.tipo
            proc_m = med.procedencia
            if proc_m != 'pronos':
                if tipo_m in ('vel','dir','rad','tem'):
                    med_ref = self.get_medida(tipo_m,'pronos')
                    if med_ref != None:
                        #med_corr = f.corr_medidas(med_ref,med,6,0,True)
                        #self._filtros.update(med_corr.get_filtros())
                        a = 1
        return None

        
##############################################################################

class Parque(object):
    """
    Representa un parque de generación de energía
    Tiene asociadas medidas de potencia y uno o
    mas medidores.
    """
    def __init__(self, id, medidores, cgm, pot, dis, PAutorizada, cant_molinos):
        if isinstance(medidores,list):
            self.medidores = medidores
        else:
            self.medidores = list()
            self.medidores.append(medidores)
        self.id = id
        self.cgm = cgm
        self.pot = pot # medida principal del parque
        self.pot_SMEC = None # medida principal según SMEC, no siempre disponible
        self.dis = dis
        self.decorr = None
        self._filtros = None
        self.PAutorizada = PAutorizada
        self.cant_molinos = cant_molinos

    def get_periodo(self):
        
        if (self.pot != None):
            t  = self.pot.get_tiempo()
            tmin,tmax = t[0],t[-1]
        
        if (self.cgm != None):
            t  = self.cgm.get_tiempo()
            tmin,tmax = min(t[0],tmin),max(t[-1],tmax)
            
        if (self.dis != None):
            t  = self.dis.get_tiempo()
            tmin,tmax = min(t[0],tmin),max(t[-1],tmax)


        for i in range(len(self.medidores)):
            t = self.medidores[i].get_tiempo()
            t0, t1 = t[0],t[-1]
            if t0 < tmin:
                tmin = t0
            if t1 > tmax:
                tmax = t1
        dtd10 = datetime.timedelta(minutes=10)
        n = int( np.ceil( (tmax-tmin)/dtd10 ) ) + 1 
        tiempo = [tmin+dtd10*i for i in range(n)]
        return tiempo

    def registrar(self):
        periodo = self.get_periodo()
        for M in self.medidores:
            M.registrar(periodo)
        self.pot.registrar(periodo)
        
        if self.pot_SMEC is not None:
            self.pot_SMEC.registrar(periodo)
        if self.cgm is not None:
            self.cgm.registrar(periodo)
        if self.dis is not None:
            self.dis.registrar(periodo)            
        # resetear filtros, etc.
        self.decorr = None
        self._filtros = None

    def get_filtros(self):

        if (self.pot != None) and (self.pot._filtros is None):
            self.pot.calcular_filtros()            
        
        if (self.cgm != None) and (self.cgm._filtros is None):
            self.cgm.calcular_filtros()            

        if (self.pot_SMEC != None) and (self.pot_SMEC._filtros is None):
            self.pot_SMEC.calcular_filtros()   

        if (self.dis != None) and (self.dis._filtros is None):
            self.dis.calcular_filtros()             
    
        if self._filtros is  None:
            return self.calcular_filtros()
        return self._filtros

    def get_medidas(self):
        todas_las_medidas = list()

        if self.pot is not None:
            todas_las_medidas.append(self.pot)
        if self.cgm is not None:
            todas_las_medidas.append(self.cgm)
        if self.dis is not None:
            todas_las_medidas.append(self.dis)
            
        for M in self.medidores:
            for m in M.get_medidas():
                todas_las_medidas.append(m)
        return todas_las_medidas


    def exportar_medidas(self):
        """
        Devuelve las medidas y sus filtros totales correspondientes como dos matrices M y F
        La primera tiene valores reales y contiene una medida en cada fila i
        LA segunda tiene valores booleanos y la fila i tiene el filtro de la medida i
        """
        meds = self.get_medidas()
        ncols = len(meds)
        nrows = len(meds[0].tiempo)
        M = np.zeros((nrows,ncols),dtype=np.single)
        F = np.zeros((nrows,ncols),dtype=np.bool)
        nombres =list()
        for i in range(ncols):
            nombres.append(meds[i].nombre)
            M[:,i] = meds[i].muestras
            F[:,i] = meds[i].filtrada()
        
        tiempo_ = meds[0].tiempo
        
        return M,F,nombres,tiempo_
    

    def calcular_filtros(self):

        self._filtros = dict()
        
        '''
        Calcular los filtros de los medidores
        '''
        for med in self.medidores:
            dict.update(med.get_filtros())
        
        '''
        Calcular los filtros del parque
        '''        
        #filt_cgm = np.abs(self.pot.muestras - self.cgm.muestras) < (self.PAutorizada * 0.05)
        filt_cgm = np.abs(self.pot.muestras - self.cgm.muestras) <= 2 # MW
        filt_cgm = filt_cgm & (self.cgm.muestras < 0.999 * self.PAutorizada )#* np.ones(len(self.cgm.muestras)))
        
        Ndatos_afectados_RO_izq = 1
        Ndatos_afectados_RO_der = 5
        
        filt_cgm_ = copy.deepcopy(filt_cgm)
        
        for k in range(Ndatos_afectados_RO_izq, len(filt_cgm)-Ndatos_afectados_RO_der):            
            if filt_cgm_[k] == 1:
                filt_cgm[k-Ndatos_afectados_RO_izq:k+Ndatos_afectados_RO_der] = 1

            
            
        
        self.pot.agregar_filtro('cgm',filt_cgm)
        
        filt_dis = self.dis.muestras < self.cant_molinos

        #self.pot.agregar_filtro('disp_menor_max',filt_dis)
        
        self._filtros['cgm'] = filt_cgm
        self._filtros['disp_menor_max'] = filt_dis
        self._filtros['pot_baja'] = np.abs(self.pot.muestras) < (self.cgm.maxval * 0.05)

        return None


    def decorrelacion(self):
        """
        Arma un filtro para cada medidor
        donde 1 en cada filtro indica que en ese momento
        el medidor corresp. esta decorrelacionado con
        la potencia reportada por el parque
        """
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

        vel_m_mask_u = r.rankdata(vel_m_mask, "average")#/len(vel_m_mask)
        pot_m_mask_u = r.rankdata(pot_m_mask, "average")#/len(pot_m_mask)        
        
        vel_m_mask_u = vel_m_mask_u / np.max(vel_m_mask_u)
        pot_m_mask_u = pot_m_mask_u / np.max(pot_m_mask_u)
        
        vel_m_u = np.zeros(len(vel_m))
        pot_m_u = np.zeros(len(pot_m))
        
        vel_m_u [idx_mask] = vel_m_mask_u 
        pot_m_u [idx_mask] = pot_m_mask_u

        vel_m_g = np.zeros(len(vel_m))
        pot_m_g = np.zeros(len(pot_m))
        
        vel_m_g [idx_mask] = r.norm.ppf(vel_m_mask_u) 
        pot_m_g [idx_mask] = r.norm.ppf(pot_m_mask_u)

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


    def correlacion_dir(self,medidores):
        dir_ref = medidores[0].get_medida('dir')
        dir_2 = medidores[1].get_medida('dir')
        dir_dif  = dir_2 - dir_ref
        return Medida(np.cos(dir_dif),dir_ref.tiempo,'corr','corr_dir_dir',0.7,1.0,0)

        
##############################################################################
            
class Concentrador(object):
    """
    Reune (concentra) la informacion de varios parques.
    Sirve para tener un panorama general, y para detectar
    anomalias a escala global, como derivas.
    """

    def __init__(self,parques):
        self.parques = parques


    @staticmethod
    def geteventos(filtro, margen=0):
        """
        devuelve una lista de rangos de a forma
        np.arange(a,b) -> a,.....,b-1
        correspondientes a lapsos de tiempo durante
        los que se detectó alguna anomalía.
        los rangos pueden ser "engorfados" por un margen dado
        (por defecto 0) para darle contexto a un graficado posterior.
        """
        N = len(filtro)
        f0 = filtro.astype(filtro,np.int8)
    
##############################################################################
