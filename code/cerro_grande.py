#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 03:44:51 2019

@author: felipep
"""

#from IPython import get_ipython NO PORTABLE
#get_ipython().magic('reset -sf') NO PORTABLE

import archivos
import matplotlib.pyplot as plt
import filtros
import graficas
import copy
import time

##############################################################################

if __name__ == '__main__':
    
    
    plt.close('all')
    
    nidCentral = 7    

    parque = archivos.leerArchivosCentral(nidCentral)
    medidor = parque.medidores[0]
    parque.registrar()
    filtros = medidor.get_filtros()
    archivos.guardarCentral(parque)

    med1 = medidor.get_medida('vel','pronos')
    med2 = parque.pot


    vel_SCADA = parque.medidores[0].get_medida('vel','scada')
    vel_pronos= parque.medidores[0].get_medida('vel','pronos')

    dir_SCADA = parque.medidores[0].get_medida('dir','scada')
    dir_pronos= parque.medidores[0].get_medida('dir','pronos')
    consigna = parque.cgm

    pot_scada = parque.pot
    
    meds = list()
    
    meds.append(vel_SCADA)
    meds.append(vel_pronos)

    meds.append(dir_SCADA)
    meds.append(dir_pronos)

    meds.append(pot_scada)
    meds.append(consigna)

    graficas.clickplot(meds)
    plt.show()


    
