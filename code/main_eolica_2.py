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

##############################################################################

if __name__ == '__main__':
    
    
    plt.close('all')
    
    nidCentral = 5    

    parque = archivos.leerArchivosCentral(nidCentral)
    
    parque.calcular_filtros()
    
    vel_SCADA = parque.medidores[0].get_medida('vel','scada')
    vel_pronos= parque.medidores[0].get_medida('vel','pronos')
    
    meds = []
    
    meds.append(vel_SCADA)
    meds.append(vel_pronos)
    
    graficas.clickplot(meds)
    plt.show()
    
    
