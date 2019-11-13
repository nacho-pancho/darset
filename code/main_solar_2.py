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
    
    nidCentral = 93    

    parque = archivos.leerArchivosCentral(nidCentral)
    
    parque.calcular_filtros()
    
    rad_SCADA = parque.medidores[0].get_medida('rad','scada')
    rad_pronos = parque.medidores[0].get_medida('rad','pronos')
    
    tem_SCADA = parque.medidores[0].get_medida('tem','scada')
    tem_pronos = parque.medidores[0].get_medida('tem','pronos')    
    
    pot_scada = parque.pot
    pot_smec = parque.pot_SMEC
 
    
    meds = []
    
    meds.append(rad_SCADA)
    meds.append(rad_pronos)
    
    meds.append(tem_SCADA)
    meds.append(tem_pronos)
    
    meds.append(pot_scada)
    meds.append(pot_smec)
    
    graficas.clickplot(meds)
    plt.show()
    