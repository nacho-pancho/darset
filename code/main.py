#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:57:26 2019
"""

##############################################################################

import archivos
import matplotlib.pyplot as plt
import filtros
import graficas
import copy

##############################################################################

if __name__ == '__main__':
    plt.close('all')
    
    nidCentral = 93    
    med_10min, med_15min = archivos.leerArchiSMEC(nidCentral)
    parque = archivos.leerArchi(nidCentral,'scada')
    parque2 = archivos.leerArchi(nidCentral,'gen') 

    medidor_pronos10min = archivos.leerArchiPRONOS(nidCentral,10)
    pot_SCADA = parque.pot
    
    rad_SCADA = parque.medidores[0].get_medida('rad','scada')
    tem_SCADA = parque.medidores[0].get_medida('tem','scada')
    rad_GEN = parque2.medidores[0].get_medida('rad','gen')
    tem_GEN = parque2.medidores[0].get_medida('tem','gen')

    meds = []
    meds.append(pot_SCADA)
    meds.append(rad_SCADA)
    meds.append(tem_SCADA)
    meds.append(rad_GEN)
    meds.append(tem_GEN)    
    graficas.clickplot(meds)
    plt.show()

##############################################################################

