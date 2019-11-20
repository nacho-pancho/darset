#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:57:26 2019
"""

##############################################################################

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
    med_10min, med_15min = archivos.leerArchiSMEC(nidCentral)
    parque = archivos.leerArchi(nidCentral,'scada')

    medidor_pronos10min = archivos.leerArchiPRONOS(nidCentral,10)
    pot_SCADA = parque.pot
        
    vel_SCADA = parque.medidores[0].get_medida('vel','scada')
    dir_SCADA = parque.medidores[0].get_medida('dir','scada')

    # no está el archivo!    
    #parque2 = archivos.leerArchi(nidCentral,'gen') 
    #vel_GEN = parque2.medidores[0].get_medida('vel','gen')
    #vel_GEN_desf = copy.deepcopy(vel_GEN)
    
    #dir_GEN = parque2.medidores[0].get_medida('dir','gen')
    
    #pot_GEN = parque2.pot
    #pot_GEN_desf = copy.deepcopy(pot_GEN)
    
    meds = []
    
    NDesfOpt_potGEN = filtros.corrMAX_Ndesf(pot_SCADA,pot_GEN_desf,-18,5,True,True)
    
    
    meds.append(pot_SCADA)
    #meds.append(pot_GEN_desf)
    meds.append(NDesfOpt_potGEN)

    
    graficas.clickplot(meds)
    plt.show()

##############################################################################

