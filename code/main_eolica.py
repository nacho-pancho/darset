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
    parque2 = archivos.leerArchi(nidCentral,'gen') 

    medidor_pronos10min = archivos.leerArchiPRONOS(nidCentral,10)
    #medidor_pronos60min = archivos.leerArchiPRONOS(nidCentral,60)
    pot_SCADA = parque.pot
    #parque.pot_SMEC  = med_10min
        
    vel_SCADA = parque.medidores[0].get_medida('vel')
    dir_SCADA = parque.medidores[0].get_medida('dir')
    
    
    vel_GEN = parque2.medidores[0].get_medida('vel')
    vel_GEN_desf = copy.deepcopy(vel_GEN)
    
    dir_GEN = parque2.medidores[0].get_medida('dir')
    
    pot_GEN = parque2.pot
    pot_GEN_desf = copy.deepcopy(pot_GEN)
    
    
    
    #vel_pronos10min = medidor_pronos10min.get_medida('vel')
    #vel_pronos10min_desf = copy.deepcopy(vel_pronos10min)
    #dir_pronos10min = medidor_pronos10min.get_medida('dir')
    #dir_pronos10min_desf = copy.deepcopy(dir_pronos10min)


    
    meds = []
    
    
    #NDesfOpt_velGEN = filtros.corrMAX_Ndesf(vel_SCADA,vel_GEN_desf,-5,5,True,True)
    # aca en realidad tengo que calcular el optimo descartando datos con consigna
    
    NDesfOpt_potGEN = filtros.corrMAX_Ndesf(pot_SCADA,pot_GEN_desf,-18,5,True,True)
    
    
    meds.append(pot_SCADA)
    #meds.append(pot_GEN)
    meds.append(pot_GEN_desf)
    meds.append(NDesfOpt_potGEN)

    
    #meds.append(parque.pot_SMEC)
    '''
    meds.append(vel_SCADA)
    meds.append(vel_GEN)
    meds.append(vel_GEN_desf)
    meds.append(NDesfOpt_velGEN)
    '''
    #meds.append(vel_pronos10min)
    #meds.append(vel_pronos10min_desf)
    #meds.append(vel_GEN_desf)
    
    #meds.append(dir_SCADA)
    #meds.append(dir_GEN)
    #meds.append(dir_pronos10min)
    #meds.append(dir_pronos10min_desf)
    #meds.append(dir_pronos60min)    
    
    #meds.append(med_10min)
    #meds.append(med_15min)
    
    #meds.append(dir_GEN_desf)
    #meds.append(corr_dirSCADA_dirGen_max)
    #meds.append(corr_dirSCADA_dirPronos_max)
    #meds.append(corr_velSCADA_velPRONOS_max)
    #meds.append(corr_vel_vel_max)
    graficas.clickplot(meds)
    plt.show()

##############################################################################

