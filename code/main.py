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

##############################################################################

if __name__ == '__main__':
    plt.close('all')
    
    nidCentral = 5    
    #med_10min, med_15min = archivos.leerArchiSMEC(nidCentral)
    parque = archivos.leerArchi(nidCentral,'scada')
    parque2 = archivos.leerArchi(nidCentral,'gen') 
    medidor_pronos10min = archivos.leerArchiPRONOS(nidCentral,10)
    medidor_pronos60min = archivos.leerArchiPRONOS(nidCentral,60)
    
    #parque.pot_SMEC  = med_10min
    
    vel_SCADA = parque.medidores[0].get_medida('vel')
    dir_SCADA = parque.medidores[0].get_medida('dir')
    vel_GEN = parque2.medidores[0].get_medida('vel')
    dir_GEN = parque2.medidores[0].get_medida('dir')
    
    vel_pronos10min = medidor_pronos10min.get_medida('vel')
    dir_pronos10min = medidor_pronos10min.get_medida('dir')
    
#    vel_pronos60min = medidor_pronos60min.get_medida('vel')
#    dir_pronos60min = medidor_pronos60min.get_medida('dir')
    
    meds = []

    #corr_vel_vel_max = filtros.corrMAX_Ndesf(vel_SCADA,vel_GEN,-20,-15,True)
    
    #corr_dirSCADA_dirPronos_max = filtros.corrMAX_Ndesf(dir_SCADA,dir_pronos10min,-8,-8,True)

    corr_dirSCADA_dirGen_max = filtros.corrMAX_Ndesf(dir_SCADA,dir_GEN,-20,-15,True)        
    
        
    meds.append(corr_dirSCADA_dirGen_max)
    
    #decorr = parque.decorrelacion()
    #for v in decorr.values():
    #    meds.append(v)
    
    #meds.append(parque.pot)

    #meds.append(parque.cgm)
    
    #meds.append(vel_SCADA)
    #meds.append(vel_pronos10min)
    #meds.append(vel_GEN)
    
    meds.append(dir_SCADA)
    meds.append(dir_GEN)
    #meds.append(dir_pronos10min)
    #meds.append(dir_pronos60min)    
    
    #meds.append(med_10min)
    #meds.append(med_15min)
   
    graficas.clickplot(meds)
    plt.show()

##############################################################################

