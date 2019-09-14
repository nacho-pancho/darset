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
    
    nidCentral = 11    
    med_10min, med_15min = archivos.leerArchiSMEC(nidCentral)
    parque = archivos.leerArchi(nidCentral,'scada')
    parque2 = archivos.leerArchi(nidCentral,'gen') 

    medidor_pronos10min = archivos.leerArchiPRONOS(nidCentral,10)
    #medidor_pronos60min = archivos.leerArchiPRONOS(nidCentral,60)
    #pot_SCADA = parque.pot
    #parque.pot_SMEC  = med_10min
        
    vel_SCADA = parque.medidores[0].get_medida('vel')
    #dir_SCADA = parque.medidores[0].get_medida('dir')
    
    #rad_SCADA = parque.medidores[0].get_medida('rad')
    #tem_SCADA = parque.medidores[0].get_medida('tem')

    
    vel_GEN = parque2.medidores[0].get_medida('vel')
    #dir_GEN = parque2.medidores[0].get_medida('dir')
    vel_GEN_desf = copy.deepcopy(vel_GEN)
    
    
    vel_pronos10min = medidor_pronos10min.get_medida('vel')
    vel_pronos10min_desf = copy.deepcopy(vel_pronos10min)
    #dir_pronos10min = medidor_pronos10min.get_medida('dir')
    #dir_pronos10min_desf = copy.deepcopy(dir_pronos10min)

    #rad_pronos10min = medidor_pronos10min.get_medida('rad')
    #tem_pronos10min = medidor_pronos10min.get_medida('tem')

    
#    vel_pronos60min = medidor_pronos60min.get_medida('vel')
#    dir_pronos60min = medidor_pronos60min.get_medida('dir')
    
    meds = []
    
    #meds.append(corr_vel_vel_max)
    
    corr_vel_vel_max = filtros.corrMAX_Ndesf(vel_SCADA,vel_GEN_desf,-20,20,True,True)
    meds.append(corr_vel_vel_max)
    
    #corr_vel_pot_max = filtros.corrMAX_Ndesf(parque.pot_SMEC,pot_SCADA,-5,5,True,True)
    #meds.append(corr_vel_pot_max)    
    
    #corr_dirSCADA_dirGen_max = filtros.corrMAX_Ndesf(dir_SCADA,dir_GEN_desf,-20,20,True,True)
    #corr_dirSCADA_dirPronos_max = filtros.corrMAX_Ndesf(dir_SCADA,dir_pronos10min_desf,-20,-15,True,False)

    #corr_dirSCADA_dirGen_max = filtros.corrMAX_Ndesf(dir_SCADA,dir_GEN,-20,-15,True) 

    #corr_velSCADA_velPRONOS_max = filtros.corrMAX_Ndesf(vel_SCADA,vel_pronos10min_desf,-20,-15,True,False)        
    
    #corr_radSCADA_radPRONOS_max,Ndesf_corr_max = filtros.corrMAX_Ndesf(pot_SCADA,rad_pronos10min,-20,20,True)
    #corr_radSCADA_radPRONOS_max,Ndesf_corr_max = filtros.corrMAX_Ndesf(tem_SCADA,tem_pronos10min,-20,20,True)    

    #corr_potSCADA_potSMEC_max,Ndesf_corr_max = filtros.corrMAX_Ndesf(pot_SCADA,med_10min,-20,20,True)

    
    #meds.append(corr_dirSCADA_dirPronos_max)
    
    #decorr = parque.decorrelacion()
    #for v in decorr.values():
    #    meds.append(v)
    
    #meds.append(pot_SCADA)
    #meds.append(rad_SCADA)
    #meds.append(tem_SCADA)

    #meds.append(rad_pronos10min)
    #meds.append(tem_pronos10min)
    
    #meds.append(parque.pot_SMEC)
    
    meds.append(vel_SCADA)
    meds.append(vel_GEN)
    #meds.append(vel_pronos10min)
    #meds.append(vel_pronos10min_desf)
    meds.append(vel_GEN_desf)
    
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
    graficas.clickplot(meds)
    plt.show()

##############################################################################

