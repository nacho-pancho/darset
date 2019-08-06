#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:57:26 2019
"""

##############################################################################

#from IPython import get_ipython NO PORTABLE
#get_ipython().magic('reset -sf') NO PORTABLE

import rutas as r
#import prueba_plot_con_scroll as pltScroll
import matplotlib.pyplot as plt
import filtros as f
import graficas

##############################################################################

if __name__ == '__main__':
    plt.close('all')
    
    nidCentral = 5    
    #med_10min, med_15min = r.leerArchiSMEC(nidCentral)
    parque = r.leerArchiSCADA(nidCentral) 
    medidor_pronos10min = r.leerArchiPRONOS(nidCentral,10)
    medidor_pronos60min = r.leerArchiPRONOS(nidCentral,60)
    
    #parque.pot_SMEC  = med_10min
    
    vel_SCADA = parque.medidores[0].get_medida('vel')
    dir_SCADA = parque.medidores[0].get_medida('dir')
    
    vel_pronos10min = medidor_pronos10min.get_medida('vel')
    dir_pronos10min = medidor_pronos10min.get_medida('dir')
    
#    vel_pronos60min = medidor_pronos60min.get_medida('vel')
#    dir_pronos60min = medidor_pronos60min.get_medida('dir')
    
    meds = []
    
    filtro_total = dir_SCADA.filtrada()

    corr_dir_dir = f.corr_medidas(dir_SCADA,dir_pronos10min,filtro_total,12)
    meds.append(corr_dir_dir)
    
    decorr = parque.decorrelacion()
    for v in decorr.values():
        meds.append(v)
    
    meds.append(parque.pot)
    meds.append(parque.cgm)
    
    meds.append(vel_SCADA)
    meds.append(vel_pronos10min)
    
    meds.append(dir_SCADA)
    meds.append(dir_pronos10min)
    #meds.append(dir_pronos60min)    
    
    #pltGrfs.plotMedidas(meds,'False','2018-10-25','2018-10-30',r.path(nidCentral),True)
    #meds.append(med_10min)
    #meds.append(med_15min)
   
    #app = pltScroll.MyApp(meds)
    #app.MainLoop()
    graficas.clickplot(meds)
    plt.show()

##############################################################################

