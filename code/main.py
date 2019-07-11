# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:57:26 2019

@author: jfpbf
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import rutas as r
import plotGrafs as pltGrfs
import prueba_plot_con_scroll as pltScroll

nidCentral = 5    
med_10min, med_15min = r.leerArchiSMEC(nidCentral)
parque = r.leerArchiSCADA(nidCentral) 

parque.pot_SMEC  = med_10min

decorr = parque.decorrelacion()

meds = []
#meds.append(med_10min)
#meds.append(med_15min)
vel_SCADA = parque.medidores[0].medidas[0]
meds.append(vel_SCADA)
meds.append(parque.pot)
meds.append(parque.cgm)
meds.append(decorr)

#pltGrfs.plotMedidas(meds,'False','2018-10-25','2018-10-30',r.path(nidCentral),True)

app = pltScroll.MyApp(meds)
app.MainLoop()