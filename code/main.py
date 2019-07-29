# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:57:26 2019
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import rutas as r
import plotGrafs as pltGrfs
import prueba_plot_con_scroll as pltScroll
import matplotlib.pyplot as plt

plt.close('all')

nidCentral = 5    
med_10min, med_15min = r.leerArchiSMEC(nidCentral)
parque = r.leerArchiSCADA(nidCentral) 
vel_SCADA = parque.medidores[0].get_medida('vel')
dir_SCADA = parque.medidores[0].get_medida('dir')

medidor_pronos = r.leerArchiPRONOS(nidCentral)
vel_pronos = medidor_pronos.get_medida('vel')
dir_pronos = medidor_pronos.get_medida('dir')


parque.pot_SMEC  = med_10min

meds = []
#decorr = parque.decorrelacion()
#for v in decorr.values():
#    meds.append(v)

#meds.append(med_10min)
#meds.append(med_15min)

#meds.append(vel_SCADA)
#meds.append(vel_pronos)
#meds.append(parque.pot)
#meds.append(parque.cgm)

#meds.append(vel_SCADA)
#meds.append(vel_pronos)

meds.append(dir_SCADA)
meds.append(dir_pronos)


#pltGrfs.plotMedidas(meds,'False','2018-10-25','2018-10-30',r.path(nidCentral),True)

app = pltScroll.MyApp(meds)
app.MainLoop()