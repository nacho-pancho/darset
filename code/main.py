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

import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname as up
import os 
import rutas as r
import plotGrafs as pltGrfs

nidCentral = 5    
med_10min, med_15min = r.leerArchiSMEC(nidCentral)
parque = r.leerArchiSCADA(nidCentral)    


meds = []
meds.append(med_10min)
meds.append(med_15min)
vel_SCADA = parque.medidores.medidas[0]
meds.append(vel_SCADA)
meds.append(parque.cgm) 

pltGrfs.plotMedidas(meds,'False','2018-10-25','2018-10-30',r.path(nidCentral),True)
