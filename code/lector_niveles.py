# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:18:16 2020

@author: fpalacio
"""

import numpy as np

cant_niveles = 8

patamares = []
patamares.append('Leve')
patamares.append('Media')
patamares.append('Pesada')

nombre_sala = 'ExpoA2020Sem5'
path_sala = 'C:/simsee/corridas/' + nombre_sala + '/'

cme_niveles = np.zeros((cant_niveles*3,12))

kfila = 0


for knivel in range(cant_niveles):
    
    path_nivel = path_sala + nombre_sala + '_plantilla3_niv' + str(knivel) + '_resultados_/hoja_CME_'
    
    for kpatamar in range(len(patamares)):
        
        path = path_nivel + patamares[kpatamar] + '.xlt'     

        a = np.loadtxt(path, skiprows = 2)
        
        cme_pe = a[-1, 2:]
        
        cme_niveles[kfila:] = cme_pe
        
        kfila = kfila + 1
        
        np.savetxt(path_sala + 'niveles.xlt', cme_niveles, delimiter = '\t', newline = '\n')