#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 03:44:51 2019

@author: felipep
"""

#from IPython import get_ipython NO PORTABLE
#get_ipython().magic('reset -sf') NO PORTABLE

import archivos
import matplotlib.pyplot as plt
import graficas
import numpy as np

##############################################################################

if __name__ == '__main__':
    
    
    plt.close('all')
    
    nidCentral = 7 

    parque = archivos.leerArchivosCentral(nidCentral)
    parque.registrar()
    medidor = parque.medidores[0]
    filtros = parque.get_filtros()
    M,F,nombres,t = parque.exportar_medidas()
    #np.savetxt('M7.ascii',M,fmt='%7e')
    #np.savetxt('F7.ascii',F,fmt='%d')
    np.savez_compressed('M7.npz',M)
    np.savez_compressed('F7.npz',F)
    fn = open("n7.txt","w")
    for n in nombres:
        print(f"{n}",file=fn,end='\t')
    print(file=fn)
    fn.close()

    #archivos.guardarCentral(parque)

    vel_SCADA = parque.medidores[0].get_medida('vel','scada')
    vel_pronos= parque.medidores[0].get_medida('vel','pronos')
    vel_gen= parque.medidores[0].get_medida('vel','gen')

    dir_SCADA = parque.medidores[0].get_medida('dir','scada')
    dir_pronos= parque.medidores[0].get_medida('dir','pronos')
    dir_gen= parque.medidores[0].get_medida('dir','gen')
    consigna = parque.cgm

    pot_scada = parque.pot
    
    meds = list()
    
    meds.append(vel_SCADA)
    meds.append(vel_pronos)
    meds.append(vel_gen)

    meds.append(dir_SCADA)
    meds.append(dir_pronos)
    meds.append(dir_gen)

    meds.append(pot_scada)
    meds.append(consigna)
    

    graficas.clickplot(meds)
    plt.show()


    
