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
import filtros
import graficas
import copy

##############################################################################

if __name__ == '__main__':
    
    
    plt.close('all')
    
    nidCentral = 5    

    parque = archivos.leerArchivosCentral(nidCentral)
    
    
