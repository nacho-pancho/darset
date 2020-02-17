# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:32:25 2019

@author: fpalacio
"""
# semilla de noseque de PYthon

import archivos

import numpy as np
import modelo_RNN as modelo
import plot_scatter as pltxy
import matplotlib.pyplot as plt
#import mdn



if __name__ == '__main__':

    
    flg_estimar_RO = False

    
    plt.close('all')

    # lectura de los datos del parque1 que es el proporciona al parque2 los 
    # datos meteorológicos para el cálculo de las RO.
    
    # Kiyu    
    parque1 = archivos.leerArchivosCentral(9)
    parque1.registrar() 
    medidor1 = parque1.medidores[0]
    filtros1 = parque1.get_filtros()
    M1, F1, nom1, t1 = parque1.exportar_medidas()
    #nom_series_p1 = ['velGEN','dirGEN','velPRONOS','dirPRONOS','potSCADA']
    nom_series_p1 = ['velGEN_9','cosdirGEN_9','sindirGEN_9']
    vel_GEN_p1 = parque1.medidores[0].get_medida('vel','gen')
    vel_scada_p1 = parque1.medidores[0].get_medida('vel','scada')
    dir_scada_p1 = parque1.medidores[0].get_medida('dir','scada')
    dir_pronos_p1 = parque1.medidores[0].get_medida('dir','pronos')
    meds_plot_p1 = [vel_GEN_p1, vel_scada_p1, dir_scada_p1, dir_pronos_p1]

    # lectura de los datos del parque2 al cual se le van a calcular las RO.
    # 18 de Julio
    parque2 = archivos.leerArchivosCentral(57)
    parque2.registrar()
    medidor2 = parque2.medidores[0]
    filtros2 = parque2.get_filtros()
    M2, F2, nom2, t2 = parque2.exportar_medidas()
    #nom_series_p2 = ['velPRONOS','dirPRONOS','potSCADA']
    #nom_series_p2 = ['velGEN','potSCADA']
    nom_series_p2 = ['cosdirPRONOS_57','sindirPRONOS_57','potSCADA_57']
    vel_PRONOS_p2 = parque2.medidores[0].get_medida('vel','pronos')
    vel_GEN_p2 = parque2.medidores[0].get_medida('vel','gen')
    vel_SCADA_p2 = parque2.medidores[0].get_medida('vel','scada')
    dir_PRONOS_p2 = parque2.medidores[0].get_medida('dir','pronos')
    meds_plot_p2 = [vel_PRONOS_p2, dir_PRONOS_p2, parque2.pot,
                    parque2.cgm]

    dt_ini_calc, dt_fin_calc = archivos.leer_ro_pendientes(parque2.id)
    delta_print_datos = 200


    modelo.main_ro(flg_estimar_RO, parque1, parque2, nom_series_p1, nom_series_p2, 
                   dt_ini_calc, dt_fin_calc, delta_print_datos, meds_plot_p1,
                   meds_plot_p2)
    
    
    
    