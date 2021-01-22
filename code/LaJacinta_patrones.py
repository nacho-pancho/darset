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
import datetime
#import mdn



if __name__ == '__main__':

    
    flg_estimar_RO = True
    

    
    plt.close('all')

    # lectura de los datos del parque1 que es el proporciona al parque2 los 
    # datos meteorológicos para el cálculo de las RO.
    
    
    # Alto Cielo   
    nid_p1 = 42
    parque1 = archivos.leerArchivosCentral(nid_p1) 
    parque1.registrar() 
    medidor1 = parque1.medidores[0]          
    filtros1 = parque1.get_filtros()
    
    M1, F1, nom1, t1 = parque1.exportar_medidas()
    nom_series_p1 = ['radSCADA','temSCADA']
    nom_series_p1 = [s + '_' + str(nid_p1) for s in nom_series_p1]
    rad_SCADA_p1 = parque1.medidores[0].get_medida('rad','scada')
    tem_SCADA_p1 = parque1.medidores[0].get_medida('tem','scada')
    meds_plot_p1 = [rad_SCADA_p1, tem_SCADA_p1, parque1.pot]


    

    # lectura de los datos del parque2 al cual se le van a calcular las RO.
    # La Jacinta
    nid_p2 = 40
    parque2 = archivos.leerArchivosCentral(nid_p2)
    
    tini = datetime.datetime(2019, 8, 17)  
    tfin = datetime.datetime(2019, 8, 26)
    archi = archivos.archi_ro_pendientes(nid_p2)
    parque2.calcular_liq_pendientes(tini, tfin, archi)
    
    parque2.registrar()
    medidor2 = parque2.medidores[0]
    filtros2 = parque2.get_filtros()
    M2, F2, nom2, t2 = parque2.exportar_medidas()
    #nom_series_p2 = ['velPRONOS','dirPRONOS','potSCADA']
    #nom_series_p2 = ['velGEN','potSCADA']
    nom_series_p2 = ['potSCADA', 'cgmSCADA']
    nom_series_p2 = [s + '_' + str(nid_p2) for s in nom_series_p2]
    
    rad_SCADA_p2 = parque2.medidores[0].get_medida('rad','scada')
    tem_SCADA_p2 = parque2.medidores[0].get_medida('tem','scada')
    meds_plot_p2 = [rad_SCADA_p2, tem_SCADA_p2, parque2.pot, parque2.cgm]

    dt_ini_calc, dt_fin_calc = archivos.leer_ro_pendientes(parque2.id)
    delta_print_datos = 500


    modelo.main_ro(flg_estimar_RO, parque1, parque2, nom_series_p1, nom_series_p2, 
                   dt_ini_calc, dt_fin_calc, delta_print_datos, meds_plot_p1,
                   meds_plot_p2, True)
    