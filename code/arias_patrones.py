# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:51:37 2021

@author: jfpbf
"""
import archivos

import numpy as np
import modelo_RNN as modelo
import plot_scatter as pltxy
import matplotlib.pyplot as plt
import datetime
#import mdn

if __name__ == '__main__':
    
    tipo_calc = 'NN'
    #tipo_calc = 'MVLR'
    # tipo_calc = 'MVLR_L'
    #tipo_calc = 'MVLR_R'
        
    flg_estimar_RO = True
    flg_recorte_SMEC = False
    flg_print_datos = False
    
    #tipo_norm = 'Gauss'
    tipo_norm = 'Standard'
    
    plt.close('all')
    
    nid_p1 = 20
    parque1 = archivos.leerArchivosCentral(nid_p1)
    parque1.registrar()
    medidor1 = parque1.medidores[0]
    filtros1 = parque1.get_filtros()
    M1, F1, nom1, t1 = parque1.exportar_medidas()
    #nom_series_p1 = ['velGEN','dirGEN','velPRONOS','dirPRONOS','potSCADA']
    #nom_series_p1 = ['velxSCADA', 'velySCADA']
    nom_series_p1 = ['velxSCADA', 'velySCADA']
    nom_series_p1 = [s + '_' + str(nid_p1) for s in nom_series_p1]
    vel_SCADA_p1 = parque1.medidores[0].get_medida('vel','scada')
    dir_SCADA_p1 = parque1.medidores[0].get_medida('dir','scada')
    vel_GEN_p1 = parque1.medidores[0].get_medida('vel','gen')
    dir_GEN_p1 = parque1.medidores[0].get_medida('dir','gen')    

    meds_plot_p1 = [vel_SCADA_p1, dir_SCADA_p1]


    nid_p2 = 32
    parque2 = archivos.leerArchivosCentral(nid_p2)
    
    tini = datetime.datetime(2021, 4, 1)     
    tfin = datetime.datetime(2021, 5, 1)
    archi = archivos.archi_ro_pendientes(nid_p2)
    parque2.calcular_liq_pendientes(tini, tfin, archi)    
    
    parque2.registrar()
    medidor2 = parque2.medidores[0]
    filtros2 = parque2.get_filtros()
    M2, F2, nom2, t2 = parque2.exportar_medidas()
    #nom_series_p2 = ['velxPRONOS','velyPRONOS','potSCADA']
    nom_series_p2 = ['potSCADA', 'cgmSCADA']
    nom_series_p2 = [s + '_' + str(nid_p2) for s in nom_series_p2]
    #vel_PRONOS_p2 = parque2.medidores[0].get_medida('vel','pronos')
    #dir_PRONOS_p2 = parque2.medidores[0].get_medida('dir','pronos')
    meds_plot_p2 = [parque2.cgm, parque2.pot]

 
    dt_ini_calc, dt_fin_calc = archivos.leer_ro_pendientes(parque2.id)
    delta_print_datos = 150


    modelo.main_ro( tini, tfin, flg_estimar_RO, parque1, parque2, nom_series_p1, nom_series_p2, 
                   dt_ini_calc, dt_fin_calc, delta_print_datos, meds_plot_p1,
                   meds_plot_p2, flg_print_datos, flg_recorte_SMEC, tipo_calc, 
                   tipo_norm )
