# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:32:25 2019

@author: fpalacio
"""
import archivos
import matplotlib.pyplot as plt
import numpy as np
import gen_series_analisis_serial as seriesAS
import pandas as pd

if __name__ == '__main__':
    
    plt.close('all')
    

    parque1 = archivos.leerArchivosCentral(5)
    parque1.registrar()
    medidor1 = parque1.medidores[0]
    filtros1 = parque1.get_filtros()
    M1, F1, nom1, t1 = parque1.exportar_medidas()
    nom_series_p1 = ['potSCADA','velSCADA']

    parque2 = archivos.leerArchivosCentral(7)
    parque2.registrar()
    medidor2 = parque2.medidores[0]
    filtros2 = parque2.get_filtros()
    M2, F2, nom2, t2 = parque2.exportar_medidas()
    nom_series_p2 = ['potSCADA','velSCADA','dirSCADA','velPRONOS','dirPRONOS']
    
    t, M, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                           nom_series_p1,nom_series_p2)    
    

    
    # creo data frame con datos    
    df = pd.DataFrame(M, index = t, columns = nom_series) 
    
    #df.tail()
    
    #Filtro valores menores a cero
    dataCerroG = dCerroG[(dCerroG >= 0).all(axis=1)]
    
    seriesAS. 