# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:32:25 2019

@author: fpalacio
"""
# semilla de noseque de PYthon

import archivos
import matplotlib.pyplot as plt
import gen_series_analisis_serial as seriesAS
import pandas as pd
import datos
import graficas
import filtros 
import datetime
import copy
import math
import numpy as np
import modelo_RNN as modelo
import plot_scatter as pltxy
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

    dt_ini_calc, dt_fin_calc = archivos.leer_ro_pendientes(parque2.id)
    carpeta_central = archivos.path(parque2.id)
    delta_print_datos = 200

    if flg_estimar_RO:
        
        carpeta_central = archivos.path(parque2.id)
    
        t, M, F, nom_series = seriesAS.gen_series_analisis_serial(parque1, parque2,
                                                     nom_series_p1, nom_series_p2)    
        
        pltxy.plot_meds(M,F,nom_series,'velGEN_9','potSCADA_57')


    
        # creo df donde voy a guardar los resultados de las RO
        columns_ro = ['dt_ini', 'dt_fin', 'Estimacion [MWh]', 'Error_PE_70% [MWh]',
                      'Error_PE_30% [MWh]', 'Error_VE [MWh]', 'Delta Error VE - PE70 [MWh]',
                      'EG [MWh]', 'ENS VE [MWh]', 'ENS PE_70 [MWh]']
            
        df_ro = pd.DataFrame(columns=columns_ro)    
    
    
        # Normalizo datos 
        
        M_n, max_pot, min_pot = modelo.normalizar_datos(M, F, t, nom_series)
                           
        pot = M[:,-1]
       
        # inicializo la pot igual a la real, luego relleno huecos
        pot_estimada = copy.copy(pot)
        pot_estimada_PE70 = copy.copy(pot)
        
        # Busco secuencias de patrones que quiero calcular su pot
        
        delta = 5 # agrega delta datos 10min antes y despues de las RO encontradas
        Pats_Data_n, Pats_Filt, Pats_Calc, dtini_ro, dtfin_ro, kini_ro = \
            modelo.patrones_ro(delta, F, M_n, t, dt_ini_calc, dt_fin_calc)       
               
                
        #for kRO in  range(len(Pats_Data_n)): #range(7,8):
        for kRO in range(10,15):
     
            carpeta_ro = archivos.path_ro( kRO + 1, carpeta_central)
            
            
            print(f"Calculando RO {kRO+1} de {len(Pats_Data_n)}")
            
            X_n,y_n = seriesAS.split_sequences_patrones(F, M_n, Pats_Data_n[kRO],
                                                 Pats_Filt[kRO], Pats_Calc[kRO])
    
            kini_RO = kini_ro[kRO] 
            X_RO_n = Pats_Data_n[kRO][~Pats_Filt[kRO]].flatten()
            X_RO_n = np.asmatrix(X_RO_n)  
            
            train_pu = 0.7
            y_test_e, y_test, y_train_e, y_train, y_RO_e = \
                modelo.estimar_ro(train_pu, X_n, y_n, X_RO_n, carpeta_ro)
    
            # desnormalizo series
                
            datos_norm = [y_RO_e, y_test_e, y_train_e, y_test, y_train]
            [y_RO_e, y_test_e, y_train_e, y_test, y_train] = \
                modelo.desnormalizar_datos(datos_norm, min_pot, max_pot)
            
                    
            # concateno (train + test) salidas del modelo y datos reales 
            y_e_all = np.concatenate((y_test_e, y_train_e), axis=0)
            y_all = np.concatenate((y_test, y_train), axis=0)
            
            
            pot_estimada[kini_RO:kini_RO+y_RO_e.size] = y_RO_e        
            
            E_gen_RO = sum(pot[kini_RO:kini_RO+y_RO_e.size])/6
                   
            y_e_all_acum = np.sum(y_e_all, axis = 1)
            y_e_all_acum_MWh = y_e_all_acum/6
    
            y_all_acum = np.sum(y_all, axis = 1)
            y_all_acum_MWh = y_all_acum/6
            
            y_dif_all_acum_MWh = np.subtract(y_e_all_acum_MWh, y_all_acum_MWh)        
            
            error_pu = np.mean(y_dif_all_acum_MWh)/np.mean(y_all_acum_MWh)
            
            print('Error medio [p.u] = ' , error_pu)
            
            std_pu = np.std(y_dif_all_acum_MWh)/np.mean(y_all_acum_MWh)
    
            print('std [p.u] = ' , std_pu)
    
            b_v = (error_pu ** 2 + std_pu ** 2) ** (1/2)
            
            print('bias-variance [p.u] = ', b_v)
    
            # calculo las dist empíricas para cada rango de ptos
            sort = np.argsort(y_e_all_acum_MWh)
            y_e_all_acum_MWh_sort = np.array(y_e_all_acum_MWh)[sort]
            y_dif_acum_MWh_sort = np.array(y_dif_all_acum_MWh)[sort]
            
            NDatos_hist = 300
            delta_datos = math.trunc(NDatos_hist/2)
            
            y_dif_acum_MWh_sort_PE70 = np.zeros(len(y_dif_acum_MWh_sort))
            y_dif_acum_MWh_sort_PE30 = np.zeros(len(y_dif_acum_MWh_sort))
            y_dif_acum_MWh_sort_PE50 = np.zeros(len(y_dif_acum_MWh_sort))
            y_dif_acum_MWh_sort_VE = np.zeros(len(y_dif_acum_MWh_sort))
            
            for k in range(len(y_e_all_acum_MWh_sort)):
                idx_izq = max(0, k-delta_datos) 
                idx_der = min(len(y_e_all_acum_MWh_sort), k+delta_datos)
                
                y_dif_delta = y_dif_acum_MWh_sort[idx_izq:idx_der]
                
                y_dif_acum_MWh_sort_PE70[k] = np.quantile(y_dif_delta, 0.3) 
                y_dif_acum_MWh_sort_PE30[k] = np.quantile(y_dif_delta, 0.7)
                y_dif_acum_MWh_sort_PE50[k] = np.quantile(y_dif_delta, 0.5)
                y_dif_acum_MWh_sort_VE[k] = np.mean(y_dif_delta)
            
            
            
            E_est_MWh = np.sum(y_RO_e)/6        
            E_dif_MWh_PE70 = np.interp(E_est_MWh,y_e_all_acum_MWh_sort
                                       , y_dif_acum_MWh_sort_PE70)      
            E_dif_MWh_PE30 = np.interp(E_est_MWh,y_e_all_acum_MWh_sort
                                       , y_dif_acum_MWh_sort_PE30)
            E_dif_MWh_VE = np.interp(E_est_MWh,y_e_all_acum_MWh_sort
                                       , y_dif_acum_MWh_sort_VE)
                    
            ENS_VE = max(E_est_MWh-E_gen_RO,0)
            
            delta_70 = np.abs(E_dif_MWh_VE-E_dif_MWh_PE70)
    
            E_est_MWh_PE70 = E_est_MWh - delta_70
            
            ENS_PE_70 = max(E_est_MWh_PE70-E_gen_RO,0)
            
            pPE70 = y_RO_e * E_est_MWh_PE70 / E_est_MWh
            
            # Esta función topea la potencia y redistribuye el resto en las horas
            # en las que no aplica el tope
            
            # aca tengo que hacer una funcion que distribuya ENS_PE_70 de modo 
            # que cuando sume (PE_70_calc - PGen) =  ENS_PE_70        
            
            
            '''
            PAut = parque2.PAutorizada 
            
            filt_mayor_PAut = (pPE70 >= PAut)
            
            while np.any(filt_mayor_PAut):
                recorte =np.sum(pPE70[filt_mayor_PAut]) 
                pPE70[filt_mayor_PAut] = PAut
                E_sin_recorte = np.sum(pPE70[~filt_mayor_PAut])
                factor = (E_sin_recorte + recorte)/E_sin_recorte
                pPE70[~filt_mayor_PAut] = pPE70[~filt_mayor_PAut] * factor
                filt_mayor_PAut = pPE70 >= PAut
            '''
            
            pot_estimada_PE70[kini_RO:kini_RO+y_RO_e.size] = pPE70
            
            plt.figure()
            plt.scatter(y_e_all_acum_MWh_sort,y_dif_all_acum_MWh, marker = '.',
                        color=(0,0,0,0.1), label = 'Datos')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE70, label = 'PE70')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE50, label = 'PE50')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_PE30, label = 'PE30')
            plt.plot(y_e_all_acum_MWh_sort, y_dif_acum_MWh_sort_VE, label = 'VE')
    
            plt.axvline(E_est_MWh, color='k', linestyle='--', label = 'ENS_estimada')
            plt.legend()
            plt.grid()
            plt.xlabel('E_modelo [MWh]')
            plt.ylabel('E_dif [MWh]')        
            #plt.show()
            plt.savefig(carpeta_ro + 'errores.png')
    
    
            archi_ro = carpeta_ro + 'resumen.txt'
    
            calculos_ro = [dtini_ro[kRO], dtfin_ro[kRO], E_est_MWh, E_dif_MWh_PE70,
                           E_dif_MWh_PE30, E_dif_MWh_VE, delta_70, E_gen_RO, 
                           ENS_VE, ENS_PE_70]
            s = pd.Series(calculos_ro, index=columns_ro)
            s.to_csv(archi_ro, index=True, sep='\t') 
            
            df_ro = df_ro.append(s,ignore_index=True)
            
    
        # guardo resumen RO
        
        df_ro.to_csv(carpeta_central + 'resumen.txt', index=True, sep='\t',
                     float_format='%.2f') 
    
            
        # creo la potencia estimada
        tipoDato = 'pot'
        nrep = filtros.Nrep(tipoDato)
        pot_p2_mod = datos.Medida('estimacion',pot_estimada , t,
                               tipoDato,'pot_estimada', parque2.pot.minval, parque2.pot.maxval, nrep)     
        # potencia 10min con probabilidad 70% de ser excedida
        pot_p2_mod_PE70 = datos.Medida('estimacion',pot_estimada_PE70 , t,
                               tipoDato,'pot_estimada_PE_70', parque2.pot.minval, parque2.pot.maxval, nrep)     

   

    pot_scada_p1 = parque1.pot
    pot_scada_p2 = parque2.pot
    cgm_p2 = parque2.cgm
    
    meds = list()
    
    meds.append(pot_scada_p2)
    #meds.append(pot_scada_p1)
    meds.append(cgm_p2)
    
    if flg_estimar_RO:
        meds.append(pot_p2_mod)
        meds.append(pot_p2_mod_PE70)

    
    meds.append(vel_GEN_p1)
    meds.append(vel_scada_p1)

    #meds.append(vel_GEN_7)
    #meds.append(vel_SCADA_7)
    
    #meds.append(vel_PRONOS_7)
    
    meds.append(dir_PRONOS_p2)
    meds.append(dir_pronos_p1)
    meds.append(dir_scada_p1)
    

    graficas.clickplot(meds)
    #plt.show()  
    
    # Guardo capturas de pantalla de los datos y estimación de todas las RO

     #range(7,8):

    if flg_estimar_RO:
        for kRO in range(10,15):
        #for kRO in range(len(Pats_Data_n)):            
            dtini_w = dtini_ro[kRO] - datetime.timedelta(minutes=delta_print_datos)
            dtfin_w = dtfin_ro[kRO] + datetime.timedelta(minutes=delta_print_datos)
            
            graficas.window = [dtini_w, dtfin_w]
            
            graficas.clickplot_redraw()
            
            carpeta_ro = archivos.path_ro(kRO+1, carpeta_central)
            plt.savefig(carpeta_ro + 'datos.png')
    else:
        for kcalc in range(len(dt_ini_calc)):
        #for kRO in range(len(Pats_Data_n)):            
            dtini_w = dt_ini_calc[kcalc] - datetime.timedelta(minutes=delta_print_datos)
            dtfin_w = dt_fin_calc[kcalc] + datetime.timedelta(minutes=delta_print_datos)
            
            graficas.window = [dtini_w, dtfin_w]
            
            graficas.clickplot_redraw()
            
            carpeta_datos = archivos.path_carpeta_datos(carpeta_central) 
            plt.savefig(carpeta_datos + str(kcalc) + '.png' )
        
