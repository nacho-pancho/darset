# -*- coding: utf-8 -*-
"""
En este archivo se implementan lo distintos filtros, es decir,
los distintos métodos para detectar que una medida dada es anómala
o no.

Created on Thu May  2 14:55:56 2019

@author: fpalacio
"""
##############################################################################

import numpy
import datos
import numpy as np
import scipy.stats as stats
import archivos as arch




##############################################################################

def str_to_tipo(s):
    if s.find('Var(vel)') != -1:
        return None
    elif s.find('vel') != -1:
        return 'vel'
    elif s.find('dir') != -1:
        return 'dir'
    elif s.find('pot') != -1:
        return 'pot'
    elif s.find('tem') != -1:
        return 'tem'
    elif s.find('pre') != -1:
        return 'pre'
    elif s.find('hum') != -1:
        return 'hum'
    elif s.find('cgm') != -1:
        return 'cgm'    
    elif s.find('dis') != -1:
        return 'dis'
    elif s.find('corr_sp') != -1:
        return 'corr_sp'
    elif s.find('rad_max') != -1:
        return 'rad_max'
    elif s.find('rad') != -1:
        return 'rad'
    elif s.find('ro') != -1:
        return 'ro'
    elif s.find('turb') != -1:
        return 'turb'        
    else:
        return None

##############################################################################
# este dato debería ser parte de Medida
def min_max(tipo,PotAut,NMolinos):
    if tipo == 'vel':
        return [0,25]
    elif tipo == 'dir':
        return [0,360]
    elif tipo == 'pot':
        return [0,PotAut]
    elif tipo == 'tem':
        return [0,80]
    elif tipo == 'pre':
        return [0,1800]
    elif tipo == 'hum':
        return [0,100]
    elif tipo == 'cgm':
        return [0,PotAut]
    elif tipo == 'dis':
        return [0,NMolinos]
    elif tipo == 'corr_sp':
        return [-1,1]
    elif tipo == 'rad' or tipo == 'rad_max':
        return [-10,1400] 
    elif tipo == 'ro':
        return [1.05,1.3]
    elif tipo == 'turb':
        return [0,0.1]
    elif (tipo == 'cosdir') or (tipo == 'sindir'):
        return [-1,1]



##############################################################################

def Nrep(tipo):
    if tipo == 'cgm' or tipo == 'dis':
        return None
    elif tipo == 'pot':
        return 100
    else:
        return 3

##############################################################################

def filtrar_rango(v,min_v,max_v):
    filtro = numpy.zeros(len(v), dtype=bool)
    tol = 0.05
    for i in range(len(v)):
        if v[i]>max_v*(1+tol) or v[i]<min_v*(1-tol):
            filtro[i] = True    
    return filtro 
        
##############################################################################
    

def filtrar_rad (med,ubicacion):
    
    '''    
    zona = ubicacion.zona
    huso = ubicacion.huso
    x = ubicacion.x
    y = ubicacion.y

    zona_ = str(huso) + zona

    myProj = Proj(proj='utm', ellps='WGS84', zone = zona_, south=True)
    lat,long = myProj(x,y,inverse=True)
    
    print(f'Latitud = {lat}, Longitud = {long}')

    loc_pv = pv.location.Location(lat,long)   
    
    times = pd.DatetimeIndex(start = med.tiempo[0], periods=len(med.muestras), freq='10min')
    
    df_med = pd.DataFrame(med.muestras, index = times, columns =['rad']) 
    print(times)
    
    ax = df_med.plot()


    cs = loc_pv.get_clearsky(times)  # ineichen with climatology table by default
    
    df_ghi = cs['ghi']

    df_ghi.plot(ax=ax)
    
    plt.ylabel('Irradiance $W/m^2$');
    
    plt.title('Ineichen, climatological turbidity');

    filtro = numpy.zeros(len(med.muestras), dtype=bool)
    return filtro 
    ''' 
    return None
 

     
##############################################################################

def filtrar_rep(v,filtro_huecos,nRep):
 
    filtro = numpy.zeros(len(v), dtype=bool)
    if nRep is None:
        return filtro
    
    k1 = 1
    cnt_total = 0
    cnt = 0
    buscando = True

    while buscando and (k1 < len(filtro_huecos)):
        if filtro_huecos[k1]:
            k1 = k1 + 1
        else:
            buscando = False
    
    if buscando:
        return filtro

    vant = v[k1]
    k = k1 + 1
    while k < len(v):
        if ((not filtro_huecos[k]) and (np.abs(v[k]-vant) < 0.01)):
            cnt = cnt + 1
            if cnt >= nRep:
                while k1 <= k:
                    filtro[k1] = True
                    k1 = k1 + 1
        else:
            vant = v[k]
            k1 = k
            if cnt >= nRep:
                cnt_total = cnt_total + cnt
            cnt = 0
        
        k = k + 1

    return filtro     

##############################################################################
    
def corr_medidas(x,y,NDatosCorr,NDatosDesf,addFiltro_y):
    """
    Las medidas deben estar sincronizadas
    """
    if x.tiempo[0] != y.tiempo[0]:
        print('ERROR: medidas deben estar registradas para calcular correlacion!')
        return None

    print(x.nombre,x.tiempo[:3])
    print(y.nombre,y.tiempo[:3])    

    if ((x.tipo == 'dir') and (y.tipo == 'dir')):
        flg_dir_dir = True
    else:
        flg_dir_dir = False    
    
    filtro_x = x.filtrada()
    filtro_y = y.filtrada() 
    
    filtro_total = filtro_x | filtro_y
           
    idx_mask = np.where(filtro_total < 1)
    
    x_m = x.muestras
    y_m = y.muestras

    if not flg_dir_dir and np.sum(idx_mask)>0:
        x_m_mask = x_m[idx_mask]
        y_m_mask = y_m[idx_mask]
        x_m_mask_u = stats.rankdata(x_m_mask, "average")
        y_m_mask_u = stats.rankdata(y_m_mask, "average")
        
        x_m_mask_u = x_m_mask_u / np.max(x_m_mask_u)
        y_m_mask_u = y_m_mask_u / np.max(y_m_mask_u)
               
        x_m_u = np.zeros(len(x_m))
        y_m_u = np.zeros(len(y_m))
        
        x_m_u [idx_mask] = x_m_mask_u 
        y_m_u [idx_mask] = y_m_mask_u

    idx_buff = np.zeros(NDatosCorr,dtype=int)
    corr = np.zeros(len(x.muestras))
    k_idx_buff = 0
    k = 0

    while k < len(x_m):
        if not filtro_total[k]:
            if k_idx_buff < NDatosCorr:
                idx_buff[k_idx_buff] = k
                k_idx_buff = k_idx_buff + 1
                k = k + 1
                continue
            else:
                idx_buff_aux = idx_buff
                idx_buff[1:NDatosCorr-1] = idx_buff_aux[0:NDatosCorr-2]
                idx_buff[0] = k

            if flg_dir_dir:
                a1 = x_m[idx_buff]
                a2 = y_m[idx_buff]
                
                dif_ang_deg = np.add(a1,-a2)
                dif_ang_deg = np.abs(dif_ang_deg)

                idx_mayor180 = np.where(dif_ang_deg > 180)
                dif_ang_deg[idx_mayor180] = 360 - dif_ang_deg[idx_mayor180] 

                dang = np.mean(dif_ang_deg)/180
                corr[k] = 1 - dang
            else:     
                corr[k] = np.dot(x_m_u[idx_buff], y_m_u [idx_buff]) / \
                    (np.linalg.norm(x_m_u[idx_buff]) * np.linalg.norm(y_m_u[idx_buff]))

        else:
            corr[k] = corr[k-1]
        k = k + 1

    if addFiltro_y:
        nombre_f = 'corr_' + x.tipo + '_' + x.procedencia
        y.agregar_filtro(nombre_f, corr < 0.9 )
    
    return datos.Medida('corr', corr, list(y.tiempo), 'corr', 'corr_' + x.tipo + '_'
                        + y.tipo, np.mean(corr) * 0.99, 1.0, 0)


##############################################################################
    

def corrMAX_Ndesf(x,y,NdesfMin,NdesfMax,corregirDesf,desf_dinamico,flg_graficar):
    
    rango = range(NdesfMin,NdesfMax)
    
    corr_mat = np.zeros((len(rango),len(y.muestras)))
    
    corr_max_med = datos.FUERA_DE_RANGO
    Ndesf_max_med = datos.FUERA_DE_RANGO
    
    fila = 0    
    for Ndesf in rango:
        corr_x_y,corr = corr_medidas(x,y,24*6*7,Ndesf,False)
        corr_mat[fila,:] =corr_x_y.muestras 
        fila = fila  + 1
        
        if corr > corr_max_med:
            corr_max_med = corr
            Ndesf_max_med = Ndesf
            
        
    Ndesf_opt_k = [ rango[x] for x in np.argmax(corr_mat,axis = 0)]
    
    
    print(Ndesf_opt_k)
    
                
    if corregirDesf:
        if desf_dinamico:
            y.desfasar_dinamico(Ndesf_opt_k)
            y.nombre = y.nombre + '_desf_din' 
        else:
            y.desfasar(Ndesf_max_med)
            y.nombre = y.nombre + '_desf_' + str(Ndesf_max_med)
       
        #Ndesf_opt_k = [ d - Ndesf_P50 for d in Ndesf_opt_k]

    print ('desfasaje('+ x.nombre + ',' + y.nombre + ') = ', Ndesf_max_med , ' muestras')   

    print(corr_mat)     
    
    Ndesf_opt_k = datos.Medida('corr', Ndesf_opt_k, y.tiempo, 'Ndesf_opt_k', 'Ndesf_opt_k_' + x.tipo + '_' + y.tipo, -NdesfMin, NdesfMax, 0)
    
    '''
    if flg_graficar:
        meds = []
        meds.append(x)
        meds.append(y)
        meds.append(Ndesf_opt_k)
        
        graficas.clickplot(meds)
    
    '''    
    
    return Ndesf_opt_k


def corregir_vel_altura (velRef,velAjustar):
    return None
    

