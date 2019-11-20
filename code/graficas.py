# -*- coding: utf-8 -*-
#
# Funciones d
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
from windrose import WindroseAxes

#=================================================================================

def rosa_de_los_vientos(self):
    '''
    muestra el resumen de direcciones y velocidades
    del viento en esta medida en todo su período
    como una rosa de los vientos
    '''
    vel = self.get_medida('vel')
    dir_ = self.get_medida('dir')
    
    filtro = vel.filtrada() | dir_.filtrada()
    
    wd = dir_.muestras[filtro < 1]
    ws = vel.muestras[filtro < 1]
    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()

#=================================================================================

MAP_WIDTH = 1200
MAP_WIDTH_ZOOM = 1200
BAR_HEIGHT = 5
DEFAULT_WINDOW_SIZE = datetime.timedelta(days=7)
DEFAULT_TIME_DELTA = datetime.timedelta(minutes=10)
ZOOM_STEP = 1.5

clickfig = None
tini = None
tfin = None
window = None
window_size = DEFAULT_WINDOW_SIZE
medidas = None
tipos = None
viridis = cm.get_cmap('viridis')
tcenter = None
alarm_map = None
alarm_map_zoom = None
map_h = None
imprimir_map_zoom = False

#---------------------------------------------------------------------------------

def clickplot_redraw():
    print('redraw')
    global window, medidas, tipos, tini, tfin, imprimir_map_zoom
    
    if 1: #imprimir_map_zoom:
        NGrafs = len(tipos)+2
    else:
        NGrafs = len(tipos)+1
    

    min_ejes = np.full(NGrafs,99999999)
    max_ejes = np.full(NGrafs,-99999999)
    
    print('figure')
    plt.figure(clickfig.number)
    #print("window:",window)archivos

    print('leyendas')
    legends = dict()
    for tipo in tipos:
        legends.update( {tipo:list()} )
        
        
    for i in range(len(medidas)):
        print('medida',i)
        med_i = medidas[i]        
        idx_tipo = tipos.index(med_i.tipo)
        t_i = med_i.tiempo
        legends[med_i.tipo].append(med_i.nombre)
        idx_i = list(map(lambda t: (t >= window[0]) and (t < window[1]), t_i))
        x_i = list()
        y_i = list()
        for j in range(len(idx_i)):
            if idx_i[j]:
                x_i.append(t_i[j])
                y_i.append(med_i.muestras[j])
        
        if len(x_i) == 0:
            plt.draw()
            continue
        
        #print(x_i[0],x_i[-1])
        plt.subplot(NGrafs,1,idx_tipo+1)
        c_i = viridis(i/len(medidas))
        plt.plot(x_i,y_i,color=c_i)
        
        min_yi = np.min(y_i)
        if  (min_yi > -1000) & (min_yi < min_ejes[idx_tipo]):
            min_ejes[idx_tipo] =  min_yi
        
        max_yi = np.max(y_i) 
        if (max_yi < 1000) & (max_yi > max_ejes[idx_tipo]):
            max_ejes[idx_tipo] =  max_yi           
        
        plt.axis([window[0],window[1],min_ejes[idx_tipo]*0.9,max_ejes[idx_tipo]*1.1])
        plt.ylabel(med_i.tipo)
        plt.draw()
        
    print('leyendas')

    for i in range(len(tipos)):
        plt.subplot( NGrafs, 1, i+1 )
        plt.legend( legends[ tipos[i] ], loc='upper right' )
        plt.grid(True)

    #
    # actualizar el mapa
    #
    print('mapa gral')
    plt.subplot(NGrafs,1,NGrafs)
    j0 = int((window[0]-tini)/(tfin-tini)*MAP_WIDTH)
    j1 = int((window[1]-tini)/(tfin-tini)*MAP_WIDTH)
    fondo = np.copy(alarm_map)
    fondo[:] = 1
    fondo[:,j0:j1,:3] = 0
    plt.imshow(alarm_map)
    plt.imshow(fondo,alpha=0.25)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.draw()
    
    #
    # actualizar el zoom del mapa
    #    
    if 1: #if imprimir_map_zoom:
        print('mapa zoom')
        plt.subplot(NGrafs, 1, NGrafs-1)
        alarm_map_zoom = create_alarm_map (map_h, MAP_WIDTH_ZOOM, medidas, window[0], window[1])        
        plt.imshow(alarm_map_zoom)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.draw()
    plt.tight_layout(pad=1.5)
    print('listo')    
    
    
#---------------------------------------------------------------------------------

def click_event_handler(event):
    '''
    manejo de eventos del mouse sobre una gráfica
    utilizado para el clickplot
    '''
    global window,tcenter
    print('click')
    #
    # capturar pos (x,y)  en la imagen
    #
    ix, iy = event.xdata, event.ydata
    #print(f'click: x = {ix}, y={iy}')
    #
    # mapear a posicion temporal
    #
    tcenter = tini + (ix/MAP_WIDTH)*(tfin-tini)
    w0 = tcenter - window_size/2
    w1 = tcenter + window_size/2
    window = (w0,w1)
    clickplot_redraw()


def scroll_event_handler(event):
    '''
    manejo de eventos del mouse sobre una gráfica
    utilizado para el clickplot
    '''
    print('scroll')
    global window_size,window,tcenter
    step = event.step
    #print(f'scroll: step = {step}')
    #
    # mapear a posicion temporal
    #
    window_size = window_size * (ZOOM_STEP ** step)
    w0 = tcenter - window_size/2
    w1 = tcenter + window_size/2
    if w0 < tini:
        w0 = tini
    if w1 >= tfin:
        w1 = tfin        
    window = (w0,w1)
    clickplot_redraw()

#---------------------------------------------------------------------------------

def clickplot(_medidas,figsize=(8,6)):
    '''
    una gráfica que permite moverse en el tiempo 
    en base a un mapa que resume todo el período en una imagen
    en donde se resalta, para cada medida, los intervalos
    en donde saltó una alarma por algún tipo de anomalía detectada
    '''
    global clickfig, tini, tfin, window, medidas, tipos, alarm_map,\
    map_h, alarm_map_zoom
    medidas = _medidas

    tini = datetime.datetime(datetime.MAXYEAR,1,1) 
    tfin = datetime.datetime(datetime.MINYEAR,1,1) 
    tipos = set()
    for m in medidas:
        tini_med = m.tiempo[0]
        tfin_med = m.tiempo[-1]
        if tini > tini_med:
            tini = tini_med
        if tfin < tfin_med:
            tfin = tfin_med
        if m.tipo not in tipos:
            tipos.add(m.tipo)
    tipos = list(tipos)
    # un plot por tipo de gráfica
    #
    # período que abarca todas las medidas a plotear
    #
    
    clickfig = plt.figure(figsize=figsize,dpi=96)
    nfiltros = 0
    for m in medidas:
        nfiltros = nfiltros + len (m.get_filtros())

    map_h = BAR_HEIGHT * nfiltros
    alarm_map = create_alarm_map (map_h, MAP_WIDTH, medidas,tini,tfin)
    
    if imprimir_map_zoom:
        alarm_map_zoom = create_alarm_map (map_h, MAP_WIDTH_ZOOM, medidas,tini,tfin)
    
    cid = clickfig.canvas.mpl_connect('button_press_event', click_event_handler)
    cid = clickfig.canvas.mpl_connect('scroll_event', scroll_event_handler)
    #
    # ahora si, se muestran las curvas en la ventana de tiempo actual
    #2000
    window_size = DEFAULT_WINDOW_SIZE
    window = (tini, tini+window_size)
    tcenter = tini + window_size/2
    clickplot_redraw()

#---------------------------------------------------------------------------------

def create_alarm_map (map_h,map_width, medidas,tini_z,tfin_z):

    alarm_map = np.zeros((map_h,map_width,3))
    # el mapa de alarma tiene MAP_WIDTH pixels de ancho
    # pixels_per_t indica a cuántos pasos de tiempo corresponde 1 pixel   
    period = tfin_z - tini_z
    map_h, MAP_WIDTH, z  = alarm_map.shape
    t_per_pixel = period * (1.0 / MAP_WIDTH) 
    
    col_i = np.zeros((map_h,1),dtype=np.bool)
    row_i = np.zeros((1,MAP_WIDTH),dtype=np.bool)
    box_i = np.zeros((map_h,MAP_WIDTH))
    i_filtro = 0
    i_medida = 0
    for m in medidas:
        c_i = viridis(i_medida/len(medidas))
        # vector con 1's donde la señal está filtrada
        # este vector es relativo a lo tiempos de la señal
        #print(m.nombre)
        t_i = m.tiempo
        tfin = t_i[-1]
        tini = t_i[0]
        filtros = m.get_filtros()
        dt = (tfin - tini) / (len(t_i) - 1)
        #print('medida',i_medida,' filtro', i_filtro, 'tini=',tini,'tfin=',tfin, 'dt',dt)
        for fnom,fdata in filtros.items():
            print('filtro:',fnom,' data:', len(fdata))
            row_i[:] = 0
            for j in range(MAP_WIDTH):
                tmed = tini_z + j*t_per_pixel # tiempo t en la medida
                if tmed < tini:
                    continue
                if tmed > tfin:
                    continue
                k0 = (tmed - tini)/dt
                row_i[0, j] = fdata[int(k0)]
            col_i[:] = 0
            col_i[ (i_filtro*BAR_HEIGHT) : ((i_filtro+1)*BAR_HEIGHT),0 ] = 1
            box_i[:,:] = np.dot(col_i,row_i).astype(np.bool)
            alarm_map[:,:,0] = alarm_map[:,:,0] + c_i[0]*box_i
            alarm_map[:,:,1] = alarm_map[:,:,1] + c_i[1]*box_i
            alarm_map[:,:,2] = alarm_map[:,:,2] + c_i[2]*box_i
            i_filtro = i_filtro + 1
        # terminamos con esta medida
        i_medida = i_medida + 1

    alarm_map[alarm_map == 0] = 1
    
    return alarm_map
