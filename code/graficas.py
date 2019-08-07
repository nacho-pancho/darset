# -*- coding: utf-8 -*-
#
# Funciones de graficado
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

MAP_WIDTH = 1000
BAR_HEIGHT = 10
DEFAULT_WINDOW_SIZE = datetime.timedelta(days=7)
DEFAULT_ZOOM = 1.5
DEFAULT_TIME_DELTA = datetime.timedelta(minutes=10)

clickfig = None
tini = None
tfin = None
window = None
medidas = None
viridis = cm.get_cmap('viridis')

#---------------------------------------------------------------------------------

def clickplot_redraw():
    global clickfig, window, medidas
    plt.figure(clickfig)
    for i in range(len(medidas)):
        med_i = medidas[i]
        t_i = med_i.tiempo
        x_i = t_i[(t_i >= window[0]) and (t_i < window[1])]
        y_i = med_i.muestras[(t_i >= tini) and (t_i < tfin)]
        plt.subplot(len(medidas)+1,1,i)
        plt.plot(x_i,y_i,color=)

#---------------------------------------------------------------------------------

def click_event_handler(event):
    '''
    manejo de eventos del mouse sobre una gráfica
    utilizado para el clickplot
    '''
    global window
    #
    # capturar pos (x,y)  en la imagen
    #
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y={iy}')
    #
    # mapear a posicion temporal
    #
    #return (ix,iy)    
    tcenter = tini + (ix/1000)*(tfin-tini)
    w0 = tcenter - window_size/2
    w1 = tcenter + window_size/2
    window = (w0,w1)
    clickplot_redraw()

#---------------------------------------------------------------------------------

def clickplot(medidas):
    '''
    una gráfica que permite moverse en el tiempo 
    en base a un mapa que resume todo el período en una imagen
    en donde se resalta, para cada medida, los intervalos
    en donde saltó una alarma por algún tipo de anomalía detectada
    '''
    global clickfig, tini, tfin, window, medidas


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
    # un plot por tipo de gráfica
    #
    # período que abarca todas las medidas a plotear
    #
    period = tfin - tini
    print(f"tini={tini} tfin={tfin} period={period}")
    #
    # el mapa de alarma tiene MAP_WIDTH pixels de ancho
    # pixels_per_t indica a cuántos pasos de tiempo corresponde 1 pixel
    #
    t_per_pixel = period * (1.0 / MAP_WIDTH)
    print(f"t_per_pixel={t_per_pixel}")
    map_h = BAR_HEIGHT * len(medidas)
    alarm_map = np.zeros((map_h,MAP_WIDTH,3))
    #
    # grafica coloreada
    #
    clickfig = plt.figure()
    ax0 = plt.subplot(len(tipos)+1,1,len(tipos))
    #
    # por cada gráfica se corresponde una tira del alarm_map de alto BAR_HEIGHT
    # esta tira es pintada con rectángulos del mismo color que la gráfica
    # en donde las alarmas están activas
    #
    col_i = np.zeros((map_h,1),dtype=np.bool)
    row_i = np.zeros((1,MAP_WIDTH),dtype=np.bool)
    box_i = np.zeros((map_h,MAP_WIDTH))
    for i in range(len(medidas)):
        c_i = viridis(i/len(medidas))
        # vector con 1's donde la señal está filtrada
        # este vector es relativo a lo tiempos de la señal
        med_i = medidas[i]
        alarm_i = med_i.filtrada()
        row_i[:] = 0
        t_i = med_i.tiempo
        dt = (t_i[-1]-t_i[0])/len(t_i)
        for j in range(MAP_WIDTH):
            tmed = tini + j*t_per_pixel # tiempo t en la medida
            if tmed < t_i[0]:
                continue
            if tmed > t_i[-1]:
                continue
            k0 = (tmed - t_i[0])/dt
            row_i[0, j] = alarm_i[ int(k0)  ]
        col_i[:] = 0
        col_i[(i*BAR_HEIGHT):((i+1)*BAR_HEIGHT),0] = 1
        box_i[:,:] = np.dot(col_i,row_i).astype(np.bool)
        alarm_map[:,:,0] = alarm_map[:,:,0] + c_i[0]*box_i
        alarm_map[:,:,1] = alarm_map[:,:,1] + c_i[1]*box_i
        alarm_map[:,:,2] = alarm_map[:,:,2] + c_i[2]*box_i
    alarm_map[alarm_map == 0] = 1
    plt.imshow(alarm_map)
    cid = fig.canvas.mpl_connect('button_press_event', click_event_handler)
    #
    # ahora si, se muestran las curvas en la ventana de tiempo actual
    #
    window_size = DEFAULT_WINDOW_SIZE
    window = (tini, tini+window_size)
    clickplot_redraw()

#---------------------------------------------------------------------------------
