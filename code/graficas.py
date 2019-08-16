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

MAP_WIDTH = 2000
BAR_HEIGHT = 30
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

#---------------------------------------------------------------------------------

def clickplot_redraw():
    global window, medidas, tipos, tini, tfin
    print("redraw")
    plt.figure(clickfig.number)
    print("window:",window)

    legends = dict()
    for tipo in tipos:
        legends.update( {tipo:list()} )
        
    for i in range(len(medidas)):
        med_i = medidas[i]        
        idx_tipo = tipos.index(med_i.tipo)
        t_i = med_i.tiempo
        legends[med_i.tipo].append(med_i.nombre)
        idx_i =list(map(lambda t: (t >= window[0]) and (t < window[1]), t_i))
        x_i = list()
        y_i = list()
        for j in range(len(idx_i)):
            if idx_i[j]:
                x_i.append(t_i[j])
                y_i.append(med_i.muestras[j])
        print(x_i[0],x_i[-1])
        plt.subplot(len(tipos)+1,1,idx_tipo+1)
        c_i = viridis(i/len(medidas))
        plt.plot(x_i,y_i,color=c_i)
        plt.axis([window[0],window[1],np.min(y_i),np.max(y_i)])
        plt.ylabel(med_i.tipo)
        plt.draw()
        
    print(legends)

    for i in range(len(tipos)):
        plt.subplot( len(tipos)+1, 1, i+1 )
        plt.legend( legends[ tipos[i] ], loc='upper right' )
        plt.grid(True)

    #
    # actualizar el mapa
    #
    plt.subplot(len(tipos)+1,1,len(tipos)+1)
    j0 = int((window[0]-tini)/(tfin-tini)*MAP_WIDTH)
    j1 = int((window[1]-tini)/(tfin-tini)*MAP_WIDTH)
    print(j0,j1)
    fondo = np.copy(alarm_map)
    fondo[:] = 1
    fondo[:,j0:j1,:3] = 0
    plt.imshow(alarm_map)
    plt.imshow(fondo,alpha=0.25)
    plt.draw()
        
#---------------------------------------------------------------------------------

def click_event_handler(event):
    '''
    manejo de eventos del mouse sobre una gráfica
    utilizado para el clickplot
    '''
    global window,tcenter
    #
    # capturar pos (x,y)  en la imagen
    #
    ix, iy = event.xdata, event.ydata
    print(f'click: x = {ix}, y={iy}')
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
    global window_size,window,tcenter
    step = event.step
    print(f'scroll: step = {step}')
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

def clickplot(_medidas):
    '''
    una gráfica que permite moverse en el tiempo 
    en base a un mapa que resume todo el período en una imagen
    en donde se resalta, para cada medida, los intervalos
    en donde saltó una alarma por algún tipo de anomalía detectada
    '''
    global clickfig, tini, tfin, window, medidas, tipos, alarm_map
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
    cid = clickfig.canvas.mpl_connect('button_press_event', click_event_handler)
    cid = clickfig.canvas.mpl_connect('scroll_event', scroll_event_handler)
    #
    # ahora si, se muestran las curvas en la ventana de tiempo actual
    #
    window_size = DEFAULT_WINDOW_SIZE
    window = (tini, tini+window_size)
    tcenter = tini + window_size/2
    clickplot_redraw()

#---------------------------------------------------------------------------------
