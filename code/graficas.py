# -*- coding: utf-8 -*-
#
# Funciones de graficado
#
from windrose import WindroseAxes

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

def click_event_handler(event):
    '''
    manejo de eventos del mouse sobre una gráfica
    utilizado para el clickplot
    '''
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y={iy}')
    return (ix,iy)

def clickplot(medidas):
    '''
    una gráfica que permite moverse en el tiempo 
    en base a un mapa que resume todo el período en una imagen
    en donde se resalta, para cada medida, los intervalos
    en donde saltó una alarma por algún tipo de anomalía detectada
    '''
    fig = plt.figure()
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
    ax0 = plt.subplot(len(tipos)+1,1,len(tipos))
    #
    # por cada gráfica se corresponde una tira del alarm_map de alto BAR_HEIGHT
    # esta tira es pintada con rectángulos del mismo color que la gráfica
    # en donde las alarmas están activas
    #
    viridis = cm.get_cmap('viridis')
    col_i = np.zeros((map_h,1),dtype=np.bool)
    row_i = np.zeros((1,MAP_WIDTH),dtype=np.bool)
    box_i = np.zeros((map_h,MAP_WIDTH))
    for i in range(len(medidas)):
        c_i = viridis(i/len(medidas))
        # vector con 1's donde la señal está filtrada
        # este vector es relativo a lo tiempos de la señal
        alarm_i = medidas[i].filtrada()
        #print(alarm_i)
        row_i[:] = 0
        for j in range(MAP_WIDTH):
            tmed = tini + j*t_per_pixel # tiempo t en la medida
            if tmed >= medidas[i].tiempo[0] and tmed < medidas[i].tiempo[-1]:
                row_i[0, j] = alarm_i[ (tmed - medidas[i].tiempo[0])/period  ]
        #print(f"offset {i}={toffset_i}")
        #print(f"row_{i}={row_i}")
        col_i[:] = 0
        col_i[(i*BAR_HEIGHT):((i+1)*BAR_HEIGHT),0] = 1
        box_i[:,:] = np.dot(col_i,row_i).astype(np.bool)
        #box_i2 = np.dot(box_i,np.ones((1,3),dtype=np.bool)).astype(np.bool)
        print(box_i.shape)
        alarm_map[:,:,0] = alarm_map[:,:,0] + c_i[0]*box_i
        alarm_map[:,:,1] = alarm_map[:,:,1] + c_i[1]*box_i
        alarm_map[:,:,2] = alarm_map[:,:,2] + c_i[2]*box_i
    plt.imshow(alarm_map)
    cid = fig.canvas.mpl_connect('button_press_event', click_event_handler)
    plt.show()
    return fig
