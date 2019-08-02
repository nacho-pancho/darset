# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import datos

MAP_WIDTH = 1000
BAR_HEIGHT = 50

def click_event_handler(event):
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y={iy}')
    return (ix,iy)

def clickplot(medidas):
    fig = plt.figure()
    tini = 0
    tfin = 0
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
    #
    # el mapa de alarma tiene MAP_WIDTH pixels de ancho
    # pixels_per_t indica a cuántos pasos de tiempo corresponde 1 pixel
    #
    pixels_per_t = MAP_WIDTH / period
    map_h = BAR_HEIGHT * len(medidas)
    alarm_map = np.zeros((map_h,MAP_WIDTH,3))
    ax0 = plt.subplot(len(tipos)+1,1)
    #
    # grafica coloreada
    #
    ones = np.ones((1,MAP_WIDTH,))
    plt.subplot(len(tipos)+1,1,len(tipos))
    plt.imshow(alarm_map)
    
    cid = fig.canvas.mpl_connect('button_press_event', click_event_handler)