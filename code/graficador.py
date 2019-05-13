# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import rutas

def plotmedidas(medidor):
    fh = plt.figure()
    leg = list()
    nombre_med = medidor.nombre
    fechas = medidor.tiempo
    for m in medidor.medidas:
        plt.plot(fechas,m.muestras)
        plt.plot(fechas,m.trancada())
        plt.plot(fechas,m.fuera_de_rango())
        leg.append(nombre_med + m.tipo)
    plt.legend(leg)            
    return fh

if __name__ == "__main__":
    parque = rutas.leerArchiSCADA(5)
    handles = list()
    for med in  parque.medidores:
        handles.append(plotmedidas(med))
    plt.show()