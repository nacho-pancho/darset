from numpy import arange, sin, pi, float, size

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
import rutas as r
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import wx

class MyFrame(wx.Frame):
    def __init__(self, medidas,parent,id):
        wx.Frame.__init__(self,parent, id, 'scrollable plot',
                style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER,
                size=(800, 400))
        self.panel = wx.Panel(self, -1)

        
        self.fig = Figure((5, 4), 75)

        self.axes = self.fig.subplots(nrows=2, ncols=1)    

        self.canvas = FigureCanvasWxAgg(self.panel, -1, self.fig)
        self.scroll_range = 400
        self.canvas.SetScrollbar(wx.HORIZONTAL, 0, 5,
                                 self.scroll_range)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, -1, wx.EXPAND)

        self.panel.SetSizer(sizer)
        self.panel.Fit()
        
        self.medidas = medidas

        self.init_data()
        self.init_plot()

        self.canvas.Bind(wx.EVT_SCROLLWIN, self.OnScrollEvt)
        
        
        self.fecha_ini_ts = self.medidas[0].tiempo[0].timestamp()
        
        print (self.fecha_ini_ts)
        self.fecha_fin_ts = self.medidas[0].tiempo[-1].timestamp()


    def init_data(self):

        # Extents of data sequence:
        self.i_min = 0
        self.i_max = len(self.medidas[0].tiempo)

        # Size of plot window:
        self.i_window = 1000

        # Indices of data interval to be plotted:
        self.i_start = 0
        self.i_end = self.i_start + self.i_window

    def init_plot(self):
        for k in range(len(self.medidas)):   
            self.df_meds = pd.DataFrame(self.medidas[k].muestras, index=self.medidas[k].tiempo,
                                   columns=[self.medidas[k].nombre])
    
            filtros, nombres = self.medidas[k].filtrosAsInt()
            
            self.df_filt = pd.DataFrame(filtros,index=self.medidas[k].tiempo,columns=nombres)
     

    def draw_plot(self):

        fecha_ini_aux_ts = self.fecha_ini_ts + self.i_start/6/24 * 3600 * 24
        print(fecha_ini_aux_ts)

        fecha_fin_aux_ts = self.fecha_ini_ts + self.i_end/6/24  * 3600 * 24
        print(fecha_fin_aux_ts)
        
        f_ini_aux = datetime.fromtimestamp(fecha_ini_aux_ts)
        print(f_ini_aux)
        f_fin_aux = datetime.fromtimestamp(fecha_fin_aux_ts)
        print(f_fin_aux)
        df_meds_filt = self.df_meds[(self.df_meds.index >= f_ini_aux) &
                               (self.df_meds.index <= f_fin_aux)]
               
        df_meds_filt.plot(ax=self.axes[0], linewidth=2)


        df_filt_filt = self.df_filt[(self.df_filt.index >= f_ini_aux) & 
                               (self.df_filt.index <= f_fin_aux)] 
        
        df_filt_filt.plot(ax=self.axes[1], linewidth=2)        

        # Redraw:
        self.canvas.draw()

    def update_scrollpos(self, new_pos):
        self.i_start = self.i_min + new_pos
        self.i_end = self.i_min + self.i_window + new_pos
        self.canvas.SetScrollPos(wx.HORIZONTAL, new_pos)
        self.draw_plot()

    def OnScrollEvt(self, event):
        evtype = event.GetEventType()

        if evtype == wx.EVT_SCROLLWIN_THUMBTRACK.typeId:
            pos = event.GetPosition()
            self.update_scrollpos(pos)
        elif evtype == wx.EVT_SCROLLWIN_LINEDOWN.typeId:
            pos = self.canvas.GetScrollPos(wx.HORIZONTAL)
            self.update_scrollpos(pos + 1)
        elif evtype == wx.EVT_SCROLLWIN_LINEUP.typeId:
            pos = self.canvas.GetScrollPos(wx.HORIZONTAL)
            self.update_scrollpos(pos - 1)
        elif evtype == wx.EVT_SCROLLWIN_PAGEUP.typeId:
            pos = self.canvas.GetScrollPos(wx.HORIZONTAL)
            self.update_scrollpos(pos - 10)
        elif evtype == wx.EVT_SCROLLWIN_PAGEDOWN.typeId:
            pos = self.canvas.GetScrollPos(wx.HORIZONTAL)
            self.update_scrollpos(pos + 10)
        else:
            print ("unhandled scroll event, type id:", evtype)

class MyApp(wx.App):
    def __init__(self, medidas):
        self.medidas = medidas
        wx.App.__init__(self)
        
    def OnInit(self):
        self.frame = MyFrame(self.medidas,parent=None,id=-1)
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

if __name__ == '__main__':
    nidCentral = 5    
    med_10min, med_15min = r.leerArchiSMEC(nidCentral)
    parque = r.leerArchiSCADA(nidCentral) 
    
    parque.pot_SMEC  = med_10min
    
    #parque.decorrelacion()
    
    meds = []
    #meds.append(med_10min)
    #meds.append(med_15min)
    vel_SCADA = parque.medidores[0].medidas[0]
    meds.append(vel_SCADA)
    meds.append(parque.pot) 
    
    #pltGrfs.plotMedidas(meds,'False','2018-10-25','2018-10-30',r.path(nidCentral),True)
    
    app = MyApp(meds)
    app.MainLoop()
