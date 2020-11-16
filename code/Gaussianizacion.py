# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:46:05 2020

@author: fpalacio
"""
import numpy as np
from scipy.stats import norm



def GenerarYGuardarLentes( Mr, Filt, t, nombres, hs_overlap, carpeta_lentes):

    horas = np.array([ti.hour for ti in t])
    #print(horas)

    for col in range(Mr.shape[1]):            
        idx_val = (Filt[:, col] == 0) 
        #print(idx_val)        
        m = Mr[idx_val, col]
        h = horas[idx_val]
        
        for h_lente in range(24):
            inth = []
            h1 = int(h_lente - hs_overlap);
            h2 = int(h_lente + hs_overlap);
            if h1<0:
                inth.extend(range(24 + h1, 23 + 1))
                inth.extend(range(h2 + 1))
            elif h2>23:
                inth.extend(range(h1, 23 + 1))
                inth.extend(range(0, h2 - 24 + 1))                
            else:
                inth.extend(range(h1, h2 + 1))
        
            mask = np.in1d(h, inth)
            #idxh = [horas.index(hi) for hi in inth]
            lente = np.sort(m[mask], axis=None) 
            archi = carpeta_lentes + 'l_' + nombres[col] + '_h' + str(h_lente) + '.npy'          

            np.save(archi, lente) 


def GaussianisarSeries( Mr, Filt, t, nombres, carpeta_lentes ):

    # cargo TODOS los lentes(se puede mejorar cargando solo los lentes de los nombres)
    lentes = {}
    for filename in os.listdir('C:/lentes/'):
        if filename.endswith('.npy'):
            lentes[filename] = np.load('C:/lentes/' + filename)
    
    horas = np.array([ti.hour for ti in t])
    #print(horas)    

    Mg = np.empty_like(Mr)
        
    for kSerie in range(Mr.shape[1]):
        m = Mr[:, kSerie]
        for k in range(len(m)):
            if (Filt[k, kSerie] == 0):                
                h = horas[k]               
                archi_lente = archi.archi_lente(nombre, h)
                lente = lentes[archi_lente]
                ni = len(lente);
                ii = max(np.where(lentes <= m[k])) # le saco el menor o igual xq me esta dando NaN
                if not ii:   # menor que el menor dato 
                  paux = 0.5             
                elif (ii == ni)    % mayor o igual que el mayor dato
                  paux = (ii - 0.5)
                else
                  paux = (ii - 0.5) * (m[k]-lente[ii]) + (ii-1.5) * (lente[ii + 1] - m[k])
                  paux = paux / (lentes[ii+1]-lentes[ii])
                paux = max( paux, 0.5 )
                paux = paux / ni                
                Mg[k, kSerie] = norm.ppf(paux)

    return Mg


def DesGaussianizarSeries( Mg, Filt, t, nombres, carpeta_lentes )

    horas = np.array([ti.hour for ti in t])
    
    # cargo TODOS los lentes (se puede mejorar cargando solo los lentes de los nombres)
    lentes = {}
    for filename in os.listdir('C:/lentes/'):
        if filename.endswith('.npy'):
            lentes[filename] = np.load('C:/lentes/' + filename)

    
    Mr = norm.cdf(Mg);
    
    for kSerie in range(Mg.shape[1]):
        m = Mg[:, kSerie]
        for k in range(len(m)):
            if (Filt[k, kSerie] == 0):                
                h = horas[k]               
                archi_lente = archi.archi_lente(nombre, h)
                lente = lentes[archi_lente]
                Mr[]


    for t=1:NDatos
       if filtro(t)== 1 
        h=hora(t,1);
        eval(['lentes=' tipoSeries 'lentes' num2str(1) 'h' num2str(h) ';']);
        Xr(t,1:NCrons)=quantile(lentes,Xr(t,1:NCrons));   
       else
          Xr(t,1:NCrons)=PotR(t);
       end        
    end
    
    return Mr
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


'''
Xg=Xr;
NDatos=length(Xr(:,1));
NSeries=length(Xr(1,:));
eval(['load lentes/' tipoSeries ' *']);
for v=1:NSeries
    serie_r=squeeze(Xr(:,v));
    for t=1:NDatos
        if serie_r(t)~=nodata
            h=hora(t,1);
            eval(['lentes=' tipoSeries 'lentes' num2str(v) 'h' num2str(h) ';']);
            ni=numel(lentes);
            ii=max(find(lentes<=Xg(t,v)));% le saco el menor o igual xq me esta dando NaN
            if isempty(ii)   % menor que el menor dato 
              paux=0.5;             
            elseif ii==ni    % mayor o igual que el mayor dato
              paux=(ii-.5);
            else
              paux=(ii-.5)*(Xg(t,v)-lentes(ii))+(ii-1.5)*(lentes(ii+1)-Xg(t,v));
              paux=paux/(lentes(ii+1)-lentes(ii));
            end
            paux=max(paux,0.5);
            paux=paux/ni;
            Xg(t,v)=norminv(paux,0,1);
            if isnan(Xg(t,v))
                input('ERROR Y/N [Y]: ', 's')
            end
        end
    end
    idx=find(Xg(:,v)~=nodata);
    figure
    hist(Xg(idx,v),100);
    [h,p] = chi2gof(Xg(idx,v),'Alpha',0.01);
    title (['chi2gof -> h = ' num2str(h,3) ' , p = ' num2str(p,3) ]);
end
eval(['clear ' tipoSeries 'lentes'  '*']);

end

'''