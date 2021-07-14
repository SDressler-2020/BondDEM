# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:02:03 2018

@author: jodie
"""
import csv
import launchDEMGPUcnstnt as DEM
import matplotlib.pyplot as plt
import importlib
import time
import numpy as np
import math

def write_a_new_csv(xxs,yys,yys2,name):
    outlist=[]
    for i in range(0,len(xxs)-1):
        if np.isnan(yys[i]):
            xxs[i]=0
        if np.isnan(yys2[i]):
            xxs[i]=0
        outlist.append([xxs[i],yys[i],yys2[i]])
    with open(name, 'w') as myfile:
        wr = csv.writer(myfile,lineterminator='\n')
        wr.writerows(outlist)
        myfile.close()


Ms=[6000,8000,100000]

r=0.7

matrixwidth=6

fiberwidth=1

flag=1
FE=5e10
ME=5e9
Fv=0.2
Mv=0.3
e=0.0
Ks=[1e9]
Tt=1e-5

Ml=[30]

Outlist = []
Outlist.append(['fnx','slp','u','num','tltldmg','brkn','fx','dx','fl','matrixlength','Vs'])
V=[[10,0]] #extraction velocity x,y
ShearFinal=[]

for Kt in Ks:
    for matrixlength in Ml:
        for Vs in V:
            if r==2:
                mods=[[0.01,0.03],[0.01,0.03],[0.01,0.03]] #calibration constants
         
            if r==1:
    #            
                mods=[[1.6,0.03],[1.6,0.03],[1.6,0.03]]#calibration constants
            if r==0.7:
                mods=[[1.8,0.001],[1.8,0.001],[1.8,0.001]]#calibration constants
            if r == 0.5:
                mods=[[1.42,0.01],[1.42,0.01],[1.42,0.01]]#calibration constants
    
    
            mod=[mods[0],mods[0],mods[2],mods[2],mods[0],mods[0],mods[0],mods[0],mods[0],mods[0],mods[0],mods[0]]
            mod=np.array(mod)
            fiberlength=matrixlength
            sample=[r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag]
            Bprp,Fnx,slip,U,number,totaldamage,broken,ForceX,dxL,Prtbase,Prtor,B,flags,Shear=DEM.runDEMGPU(Ms[0],ME,FE,Tt,Vs,1000000,1000000*Vs[0]*Tt,sample,mod,Kt)
            for s in Shear:
                ShearFinal.append(s)

            for fnx,slp,u,num,tltldmg,brkn,fx,dx,fl in zip(Fnx,slip,U,number,totaldamage,broken,ForceX,dxL,flags):
                Outlist.append([fnx,slp,u,num,tltldmg,brkn,fx,dx,fl,matrixlength,Vs[0],matrixlength])

# with open('C:/Users/SDressler/Documents/PostGrad/Shear_at_yield.csv','w') as output:
#     writr=csv.writer(output, lineterminator='\n')
#     writr.writerows(ShearFinal)

# with open('C:/Users/SDressler/Documents/PostGrad/forcecontrol.csv','w') as output:
#     writr=csv.writer(output, lineterminator='\n')
#     writr.writerows(Outlist)