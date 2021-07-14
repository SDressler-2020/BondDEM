#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 22:29:30 2018

@author: sven
"""

import numpy as np
from numba import cuda, int16, float32, from_dtype, jit
import numba as nb
import math

def hccp(d,matrixlength,matrixwidth):
    r=d/2
    maxD=max([matrixlength,matrixwidth])
    rootthree=math.sqrt(3)
    rootsix=math.sqrt(6)
    Points=[]
    print(maxD,r)
    print((maxD/d+3)*(maxD/d+3)*(maxD/d+3))
    for i in range (-int(maxD/d+3),int(maxD/d+3)):
        for j in range (-int(maxD/d+3),int(maxD/d+3)):
            for k in range (-int(maxD/d+3),int(maxD/d+3)):
                px=(2*i+(j+k)%2)*r
                py=rootthree*(j+(1/3)*(k%2))*r
                pz=2*rootsix/3*k*r
#                if pz==0:
                Points.append([px,py,pz])
#                print([px,py,pz])
    return Points

def hccprotated(d,matrixlength,matrixwidth,fl):
    r=d/2
    maxD=max([matrixlength,matrixwidth,fl])
    rootthree=math.sqrt(3)
    rootsix=math.sqrt(6)
    Points=[]
    print(maxD,r)
    print((maxD/d+10)*(maxD/d+10)*(maxD/d+10))
    for i in range (-round(maxD/d+10),round(maxD/d+10)):
        for j in range (-round(maxD/d+10),round(maxD/d+10)):
            for k in range (-round(maxD/d+20),round(maxD/d+20)):
                px=(2*i+(j+k)%2)*r
                py=rootthree*(j+(1/3)*(k%2))*r
                pz=2*rootsix/3*k*r
#                if pz==0:
                Points.append([pz,px,py])
#                print([px,py,pz])
    return Points

def sample_genhccp(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    ('hash', 'u4'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular 
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    Pnt=hccprotated(r,matrixlength+2,matrixwidth+2,fiberlength+2)
    Pnt=np.array(Pnt)
    lengthRange=matrixlength
    maxL=lengthRange+2
#    maxL=np.max(Pnt[Pnt[:,0]<lengthRange+5*r,0])
#    maxL=np.max(Pnt[:,0])
    lengthStart=0
#    lengthStart=np.min(Pnt[Pnt[:,0]>0,0])
#   
# r=diameter

    for p in Pnt:
        i=p[0]
        j=p[1]
        k=p[2]
        if math.sqrt(k**2+j**2)<=matrixwidth and i<=lengthRange and i>=0: # and k==0:
            if math.sqrt(k**2+j**2)>fiberwidth:# or i<lengthRange-fiberlength:
                if i==lengthStart and math.sqrt(k**2+j**2)>=matrixwidth/2.:
                   Points.append([i,j,k,0])
                else:
                    Points.append([i,j,k,1])
        if i>=lengthStart and i<maxL-1 and math.sqrt(k**2+j**2)<=fiberwidth:
            Points.append([i,j,k,2])
#            print(Points[len(Points)-1])
        if i>=maxL-1 and math.sqrt(k**2+j**2)<=fiberwidth:
            Points.append([i,j,k,3])
#            print('end appended')

    Particles=np.empty(len(Points),dtype=Particle_nb)
    print(len(Points),'number of particles')
    cellsize = 4*r
    Pointsarr=np.array(Points)
    maxx=max(Pointsarr[:,0])
    maxy=max(Pointsarr[:,1])
    maxz=max(Pointsarr[:,2])
    minx=min(Pointsarr[:,0]) 
    miny=min(Pointsarr[:,1])
    minz=min(Pointsarr[:,2])
    print(maxx,maxy,maxz,minx,miny,minz)
    wx=int((maxx-minx)/cellsize)
    wy=int((maxy-miny)/cellsize)
    wz=int((maxz-minz)/cellsize)
    print(wx,wy,wz,'size')
    Points2=[]
    for p in Points:
        xx=int((p[0]-minx)/cellsize)
        yy=int((p[1]-miny)/cellsize)
        zz=int((p[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
        hsh=int(hsh)
        p.append(hsh)
        Points2.append(p)
        
    def takehash(elem):
        return elem[4]
    Points2.sort(key=takehash)
#    print(Points2)
    i=0
    for x in Points2:
        xx=int((x[0]-minx)/cellsize)
        yy=int((x[1]-miny)/cellsize)
        zz=int((x[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
#        print(xx,yy,zz,hsh)
        hsh=int(hsh)
        Particles[i]['hash']=hsh
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
#        if x[0]>30:
#            print(x)
        i+=1
#    print(1/0)
    maxhash=max(Particles[:]['hash'])
    hashdata=np.zeros((2,maxhash+1))
    hashdata.fill(-1)
    c=0
    for P in Particles:
        if hashdata[0,P['hash']]==-1:
            hashdata[0,P['hash']]=c
        hashdata[1,P['hash']]+=1
        c+=1
#    print(hashdata)
#    print(1/0)
    hashcreationdata=np.array([minx,miny,minz,wx,wy,wz,cellsize])
    return Particles,hashdata,hashcreationdata

def sample_genhccp2(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    ('hash', 'u4'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular 
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    Pnt=hccprotated(r,matrixlength+10,matrixwidth+10,fiberlength+10)
    Pnt=np.array(Pnt)
    lengthStart=0
    lengthStartF=5
    lengthendM=lengthStartF+matrixlength
    lengthendF=5+matrixlength+5
    maxL=np.max(Pnt[:,0])
#    lengthStart=np.min(Pnt[Pnt[:,0]>0,0])
#   
# r=diameter

    for p in Pnt:
        i=p[0]
        j=p[1]
        k=p[2]
        if math.sqrt(k**2+j**2)<=matrixwidth and i<=lengthendM and i>=0: # and k==0:
            if math.sqrt(k**2+j**2)>fiberwidth or i<lengthStartF-1.5*r:
                if i==lengthStart:
                   Points.append([i,j,k,0])
                else:
                    Points.append([i,j,k,1])
        if i>=lengthStartF and i<maxL and math.sqrt(k**2+j**2)<=fiberwidth:
            Points.append([i,j,k,2])
#            print(Points[len(Points)-1])
#        if i>lengthendF-1 and math.sqrt(k**2+j**2)<=fiberwidth:
#            Points.append([i,j,k,3])
        if i==maxL and math.sqrt(k**2+j**2)<=fiberwidth:
            Points.append([i,j,k,3])
            print('end appended')
    
    
    Particles=np.empty(len(Points),dtype=Particle_nb)
    print(len(Points),'number of particles')
    cellsize = 4*r
    Pointsarr=np.array(Points)
#    print(np.min(Pointsarr[Pointsarr[:,3]==2,0]))
#    print(np.max(Pointsarr[Pointsarr[:,3]==1,0]))
#    print(1/0)
    maxx=max(Pointsarr[:,0])
    maxy=max(Pointsarr[:,1])
    maxz=max(Pointsarr[:,2])
    minx=min(Pointsarr[:,0]) 
    miny=min(Pointsarr[:,1])
    minz=min(Pointsarr[:,2])
    print(maxx,maxy,maxz,minx,miny,minz)
    wx=int((maxx-minx)/cellsize)
    wy=int((maxy-miny)/cellsize)
    wz=int((maxz-minz)/cellsize)
    print(wx,wy,wz,'size')
    Points2=[]
    for p in Points:
        xx=int((p[0]-minx)/cellsize)
        yy=int((p[1]-miny)/cellsize)
        zz=int((p[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
        hsh=int(hsh)
        p.append(hsh)
        Points2.append(p)
        
    def takehash(elem):
        return elem[4]
    Points2.sort(key=takehash)
#    print(Points2)
    i=0
    for x in Points2:
        xx=int((x[0]-minx)/cellsize)
        yy=int((x[1]-miny)/cellsize)
        zz=int((x[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
#        print(xx,yy,zz,hsh)
        hsh=int(hsh)
        Particles[i]['hash']=hsh
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
#        if x[0]>30:
#            print(x)
        i+=1
#    print(1/0)
    maxhash=max(Particles[:]['hash'])
    hashdata=np.zeros((2,maxhash+1))
    hashdata.fill(-1)
    c=0
    for P in Particles:
        if hashdata[0,P['hash']]==-1:
            hashdata[0,P['hash']]=c
        hashdata[1,P['hash']]+=1
        c+=1
#    print(hashdata)
#    print(1/0)
    hashcreationdata=np.array([minx,miny,minz,wx,wy,wz,cellsize])
    return Particles,hashdata,hashcreationdata

def sample_genhccpK(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    ('hash', 'u4'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular 
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    Pnt=hccprotated(r,matrixlength+5,matrixwidth+5,fiberlength+5)
    Pnt=np.array(Pnt)
#    lengthRange=matrixlength
#    maxL=lengthRange+2
#    maxL=np.max(Pnt[Pnt[:,0]<lengthRange+5*r,0])
#    maxL=np.max(Pnt[:,0])
#    lengthStart=0
#    lengthStart=np.min(Pnt[Pnt[:,0]>0,0])
#   
# r=diameter

    for p in Pnt:
        i=p[0]
        j=p[1]
        k=p[2]
        if math.sqrt(k**2+j**2)<=matrixwidth and i>=0 and i<=10: 
            if math.sqrt(k**2+j**2)<=fiberwidth:
                Points.append([i,j,k,2])
            else:
                Points.append([i,j,k,0])
        

    Particles=np.empty(len(Points),dtype=Particle_nb)
    print(len(Points),'number of particles')
    cellsize = 8*r
    Pointsarr=np.array(Points)
    maxx=max(Pointsarr[:,0])
    maxy=max(Pointsarr[:,1])
    maxz=max(Pointsarr[:,2])
    minx=min(Pointsarr[:,0]) 
    miny=min(Pointsarr[:,1])
    minz=min(Pointsarr[:,2])
    print(maxx,maxy,maxz,minx,miny,minz)
    wx=int((maxx-minx)/cellsize)
    wy=int((maxy-miny)/cellsize)
    wz=int((maxz-minz)/cellsize)
    print(wx,wy,wz,'size')
    Points2=[]
    for p in Points:
        xx=int((p[0]-minx)/cellsize)
        yy=int((p[1]-miny)/cellsize)
        zz=int((p[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
        hsh=int(hsh)
        p.append(hsh)
        Points2.append(p)
        
    def takehash(elem):
        return elem[4]
    Points2.sort(key=takehash)
#    print(Points2)
    i=0
    for x in Points2:
        xx=int((x[0]-minx)/cellsize)
        yy=int((x[1]-miny)/cellsize)
        zz=int((x[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
#        print(xx,yy,zz,hsh)
        hsh=int(hsh)
        Particles[i]['hash']=hsh
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
#        if x[0]>30:
#            print(x)
        i+=1
#    print(1/0)
    maxhash=max(Particles[:]['hash'])
    hashdata=np.zeros((2,maxhash+1))
    hashdata.fill(-1)
    c=0
    for P in Particles:
        if hashdata[0,P['hash']]==-1:
            hashdata[0,P['hash']]=c
        hashdata[1,P['hash']]+=1
        c+=1
#    print(hashdata)
#    print(1/0)
    hashcreationdata=np.array([minx,miny,minz,wx,wy,wz,cellsize])
    return Particles,hashdata,hashcreationdata


def sample_genhccpFib(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    ('hash', 'u4'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular 
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    Pnt=hccprotated(r,10,10,10)
    Pnt=np.array(Pnt)
#    lengthRange=matrixlength
#    maxL=lengthRange+2
    maxL=np.max(Pnt[Pnt[:,0]<=10,0])
#    maxL=np.max(Pnt[:,0])
#    lengthStart=0
#    lengthStart=np.min(Pnt[Pnt[:,0]>0,0])
#   
# r=diameter
    count=0
    for p in Pnt:
        i=p[0]
        j=p[1]
        k=p[2]
#        if math.sqrt(k**2+j**2)<=matrixwidth and i>=0 and i<=10: 
        if math.sqrt(k**2+j**2)<=fiberwidth and i>=0 and i<=maxL:
            if i==0:
               Points.append([i,j,k,0])
            elif i == maxL:
                Points.append([i,j,k,3])
                count+=1
            else:
                Points.append([i,j,k,1])
        
    print("number of end p", count)
    Particles=np.empty(len(Points),dtype=Particle_nb)
    print(len(Points),'number of particles')
    cellsize = 8*r
    Pointsarr=np.array(Points)
    maxx=max(Pointsarr[:,0])
    maxy=max(Pointsarr[:,1])
    maxz=max(Pointsarr[:,2])
    minx=min(Pointsarr[:,0]) 
    miny=min(Pointsarr[:,1])
    minz=min(Pointsarr[:,2])
    print(maxx,maxy,maxz,minx,miny,minz)
    wx=int((maxx-minx)/cellsize)
    wy=int((maxy-miny)/cellsize)
    wz=int((maxz-minz)/cellsize)
    print(wx,wy,wz,'size')
    Points2=[]
    for p in Points:
        xx=int((p[0]-minx)/cellsize)
        yy=int((p[1]-miny)/cellsize)
        zz=int((p[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
        hsh=int(hsh)
        p.append(hsh)
        Points2.append(p)
        
    def takehash(elem):
        return elem[4]
    Points2.sort(key=takehash)
#    print(Points2)
    i=0
    for x in Points2:
        xx=int((x[0]-minx)/cellsize)
        yy=int((x[1]-miny)/cellsize)
        zz=int((x[2]-minz)/cellsize)
        hsh=zz*wx*wy+xx*wz+yy
#        print(xx,yy,zz,hsh)
        hsh=int(hsh)
        Particles[i]['hash']=hsh
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
#        if x[0]>30:
#            print(x)
        i+=1
#    print(1/0)
    maxhash=max(Particles[:]['hash'])
    hashdata=np.zeros((2,maxhash+1))
    hashdata.fill(-1)
    c=0
    for P in Particles:
        if hashdata[0,P['hash']]==-1:
            hashdata[0,P['hash']]=c
        hashdata[1,P['hash']]+=1
        c+=1
#    print(hashdata)
#    print(1/0)
    hashcreationdata=np.array([minx,miny,minz,wx,wy,wz,cellsize])
    return Particles,hashdata,hashcreationdata

def sample_genhccpCali(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular 
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    Pnt=hccp(r,matrixlength,matrixwidth)
    Pnt=np.array(Pnt)
#    lengthRange=np.max(Pnt[:,0])-0.002
    lengthRange=matrixlength
    maxL=np.max(Pnt[:,0])
    lengthStart=np.min(Pnt[Pnt[:,0]>0,0])
#    print(lengthRange,maxL,lengthStart)
#    lengthStart=np.min(Pnt[:,0])
#    matrixlength=matrixlength+0.002
    for p in Pnt:
        i=p[0]
        j=p[1]
        k=p[2]
        if math.sqrt(k**2+j**2)<=matrixwidth and i<=maxL and i>=lengthStart: # and k==0:
            if i==lengthStart:
               Points.append([i,j,k,0])
            if i>lengthStart and i < maxL:
               Points.append([i,j,k,1])
            if i ==maxL:
                Points.append([i,j,k,3])
    Particles=np.empty(len(Points),dtype=Particle_nb)
    i=0
    for x in Points:
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
        i+=1
#    print(Particles)
    return Particles

def sample_genhccpKo(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular 
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    Pnt=hccp(r,matrixlength,matrixwidth)
    Pnt=np.array(Pnt)
#    lengthRange=np.max(Pnt[:,0])-0.002
    lengthRange=matrixlength
    maxL=np.max(Pnt[:,0])
    lengthStart=np.min(Pnt[Pnt[:,0]>0,0])
#    print(lengthRange,maxL,lengthStart)
#    lengthStart=np.min(Pnt[:,0])
#    matrixlength=matrixlength+0.002
    for p in Pnt:
        i=p[0]
        j=p[1]
        k=p[2]
        if math.sqrt(k**2+j**2)<=matrixwidth and i<=maxL and i>=lengthStart: # and k==0:
            if math.sqrt(k**2+j**2)<=fiberwidth:
               Points.append([i,j,k,0])
            else:
                Points.append([i,j,k,3])
    Particles=np.empty(len(Points),dtype=Particle_nb)
    i=0
    for x in Points:
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
        i+=1
    return Particles


def sample_genCali(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular velocity
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    lengthRange=int(matrixlength/r)
    widthRange=int(matrixwidth/r)
    for i in range (0,lengthRange+1):
        for j in range (-widthRange,widthRange+1):
            for k in range (-widthRange,widthRange+1):
                if math.sqrt((k*r)**2+(j*r)**2)<=matrixwidth:# and k==0:
                    if i==0:
                       Points.append([i*r,j*r,k*r,0])
                    if i>0 and i < lengthRange:
                        Points.append([i*r,j*r,k*r,1])
                    if i==lengthRange:
                        Points.append([(i)*r,j*r,k*r,3])
    Particles=np.empty(len(Points),dtype=Particle_nb)
    i=0
    for x in Points:
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
        i+=1
    return Particles
def sample_gen(r,matrixlength,matrixwidth,fiberlength,fiberwidth,flag):
    Particle = np.dtype([
    # sphere (x, y, z) coordinates
    ('x', 'f8'),  ('y', 'f8'), ('z', 'f8'), 
    # sphere Type 
    ('tYpe', 'u1'),
    # sphere (Vx, Vy, Vz) velocity 
    ('Vx', 'f8'),  ('Vy', 'f8'), ('Vz', 'f8'),
    # sphere (Vx, Vy, Vz) velocity 
    ('ax', 'f8'),  ('ay', 'f8'), ('az', 'f8'),
    #angular velocity
    ('Wx', 'f8'),  ('Wy', 'f8'), ('Wz', 'f8'),
    ], align=True) 
    Particle_nb = from_dtype(Particle)
    Points=[]
    lengthRange=int(matrixlength/r)
    widthRange=int(matrixwidth/r)
    for i in range (0,lengthRange+1):
        for j in range (-widthRange,widthRange+1):
            for k in range (-widthRange,widthRange+1):
                if math.sqrt((k*r)**2+(j*r)**2)<=matrixwidth:# and k==0:
                    if math.sqrt((k*r)**2+(j*r)**2)>fiberwidth or i*r<matrixlength-fiberlength:
                        if flag==0:
                            if i==0 or math.sqrt((k*r)**2+(j*r)**2)>=matrixwidth-0.5*r:
                               Points.append([i*r,j*r,k*r,0])
                            else:
                                Points.append([i*r,j*r,k*r,1])
                        if flag==1:
                            if i==0:
                               Points.append([i*r,j*r,k*r,0])
                            else:
                                Points.append([i*r,j*r,k*r,1])
                                print(1)
                    else:
                        Points.append([i*r,j*r,k*r,2])
#                        print(2)
                if i==lengthRange and math.sqrt((k*r)**2+(j*r)**2)<=fiberwidth:
                    Points.append([(i+1)*r,j*r,k*r,3])
    Particles=np.empty(len(Points),dtype=Particle_nb)
    i=0
    for x in Points:
        Particles[i]['x']=x[0]
        Particles[i]['y']=x[1]
        Particles[i]['z']=x[2]
        Particles[i]['tYpe']=x[3]
        Particles[i]['Vx']=0.0
        Particles[i]['Vy']=0.0
        Particles[i]['Vz']=0.0
        Particles[i]['ax']=0.0
        Particles[i]['ay']=0.0
        Particles[i]['az']=0.0
        Particles[i]['Wx']=0.0
        Particles[i]['Wy']=0.0
        Particles[i]['Wz']=0.0
        i+=1
    return Particles

def Prt_propgen(radii,mass,vel,damp,rotation):
    ParticleProp = np.dtype([
    ('r', 'f8'),  ('mass', 'f8'), ('I', 'f8'), 
    ('damp', 'f8'),
    ('Bx', 'f8'),  ('By', 'f8'), ('Bz', 'f8'), 
    ('BCondition', 'f8'),
    ('BWx', 'f8'),  ('BWy', 'f8'), ('BWz', 'f8'), 
    ('BWCondition', 'f8'),], align=True)
    ParticleProp_nb = from_dtype(ParticleProp)
    Pprop=np.empty(len(radii),dtype=ParticleProp_nb)
    Prtprop=[]
    pi=3.14159
    for i in range(0,len(radii)):
        ms=(4/3)*pi*mass[i]*(radii[i]/2)**3
        I=0.4*ms*(radii[i]/2)**2
        P=[]
        P.append(ms)
        P.append(I)
        P.append(radii[i]/2)
        P.append(damp[i])
        for l in vel[i]:
            P.append(l)
        for k in rotation[i]:
            P.append(k)
        Prtprop.append(P)
    i=0
    for x in Prtprop:
        Pprop[i]['mass']=x[0]
        Pprop[i]['I']=x[1]
        Pprop[i]['r']=x[2]
        Pprop[i]['damp']=x[3]
        Pprop[i]['BCondition']=x[4]
        Pprop[i]['Bx']=x[5]
        Pprop[i]['By']=x[6]
        Pprop[i]['Bz']=x[7]
        Pprop[i]['BWCondition']=x[8]
        Pprop[i]['BWx']=x[9]
        Pprop[i]['BWy']=x[10]
        Pprop[i]['BWz']=x[11]
        i+=1
    return Pprop

def Prt_propgen_Stiffness(radii,mass,vel,damp,rotation,E,G):
    ParticleProp = np.dtype([
    ('r', 'f8'),  ('mass', 'f8'), ('I', 'f8'), 
    ('damp', 'f8'),
    ('Bx', 'f8'),  ('By', 'f8'), ('Bz', 'f8'), 
    ('BCondition', 'f8'),
    ('BWx', 'f8'),  ('BWy', 'f8'), ('BWz', 'f8'), 
    ('BWCondition', 'f8'), ('Kn', 'f8'), ('Ks', 'f8'),], align=True)
    ParticleProp_nb = from_dtype(ParticleProp)
    Pprop=np.empty(len(radii),dtype=ParticleProp_nb)
    Prtprop=[]
    pi=3.14159
    for i in range(0,len(radii)):
        ms=(4/3)*pi*mass[i]*(radii[i]/2)**3
        I=0.4*ms*(radii[i]/2)**2
        P=[]
        P.append(ms)
        P.append(I)
        P.append(radii[i]/2)
        P.append(damp[i])
        for l in vel[i]:
            P.append(l)
        for k in rotation[i]:
            P.append(k)
        P.append(E[i])
        P.append(G[i])
        Prtprop.append(P)
    i=0
    for x in Prtprop:
        Pprop[i]['mass']=x[0]
        Pprop[i]['I']=x[1]
        Pprop[i]['r']=x[2]
        Pprop[i]['damp']=x[3]
        Pprop[i]['BCondition']=x[4]
        Pprop[i]['Bx']=x[5]
        Pprop[i]['By']=x[6]
        Pprop[i]['Bz']=x[7]
        Pprop[i]['BWCondition']=x[8]
        Pprop[i]['BWx']=x[9]
        Pprop[i]['BWy']=x[10]
        Pprop[i]['BWz']=x[11]
        Pprop[i]['Kn']=x[12]
        Pprop[i]['Ks']=x[13]
        i+=1
    return Pprop

                        
def Bnd_propgen(radii,E,Poisson,Strength):
    BondProp = np.dtype([
    ('A', 'f8'),  ('I', 'f8'), ('J', 'f8'),
    ('Sm', 'f8'),  ('kr', 'f8'), 
    ('Kn', 'f8'), ('Ks', 'f8'),('R', 'f8'),
    ('e0', 'f8'), ('ef', 'f8'),('sig', 'f8'),], align=True)
    BondProp_nb = from_dtype(BondProp)
    #Bndprop=np.zeros((35,11))
    pi=3.14159
    typeTable=np.zeros((11,11))
    cntr=0
    for i in range (0,11):
          for j in range (i,11):  
              typeTable[i,j]=cntr
              typeTable[j,i]=cntr
              cntr+=1
    Bndprop=np.zeros((int(cntr),11),dtype=np.float32)
    BProp=np.empty(int(cntr),dtype=BondProp_nb)
    for i in range(0,11):
        for j in range(0,11):
            Area=pi*((radii[i]/2.0+radii[j]/2.0)/2.0)**2
            I=0.25*pi*((radii[i]/2.0+radii[j]/2.0)/2.0)**4
            J=0.5*pi*((radii[i]/2.0+radii[j]/2.0)/2.0)**4
            EEq=1/((1-Poisson[i]**2)/E[i]+(1-Poisson[j]**2)/E[j])
            GEq=1/(((2*(1+Poisson[i]))*(2-Poisson[i]))/E[i]+(2*(1+Poisson[j])*(2-Poisson[j]))/E[j])
            Kn=EEq/(radii[i]/2.0+radii[j]/2.0)
            Kt=6.0*GEq/(radii[i]/2.0+radii[j]/2.0)
            maxStrength=Strength[i,j,0]
            Softening=Strength[i,j,1]
            R=(radii[i]/2+radii[j]/2)/2
            ind=int(typeTable[i,j])
            Bndprop[ind,0]=Area
            Bndprop[ind,1]=I
            Bndprop[ind,2]=J
            Bndprop[ind,3]=Kn
            Bndprop[ind,4]=Kt
            Bndprop[ind,5]=maxStrength
            Bndprop[ind,6]=Softening
            Bndprop[ind,7]=R
    i=0
    for x in Bndprop:
        BProp[i]['A']=x[0]
        BProp[i]['I']=x[1]
        BProp[i]['J']=x[2]
        BProp[i]['Kn']=float(x[3])
        BProp[i]['Ks']=float(x[4])
        BProp[i]['Sm']=x[5]
        BProp[i]['kr']=x[6]
        BProp[i]['R']=x[7]
        BProp[i]['e0']=x[8]
        BProp[i]['ef']=x[9]
        BProp[i]['sig']=x[10]
        i+=1
    return BProp

def Bnd_propgenList(radii,E,Poisson,Strength,mod,KtB):
#    e0=1e-3
#    ef=3e-3
    e0=1e-4
#    ef=4e-3
    ef=3e-3
    sig=2
    pi=3.14159
    typeTable=np.zeros((11,11))
#    cntr=0
    for i in range (0,11):
          for j in range (i,11):  
              a=i
              b=j
              if a>b:
                  a=j
                  b=i
                  
              typeTable[i,j]=10*a+b
              typeTable[j,i]=10*a+b
              
#    print(typeTable)
#    print(1/0)
    Bndprop=np.zeros((111,11),dtype=np.float64)
    for i in range(0,11):
        for j in range(0,11):
            Area=pi*((radii[i]/2.0+radii[j]/2.0)/2.0)**2
            I=0.25*pi*((radii[i]/2.0+radii[j]/2.0)/2.0)**4
            J=0.5*pi*((radii[i]/2.0+radii[j]/2.0)/2.0)**4
            EEq=1/((1-Poisson[i]**2)/E[i]+(1-Poisson[j]**2)/E[j])
#            EEq=100000000
#            GEq=1/(((2*(1+Poisson[i]))*(2-Poisson[i]))/E[i]+(2*(1+Poisson[j])*(2-Poisson[j]))/E[j])
#            GEq=100000000
            Kn=(mod[i][0]+mod[j][0])/2.*EEq/(radii[i]/2.0+radii[j]/2.0)
#            Kn=(mod[i][0]+mod[j][0])/2.*radii[i]*radii[j]*E[i]*E[j]/(radii[i]*E[i]+E[j]*radii[j])
#            Kt=mod[1]*(3.*(1-Poisson[j]))/(2-Poisson[j])*Kn
            Kt=(mod[i][1]+mod[j][1])/2.*Kn
#            
#            Kt=(mod[i][1]+mod[j][1])/2.*6.0*GEq/(radii[i]/2.0+radii[j]/2.0)
#            print(Kn,Kt)
            maxStrength=Strength[i,j,0]
            Softening=Strength[i,j,1]
            R=(radii[i]/2.+radii[j]/2.)/2.
            ind=int(typeTable[i,j])
            if ind==12:
                if radii[i]==0.5:
                    Area=0.2742*Area#r=0.5
#                    Area=0.349727*Area#r=0.5
                if radii[i]==0.7:
#                    Area=0.620781*Area#r=0.5
                    Area=0.342276*Area#r=0.5
#                    Area=0.409187*Area#r=0.5
                if radii[i]==1:
                    Area=0.2454*Area#r=0.5
#                    Area=0.31008*Area#r=0.5
                if radii[i]==2:
                    Area=0.113208*Area#r=0.5
                Kn=0*Kn
                Kt=KtB
            Bndprop[ind,0]=Area
            Bndprop[ind,1]=I
            Bndprop[ind,2]=J
#            Bndprop[ind,1]=0
#            Bndprop[ind,2]=0
            Bndprop[ind,3]=Kn
            Bndprop[ind,4]=Kt
            Bndprop[ind,5]=maxStrength
            Bndprop[ind,6]=Softening
            Bndprop[ind,7]=R
            Bndprop[ind,8]=e0
            Bndprop[ind,9]=ef
            Bndprop[ind,10]=sig
            # print(ind,Bndprop[ind,:])
    # print('Area',Area)
    Bndprop=np.array(Bndprop)
    # print(1/0)
    return Bndprop
    
def Prt_propgenList(radii,mass,vel,damp,rotation,Bndprop):
    Prtprop=[]
    pi=3.14159
#    print(Bndprop.shape)
#    print(len(radii))
    typeTable=np.zeros((11,11),dtype=np.int32)
    cntr=0
    for i in range (0,11):
          for j in range (i,11):  
              typeTable[i,j]=cntr
              typeTable[j,i]=cntr
              cntr+=1
    for i in range(0,len(radii)):
        ms=(4/3)*pi*mass[i]*(radii[i]/2.)**3
        I=0.4*ms*(radii[i]/2.)**2
        P=[]
        P.append(ms)
        P.append(I)
        P.append(radii[i]/2.)
        P.append(damp[i])
        for l in vel[i]:
            P.append(l)
        for k in rotation[i]:
            P.append(k)
#        print(typeTable[i,i])
#        P.append(0.5*2.*math.sqrt(Bndprop[typeTable[i,i],3]*ms))
#        P.append(0.5*2.*(radii[i]/2.)**2*math.sqrt(2*Bndprop[typeTable[i,i],4]*ms/5))
        P.append(1)
        P.append(1)
#        print('tc', 2*math.sqrt(2*ms/(Bndprop[typeTable[i,i],4]*5)))
        Prtprop.append(P)
    Prtprop=np.array(Prtprop,dtype=np.float64)
#    for pp in Prtprop:
#        print(pp)
#    print(1/0)
    return Prtprop
