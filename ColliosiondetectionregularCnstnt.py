# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:03:35 2018

@author: dressler
"""

import numpy as np
from numba import cuda, int16, float32, from_dtype, jit
import numba as nb

def makeglobalsCollision(Pproperties,hashdata,hashcreationdata):
    global Pprp
    global hd
    global hcd
    Pprp=Pproperties
    hd=hashdata
    hcd=hashcreationdata
    
def get_sizes(Prt,Pprop):
    xmax=np.max(Prt[:]['x'])
    ymax=np.max(Prt[:]['y'])
    zmax=np.max(Prt[:]['z'])
    xmin=np.min(Prt[:]['x'])
    ymin=np.min(Prt[:]['y'])
    zmin=np.min(Prt[:]['z'])
    nC=4*np.max(Pprop[:,2])
    widthx=int((xmax-xmin)/nC)+1
    widthy=int((ymax-ymin)/nC)+1
    widthz=int((zmax-zmin)/nC)+1
    return xmin,ymin,zmin,widthx,widthy,widthz,nC

#PointVec=np.zeros((widthx*widthy*widthz)*150+150,dtype=np.int32)
#PointVec.fill(-1)
#PointNum=np.zeros(widthx*widthy*widthz+1,dtype=np.int32)
#Sizes=np.array([xmin,ymin,zmin,widthx,widthy,widthz,150,nC])

@cuda.jit
def update_Pprop(Pprop):
    Pprop[3].BCondition=0.
    Pprop[3].Bx=0.
    Pprop[3].By=0.
    Pprop[3].Bz=0
    
    

def TableGenCPU(PVec,PNum,Prt,C):
#    print(C)
    for i in range (0,Prt.shape[0]):
#        if i< Prt.shape[0]:
        p=Prt[i]
        xx=int((p['x']-C[0])/C[7])
        yy=int((p['y']-C[1])/C[7])
        zz=int((p['z']-C[2])/C[7])
        hsh=zz*C[3]*C[4]+xx*C[4]+yy
#        print(xx,yy,zz,hsh)
        hsh=int(hsh)
        
#        print(hsh)
#        print()
        PNum[hsh]+=1
#        for ii in range (-1,2):
#            for jj in range (-1,2):
#                for kk in range (-1,2):
#                    if zz+ii>=0 and zz+ii<C[5] and xx+jj>=0 and xx+jj<C[3] and yy+kk>=0 and yy+kk<C[4]:
#                        hsh=(zz+ii)*C[3]*C[4]+(xx+jj)*C[5]+yy+kk
#                        print(xx,yy,zz,hsh)
#                        hsh=int(hsh)
#                        
#                        print(hsh)
#                        print()
#                        PNum[hsh]+=1
#                            pos=cuda.atomic.add(PNum,hsh,1)
    return PNum
                            
@cuda.jit
def TableGen(PVec,PNum,Prt,C):
    i=cuda.grid(1)
    if i< Prt.shape[0]:
        p=Prt[i]
        xx=int((p.x-C[0])/C[7])
        yy=int((p.y-C[1])/C[7])
        zz=int((p.z-C[2])/C[7])
        for ii in range (-1,2):
            for jj in range (-1,2):
                for kk in range (-1,2):
                    if zz+ii>=0 and zz+ii<C[5] and xx+jj>=0 and xx+jj<C[3] and yy+kk>=0 and yy+kk<C[4]:
                        hsh=(zz+ii)*C[3]*C[4]+(xx+jj)*C[4]+yy+kk
#                        hsh=int(hsh)
                        pos=cuda.atomic.add(PNum,hsh,1)
                        ix=C[6]*hsh+pos
                        inx=int(ix)
                        PVec[inx]=i
        cuda.syncthreads()

                   
#@cuda.jit (debug=True)
#def collisionDetect(Prt,Bnds,PVec,GCounter,C):
#    Pprop=cuda.const.array_like(Pprp)
#    rp=2
#    i=cuda.grid(1)
#    if i<Prt.shape[0]:
#        p1=Prt[i]
#        xx=int((p1.x-C[0])/C[7])
#        yy=int((p1.y-C[1])/C[7])
#        zz=int((p1.z-C[2])/C[7])
#        hsh=zz*C[3]*C[4]+xx*C[4]+yy 
#        Strt=int(C[6]*hsh)
#        for j in range(Strt,Strt+C[6]):
#            if j<PVec.shape[0]:
#                k=PVec[j]
#                if k>=0 and i>k:
#                    p2=Prt[k]
#                    l=((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)**0.5        
#                    if l<=2*(Pprop[p1.tYpe,rp]+Pprop[p1.tYpe,rp]):
#                        pos=cuda.atomic.add(GCounter,0,1)
#                        Bnds[pos].P1=i
#                        Bnds[pos].P2=k
#                        
#                        Bnds[pos].L=l
#                        Bnds[pos].Unx=0.
#                        Bnds[pos].Uny=0.
#                        Bnds[pos].Unz=0.
#                        
#                        Bnds[pos].Usx=0.
#                        Bnds[pos].Usy=0.
#                        Bnds[pos].Usz=0.
#                        
#                        Bnds[pos].Wnx=0.
#                        Bnds[pos].Wny=0.
#                        Bnds[pos].Wnz=0.
#                        
#                        Bnds[pos].Wsx=0.
#                        Bnds[pos].Wsy=0.
#                        Bnds[pos].Wsz=0.
#                    
#                        Bnds[pos].D=1.
#                        p1type=p1.tYpe
#                        p2type=p2.tYpe
#                        if p1.tYpe>p2.tYpe:
#                            p1type=p2.tYpe
#                            p2type=p1.tYpe
#                        Bnds[pos].tYpe=p1type*10+p2type
#                              
#                        Bnds[pos].Nx=(p2.x-p1.x)/l 
#                        Bnds[pos].Ny=(p2.y-p1.y)/l 
#                        Bnds[pos].Nz=(p2.z-p1.z)/l 
#                        
#                        Bnds[pos].MaxK=0.

@cuda.jit (debug=True)
def collisionDetectSize(Prt,GCounter):
    Pprop=cuda.const.array_like(Pprp)
    d_hd=cuda.const.array_like(hd)
    d_hcd=cuda.const.array_like(hcd)
    rp=2
    minx=0
    miny=1
    minz=2
    wx=3
    wy=4
    wz=5
    cellsize=6
    i=cuda.grid(1)
    if i<Prt.shape[0]:
        p1=Prt[i]
#        hsh=p1.hash
        xx=int((p1.x-d_hcd[minx])/d_hcd[cellsize])
        yy=int((p1.y-d_hcd[miny])/d_hcd[cellsize])
        zz=int((p1.z-d_hcd[minz])/d_hcd[cellsize])
        for ii in range (-1,2):
            for jj in range (-1,2):
                for kk in range (-1,2):
                    if zz+ii>=0 and zz+ii<=d_hcd[wz] and xx+jj>=0 and xx+jj<=d_hcd[wx] and yy+kk>=0 and yy+kk<=d_hcd[wy]:
                        hsh1=int((zz+ii)*d_hcd[wx]*d_hcd[wy]+(xx+jj)*d_hcd[wz]+yy+kk)
#                        if hsh1>=hsh:
                        for j in range(d_hd[0,hsh1],d_hd[0,hsh1]+d_hd[1,hsh1]+1):
                            if i>j:
                                p2=Prt[j]
                                l=((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)**0.5        
                                if l<=1.05*(Pprop[p1.tYpe,rp]+Pprop[p1.tYpe,rp]) and l>0:
                                    cuda.atomic.add(GCounter,0,1)

@cuda.jit (debug=True)
def collisionDetect(Prt,Bnds,GCounter):
    Pprop=cuda.const.array_like(Pprp)
    d_hd=cuda.const.array_like(hd)
    d_hcd=cuda.const.array_like(hcd)
    rp=2
    minx=0
    miny=1
    minz=2
    wx=3
    wy=4
    wz=5
    cellsize=6
    i=cuda.grid(1)
    if i<Prt.shape[0]:
        p1=Prt[i]
#        hsh=p1.hash
        xx=int((p1.x-d_hcd[minx])/d_hcd[cellsize])
        yy=int((p1.y-d_hcd[miny])/d_hcd[cellsize])
        zz=int((p1.z-d_hcd[minz])/d_hcd[cellsize])
        for ii in range (-1,2):
            for jj in range (-1,2):
                for kk in range (-1,2):
                    if zz+ii>=0 and zz+ii<=d_hcd[wz] and xx+jj>=0 and xx+jj<=d_hcd[wx] and yy+kk>=0 and yy+kk<=d_hcd[wy]:
                        hsh1=int((zz+ii)*d_hcd[wx]*d_hcd[wy]+(xx+jj)*d_hcd[wz]+yy+kk)
#                        if hsh1>=hsh:
                        for j in range(d_hd[0,hsh1],d_hd[0,hsh1]+d_hd[1,hsh1]+1):
                            if i>j:
                                p2=Prt[j]
                                l=((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)**0.5        
                                if l<=1.05*(Pprop[p1.tYpe,rp]+Pprop[p1.tYpe,rp]) and l>0:
                                    pos=cuda.atomic.add(GCounter,0,1)
                                    Bnds[pos].P1=i
                                    Bnds[pos].P2=j
                                    
                                    Bnds[pos].L=l
                                    Bnds[pos].Unx=0.
                                    Bnds[pos].Uny=0.
                                    Bnds[pos].Unz=0.
                                    
                                    Bnds[pos].Usx=0.
                                    Bnds[pos].Usy=0.
                                    Bnds[pos].Usz=0.
                                    
                                    Bnds[pos].Wnx=0.
                                    Bnds[pos].Wny=0.
                                    Bnds[pos].Wnz=0.
                                    
                                    Bnds[pos].Wsx=0.
                                    Bnds[pos].Wsy=0.
                                    Bnds[pos].Wsz=0.
                                
                                    Bnds[pos].D=1.
                                    p1type=p1.tYpe
                                    p2type=p2.tYpe
                                    if p1.tYpe>p2.tYpe:
                                        
                                        p1type=p2.tYpe
                                        p2type=p1.tYpe
                                    Bnds[pos].tYpe=p1type*10+p2type
                                          
                                    Bnds[pos].Nx=(p2.x-p1.x)/l 
                                    Bnds[pos].Ny=(p2.y-p1.y)/l 
                                    Bnds[pos].Nz=(p2.z-p1.z)/l 
                                    
                                    Bnds[pos].MaxK=0.
                                    Bnds[pos].X0=(p1.x-p2.x)

@cuda.jit (debug=True)
def collisionDetect2(Prt,Bnds,GCounter):
    Pprop=cuda.const.array_like(Pprp)
    d_hd=cuda.const.array_like(hd)
    d_hcd=cuda.const.array_like(hcd)
    rp=2
    minx=0
    miny=1
    minz=2
    wx=3
    wy=4
    wz=5
    cellsize=6
    i=cuda.grid(1)
    if i<Prt.shape[0]:
        p1=Prt[i]
#        hsh=p1.hash
        xx=int((p1.x-d_hcd[minx])/d_hcd[cellsize])
        yy=int((p1.y-d_hcd[miny])/d_hcd[cellsize])
        zz=int((p1.z-d_hcd[minz])/d_hcd[cellsize])
        for ii in range (-1,2):
            for jj in range (-1,2):
                for kk in range (-1,2):
                    if zz+ii>=0 and zz+ii<=d_hcd[wz] and xx+jj>=0 and xx+jj<=d_hcd[wx] and yy+kk>=0 and yy+kk<=d_hcd[wy]:
                        hsh1=int((zz+ii)*d_hcd[wx]*d_hcd[wy]+(xx+jj)*d_hcd[wz]+yy+kk)
#                        if hsh1>=hsh:
                        for j in range(d_hd[0,hsh1],d_hd[0,hsh1]+d_hd[1,hsh1]+1):
                            if i>j:
                                p2=Prt[j]
                                l=((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)**0.5        
                                if l<=1.05*(Pprop[p1.tYpe,rp]+Pprop[p1.tYpe,rp]):
                                    if p2.tYpe*p1.tYpe==2:
                                        check = 0
                                        if p1.tYpe==2 and p1.x>=p2.x:
                                            check =1
                                        if p1.tYpe==1 and p1.x<=p2.x:
                                            check =1
                                        if check==1:
                                            pos=cuda.atomic.add(GCounter,0,1)
                                            Bnds[pos].P1=i
                                            Bnds[pos].P2=j
                                            
                                            Bnds[pos].L=l
                                            Bnds[pos].Unx=0.
                                            Bnds[pos].Uny=0.
                                            Bnds[pos].Unz=0.
                                            
                                            Bnds[pos].Usx=0.
                                            Bnds[pos].Usy=0.
                                            Bnds[pos].Usz=0.
                                            
                                            Bnds[pos].Wnx=0.
                                            Bnds[pos].Wny=0.
                                            Bnds[pos].Wnz=0.
                                            
                                            Bnds[pos].Wsx=0.
                                            Bnds[pos].Wsy=0.
                                            Bnds[pos].Wsz=0.
                                        
                                            Bnds[pos].D=1.
                                            p1type=p1.tYpe
                                            p2type=p2.tYpe
                                            if p1.tYpe>p2.tYpe:
                                                p1type=p2.tYpe
                                                p2type=p1.tYpe
                                            Bnds[pos].tYpe=p1type*10+p2type
                                                  
                                            Bnds[pos].Nx=(p2.x-p1.x)/l 
                                            Bnds[pos].Ny=(p2.y-p1.y)/l 
                                            Bnds[pos].Nz=(p2.z-p1.z)/l 
                                            
                                            Bnds[pos].MaxK=0.
                                            Bnds[pos].X0=abs(p1.x-p2.x)
                                    else:
                                        pos=cuda.atomic.add(GCounter,0,1)
                                        Bnds[pos].P1=i
                                        Bnds[pos].P2=j
                                        
                                        Bnds[pos].L=l
                                        Bnds[pos].Unx=0.
                                        Bnds[pos].Uny=0.
                                        Bnds[pos].Unz=0.
                                        
                                        Bnds[pos].Usx=0.
                                        Bnds[pos].Usy=0.
                                        Bnds[pos].Usz=0.
                                        
                                        Bnds[pos].Wnx=0.
                                        Bnds[pos].Wny=0.
                                        Bnds[pos].Wnz=0.
                                        
                                        Bnds[pos].Wsx=0.
                                        Bnds[pos].Wsy=0.
                                        Bnds[pos].Wsz=0.
                                    
                                        Bnds[pos].D=1.
                                        p1type=p1.tYpe
                                        p2type=p2.tYpe
                                        if p1.tYpe>p2.tYpe:
                                            p1type=p2.tYpe
                                            p2type=p1.tYpe
                                        Bnds[pos].tYpe=p1type*10+p2type
                                              
                                        Bnds[pos].Nx=(p2.x-p1.x)/l 
                                        Bnds[pos].Ny=(p2.y-p1.y)/l 
                                        Bnds[pos].Nz=(p2.z-p1.z)/l 
                                        
                                        Bnds[pos].MaxK=0.
                                        Bnds[pos].X0=abs(p1.x-p2.x)
#@cuda.jit
#def collision(Prt,Bnds,PVec,GCounter,C,Pprp,d_Fx,d_Fy,d_Fz,d_Mx,d_My,d_Mz,d_dT):
#    i=cuda.grid(1)
#    dt=d_dT[0]
#    if i<Prt.shape[0]:
#        p1=Prt[i]
#        xx=int((p1.x-C[0])/C[7]+1)
#        yy=int((p1.y-C[1])/C[7]+1)
#        zz=int((p1.z-C[2])/C[7]+1)
#        hsh=(zz-1)*C[3]*C[4]+(xx-1)*C[5]+yy 
#        Strt=int(C[6]*(hsh-1))
#        for j in range(Strt,Strt+C[6]):
#            if j<PVec.shape[0]:
#                k=PVec[j]
#                if k>=0 and i>k:
#                    p2=Prt[k]
#                    l=(p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2        
#                    if l<((Pprp[p1.tYpe].r+Pprp[p1.tYpe].r)**2):
#                        dFnx,dFny,dFnz,dFsx,dFsy,dFsz,Mx1,My1,Mz1,Mx2,My2,Mz2=simple_contact(l,p1,p2,Pprp,dt)
#                        
#                        cuda.atomic.add(d_Fx,i,dFnx+dFsx)
#                        cuda.atomic.add(d_Fy,i,dFny+dFsy)
#                        cuda.atomic.add(d_Fz,i,dFnz+dFsz)
#                        
#                        cuda.atomic.add(d_Fx,k,-dFnx-dFsx)
#                        cuda.atomic.add(d_Fy,k,-dFny-dFsy)
#                        cuda.atomic.add(d_Fz,k,-dFnz-dFsz)
#                        
#                        cuda.atomic.add(d_Mx,i,Mx1)
#                        cuda.atomic.add(d_My,i,My1)
#                        cuda.atomic.add(d_Mz,i,Mz1)
#                        
#                        cuda.atomic.add(d_Mx,k,Mx2)
#                        cuda.atomic.add(d_My,k,My2)
#                        cuda.atomic.add(d_Mz,k,Mz2)
        
@cuda.jit
def simple_contact(l,p1,p2,Pprp,dt):
    l=l**0.5
    Nx=(p2.x-p1.x)/l 
    Ny=(p2.y-p1.y)/l 
    Nz=(p2.z-p1.z)/l 
    Cox=(p1.x+p2.x)/2.0
    Coy=(p1.y+p2.y)/2.0
    Coz=(p1.z+p2.z)/2.0
    Vrx=(p2.Vx+(p2.Wy*(Coz-p2.z)-p2.Wz*(Coy-p2.y)))-(p1.Vx+(p1.Wy*(Coz-p1.z)-p1.Wz*(Coy-p1.y)))
    Vry=(p2.Vy+(p2.Wz*(Cox-p2.x)-p2.Wx*(Coz-p2.z)))-(p1.Vy+(p1.Wz*(Cox-p1.x)-p1.Wx*(Coz-p1.z)))
    Vrz=(p2.Vz+(p2.Wx*(Coy-p2.y)-p2.Wy*(Cox-p2.x)))-(p1.Vz+(p1.Wx*(Coy-p1.y)-p1.Wy*(Cox-p1.x)))
    dUx=Vrx*dt
    dUy=Vry*dt
    dUz=Vrz*dt
    Un=l-(Pprp[p1.tYpe].r+Pprp[p2.tYpe].r)
    Udotted=dUx*Nx+dUy*Ny+dUz*Nz
    dUnx=Nx*Un
    dUny=Ny*Un
    dUnz=Nz*Un
    dUsx=dUx-Nx*Udotted
    dUsy=dUy-Ny*Udotted
    dUsz=dUz-Nz*Udotted
    
    Kn=(Pprp[p1.tYpe].Kn+Pprp[p2.tYpe].Kn)/2.0
    Ks=(Pprp[p1.tYpe].Ks+Pprp[p2.tYpe].Ks)/2.0
    A=0.25*(Pprp[p1.tYpe].r+Pprp[p2.tYpe].r)**2*3.14159
    
    dFnx=Kn*A*dUnx
    dFny=Kn*A*dUny
    dFnz=Kn*A*dUnz
    
    dFsx=Ks*A*dUsx
    dFsy=Ks*A*dUsy
    dFsz=Ks*A*dUsz
    
    Mx1=(Ny*Pprp[p1.tYpe].r)*dFsz-(Nz*Pprp[p1.tYpe].r)*dFsy
    My1=(Nz*Pprp[p1.tYpe].r)*dFsx-(Nx*Pprp[p1.tYpe].r)*dFsz
    Mz1=(Nx*Pprp[p1.tYpe].r)*dFsy-(Ny*Pprp[p1.tYpe].r)*dFsx
    
    Mx2=(Ny*Pprp[p2.tYpe].r)*dFsz-(Nz*Pprp[p2.tYpe].r)*dFsy
    My2=(Nz*Pprp[p2.tYpe].r)*dFsx-(Nx*Pprp[p2.tYpe].r)*dFsz
    Mz2=(Nx*Pprp[p2.tYpe].r)*dFsy-(Ny*Pprp[p2.tYpe].r)*dFsx
    return dFnx,dFny,dFnz,dFsx,dFsy,dFsz,Mx1,My1,Mz1,Mx2,My2,Mz2

@cuda.jit
def ResetTable(PVec,PNum,GCounter):
    i=cuda.grid(1)
    if i==0:
        GCounter[0]=0
    if i< PVec.shape[0]:
        PVec[i]=0
    if i< PNum.shape[0]:
        PNum[i]=0
