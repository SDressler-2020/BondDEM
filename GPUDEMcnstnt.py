# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 20:17:01 2018

@author: jodie
"""

import numpy as np
import math
from numba import cuda, int16, float32, from_dtype, jit
import numba as nb
import csv
from tkinter import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ColliosiondetectionregularCnstnt import *

def makeglobalsDem(Pproperties,Bproperties,dt):
    global Pprp
    global Bprp
    global dTT
    Pprp=Pproperties
    Bprp=Bproperties
    dTT=dt



@cuda.jit
def ContactHandler(Prt,Bnds,Fx_All,Fy_All,Fz_All,Mx_All,My_All,Mz_All):
    BProp=cuda.const.array_like(Bprp)
    A=0
    I=1
    J=2
    Kn=3
    Kt=4
    e0=8
    ef=9
    Pprop=cuda.const.array_like(Pprp)
    mass=0
#    Ip=1
    rp=2
    d_dT=cuda.const.array_like(dTT)
    dt=d_dT[0]
    index=cuda.grid(1)
    if index <Bnds.shape[0]:
        B=Bnds[index]
        if B.P1!=B.P2:
            B=Bnds[index]
            P1p=Prt[B.P1]
            P2p=Prt[B.P2]
            
            Nxo=B.Nx
            Nyo=B.Ny
            Nzo=B.Nz        
            
            l=((P1p.x-P2p.x)*(P1p.x-P2p.x)+(P1p.y-P2p.y)*(P1p.y-P2p.y)+(P1p.z-P2p.z)*(P1p.z-P2p.z))**0.5
            if l==0: l=1
                    
            B.Nx=(P2p.x-P1p.x)/l 
            B.Ny=(P2p.y-P1p.y)/l 
            B.Nz=(P2p.z-P1p.z)/l 
            
            Nxx=Nyo*B.Nz-Nzo*B.Ny
            Nyy=Nzo*B.Nx-Nxo*B.Nz
            Nzz=Nxo*B.Ny-Nyo*B.Nx
#            
            Utx1=(B.Usy*Nzz-B.Usz*Nyy)
            Uty1=(B.Usz*Nxx-B.Usx*Nzz)
            Utz1=(B.Usx*Nyy-B.Usy*Nxx)
            
#            Utx1=0
#            Uty1=0
#            Utz1=0
#            
            rotdotted=dt/2*((P1p.Wx+P2p.Wx)*B.Nx+(P1p.Wy+P2p.Wy)*B.Ny+(P1p.Wz+P2p.Wz)*B.Nz)
            rx=rotdotted*B.Nx
            ry=rotdotted*B.Ny
            rz=rotdotted*B.Nz
#            
            Utx2=(B.Usy*rz-B.Usz*ry)
            Uty2=(B.Usz*rx-B.Usx*rz)
            Utz2=(B.Usx*ry-B.Usy*rx)
            
#            Utx2=0
#            Uty2=0
#            Utz2=0

            DFn=0.3*math.sqrt(BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*2*Pprop[P1p.tYpe,mass])
            DMn=0.1*2*Pprop[Prt[B.P1].tYpe,rp]**2*math.sqrt(BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*2*Pprop[P1p.tYpe,mass]/5)
            
            Cox=(P1p.x+P2p.x)/2.0
            Coy=(P1p.y+P2p.y)/2.0
            Coz=(P1p.z+P2p.z)/2.0
            #print(index,P1p.Wx,P1p.Wy,P1p.Wz)
            
            Vrx=(P2p.Vx+(P2p.Wy*(Coz-P2p.z)-P2p.Wz*(Coy-P2p.y)))-(P1p.Vx+(P1p.Wy*(Coz-P1p.z)-P1p.Wz*(Coy-P1p.y)))
            Vry=(P2p.Vy+(P2p.Wz*(Cox-P2p.x)-P2p.Wx*(Coz-P2p.z)))-(P1p.Vy+(P1p.Wz*(Cox-P1p.x)-P1p.Wx*(Coz-P1p.z)))
            Vrz=(P2p.Vz+(P2p.Wx*(Coy-P2p.y)-P2p.Wy*(Cox-P2p.x)))-(P1p.Vz+(P1p.Wx*(Coy-P1p.y)-P1p.Wy*(Cox-P1p.x)))
            
#            Vrx=P2p.Vx-P1p.Vx
#            Vry=P2p.Vy-P1p.Vy
#            Vrz=P2p.Vz-P1p.Vz
            
            Vdotted=Vrx*B.Nx+Vry*B.Ny+Vrz*B.Nz
            
            dUx=Vrx*dt
            dUy=Vry*dt
            dUz=Vrz*dt
            
            Vnx=Vdotted*B.Nx
            Vny=Vdotted*B.Ny
            Vnz=Vdotted*B.Nz
            
            Vtx=Vrx-Vnx
            Vty=Vry-Vny
            Vtz=Vrz-Vnz
            
            
            #print(index,dUx,dUy,dUz)
            Udotted=dUx*B.Nx+dUy*B.Ny+dUz*B.Nz
            
            dUnx=B.Nx*Udotted
            dUny=B.Ny*Udotted
            dUnz=B.Nz*Udotted
            
            

#            dUsx=dUx-dUnx
#            dUsy=dUy-dUny
#            dUsz=dUz-dUnz    

            dUsx=Vtx*dt
            dUsy=Vty*dt
            dUsz=Vtz*dt
            
            U=-(B.L-l)
            
            dUnx=B.Nx*U
            dUny=B.Ny*U
            dUnz=B.Nz*U
            
            B.Unx=dUnx
            B.Uny=dUny
            B.Unz=dUnz
            
            
            
            B.Usx-=dUsx+Utx1+Utx2
            B.Usy-=dUsy+Uty1+Uty2
            B.Usz-=dUsz+Utz1+Utz2
            if B.tYpe==12:
                B.Usx=-((B.X0)-(P1p.x-P2p.x))
#                B.Usy=0
#                B.Usz=0
#            else:
#                B.Usx-=dUsx
#                B.Usy-=dUsy
#                B.Usz-=dUsz
#            B.Usx-=dUsx
#            B.Usy-=dUsy
#            B.Usz-=dUsz
            
            dmg=1
            if B.tYpe==12:
#                e=(B.Usx*B.Usx+B.Usy*B.Usy+B.Usz*B.Usz)**0.5
#                e=(B.Usx**2+B.Usy**2+B.Usz**2)**0.5
#                e=abs(B.Usx)
#                e=abs((B.X0)-(P1p.x-P2p.x))
                e=abs(B.X0-(P1p.x-P2p.x))
#                e=((B.Unx*B.Unx)/(B.L*B.L)+(B.Uny*B.Uny)/(B.L*B.L)+(B.Unz*B.Unz)/(B.L*B.L))**0.5+((B.Usx*B.Usx)/(B.L*B.L)+(B.Usy*B.Usy)/(B.L*B.L)+(B.Usz*B.Usz)/(B.L*B.L))**0.5 #damage causing strain (linear combination of normal and shear strain)
                if e>B.MaxK:
                    B.MaxK=e
                if B.MaxK>BProp[B.tYpe,e0]:
#                    B.D=1-(B.MaxK-BProp[B.tYpe,e0])/(BProp[B.tYpe,ef]-BProp[B.tYpe,e0])
                    B.D=1-(BProp[B.tYpe,ef]*(B.MaxK-BProp[B.tYpe,e0]))/(B.MaxK*(BProp[B.tYpe,ef]-BProp[B.tYpe,e0]))
#                    if B.D-D>0.000005:
#                        B.D-=0.000005
#                    else:
#                        B.D=D
#                    B.D=1-(BProp[B.tYpe,ef]*(B.MaxK-BProp[B.tYpe,e0]))/(B.MaxK*(BProp[B.tYpe,ef]-BProp[B.tYpe,e0]))
#                    B.D=(BProp[B.tYpe,e0]/B.MaxK)*math.exp(-(B.MaxK-BProp[B.tYpe,e0])/BProp[B.tYpe,ef])
                    
#                    if B.MaxK>BProp[B.tYpe,ef]:
#                        B.D=0
                    if B.D<0:
                        B.D=0
#            B.D=0
                        
    #        if n>0: # if bond in tension, this is a problem
            dmg=B.D
#            dmg=1
            Wrx=(P2p.Wx-P1p.Wx)
            Wry=(P2p.Wy-P1p.Wy)
            Wrz=(P2p.Wz-P1p.Wz)
    
            dMrx=dmg*DMn*Wrx
            dMry=dmg*DMn*Wry
            dMrz=dmg*DMn*Wrz
            
            dFnx=-dmg*DFn*Vnx
            dFny=-dmg*DFn*Vny
            dFnz=-dmg*DFn*Vnz
           
            dFtx=-dmg*DFn*Vtx
            dFty=-dmg*DFn*Vty
            dFtz=-dmg*DFn*Vtz
            
            dWx=Wrx*dt
            dWy=Wry*dt
            dWz=Wrz*dt
            
            Wdotted=dWx*B.Nx+dWy*B.Ny+dWz*B.Nz
            
            dWnx=B.Nx*Wdotted
            dWny=B.Ny*Wdotted
            dWnz=B.Nz*Wdotted
            
            dWsx=dWx-dWnx
            dWsy=dWy-dWny
            dWsz=dWz-dWnz
            
            B.Wnx+=dWnx
            B.Wny+=dWny
            B.Wnz+=dWnz
            
            B.Wsx+=dWsx
            B.Wsy+=dWsy
            B.Wsz+=dWsz
            
            Fnx=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unx
            Fny=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Uny
            Fnz=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unz
            
            Fsx=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usx
            Fsy=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usy
            Fsz=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usz
            #print(index,B.Unx,B.Usx,B.Uny,B.Usy,B.Unz,B.Usz,)
            
            Mnx=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,J]*B.Wnx
            Mny=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,J]*B.Wny
            Mnz=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,J]*B.Wnz
            
            Msx=-dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*B.Wsx
            Msy=-dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*B.Wsy
            Msz=-dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*B.Wsz
                
            Fx=Fnx+Fsx-dFnx-dFtx
            Fy=Fny+Fsy-dFny-dFty
            Fz=Fnz+Fsz-dFnz-dFtz
            
            cuda.atomic.add(Fx_All,B.P1,Fx)
            cuda.atomic.add(Fy_All,B.P1,Fy)
            cuda.atomic.add(Fz_All,B.P1,Fz)
            
            cuda.atomic.add(Fx_All,B.P2,-Fx)
            cuda.atomic.add(Fy_All,B.P2,-Fy)
            cuda.atomic.add(Fz_All,B.P2,-Fz)
            
            
            Mx=(B.Ny*Pprop[Prt[B.P1].tYpe,rp])*(Fsz-dFtz)-(B.Nz*Pprop[Prt[B.P1].tYpe,rp])*(Fsy-dFty)
            My=(B.Nz*Pprop[Prt[B.P1].tYpe,rp])*(Fsx-dFtx)-(B.Nx*Pprop[Prt[B.P1].tYpe,rp])*(Fsz-dFtz)
            Mz=(B.Nx*Pprop[Prt[B.P1].tYpe,rp])*(Fsy-dFty)-(B.Ny*Pprop[Prt[B.P1].tYpe,rp])*(Fsx-dFtx)
    
            cuda.atomic.add(Mx_All,B.P1,-Mnx-Msx+Mx-dMrx)
            cuda.atomic.add(My_All,B.P1,-Mny-Msy+My-dMry)
            cuda.atomic.add(Mz_All,B.P1,-Mnz-Msz+Mz-dMrz)

            cuda.atomic.add(Mx_All,B.P2,Mnx+Msx+Mx-dMrx)
            cuda.atomic.add(My_All,B.P2,Mny+Msy+My-dMry)
            cuda.atomic.add(Mz_All,B.P2,Mnz+Msz+Mz-dMrz)
            
            Bnds[index]=B


@cuda.jit
def ContactHandlerNoDMG(Prt,Bnds,Fx_All,Fy_All,Fz_All,Mx_All,My_All,Mz_All):
    BProp=cuda.const.array_like(Bprp)
    A=0
    I=1
    J=2
    Kn=3
    Kt=4
    e0=8
    ef=9
    Pprop=cuda.const.array_like(Pprp)
    mass=0
#    Ip=1
    rp=2
    d_dT=cuda.const.array_like(dTT)
    dt=d_dT[0]
    index=cuda.grid(1)
    if index <Bnds.shape[0]:
        B=Bnds[index]
        if B.P1!=B.P2:
            B=Bnds[index]
            P1p=Prt[B.P1]
            P2p=Prt[B.P2]
            
            Nxo=B.Nx
            Nyo=B.Ny
            Nzo=B.Nz        
            
            l=((P1p.x-P2p.x)*(P1p.x-P2p.x)+(P1p.y-P2p.y)*(P1p.y-P2p.y)+(P1p.z-P2p.z)*(P1p.z-P2p.z))**0.5
            if l==0: l=1
                    
            B.Nx=(P2p.x-P1p.x)/l 
            B.Ny=(P2p.y-P1p.y)/l 
            B.Nz=(P2p.z-P1p.z)/l 
            
            Nxx=Nyo*B.Nz-Nzo*B.Ny
            Nyy=Nzo*B.Nx-Nxo*B.Nz
            Nzz=Nxo*B.Ny-Nyo*B.Nx
#            
            Utx1=(B.Usy*Nzz-B.Usz*Nyy)
            Uty1=(B.Usz*Nxx-B.Usx*Nzz)
            Utz1=(B.Usx*Nyy-B.Usy*Nxx)
            
#            Utx1=0
#            Uty1=0
#            Utz1=0
#            
            rotdotted=dt/2*((P1p.Wx+P2p.Wx)*B.Nx+(P1p.Wy+P2p.Wy)*B.Ny+(P1p.Wz+P2p.Wz)*B.Nz)
            rx=rotdotted*B.Nx
            ry=rotdotted*B.Ny
            rz=rotdotted*B.Nz
#            
            Utx2=(B.Usy*rz-B.Usz*ry)
            Uty2=(B.Usz*rx-B.Usx*rz)
            Utz2=(B.Usx*ry-B.Usy*rx)
            
#            Utx2=0
#            Uty2=0
#            Utz2=0

            DFn=0.3*math.sqrt(BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*2*Pprop[P1p.tYpe,mass])
            DMn=0.1*2*Pprop[Prt[B.P1].tYpe,rp]**2*math.sqrt(BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*2*Pprop[P1p.tYpe,mass]/5)
            
            Cox=(P1p.x+P2p.x)/2.0
            Coy=(P1p.y+P2p.y)/2.0
            Coz=(P1p.z+P2p.z)/2.0
            #print(index,P1p.Wx,P1p.Wy,P1p.Wz)
            
            Vrx=(P2p.Vx+(P2p.Wy*(Coz-P2p.z)-P2p.Wz*(Coy-P2p.y)))-(P1p.Vx+(P1p.Wy*(Coz-P1p.z)-P1p.Wz*(Coy-P1p.y)))
            Vry=(P2p.Vy+(P2p.Wz*(Cox-P2p.x)-P2p.Wx*(Coz-P2p.z)))-(P1p.Vy+(P1p.Wz*(Cox-P1p.x)-P1p.Wx*(Coz-P1p.z)))
            Vrz=(P2p.Vz+(P2p.Wx*(Coy-P2p.y)-P2p.Wy*(Cox-P2p.x)))-(P1p.Vz+(P1p.Wx*(Coy-P1p.y)-P1p.Wy*(Cox-P1p.x)))
            
#            Vrx=P2p.Vx-P1p.Vx
#            Vry=P2p.Vy-P1p.Vy
#            Vrz=P2p.Vz-P1p.Vz
            
            Vdotted=Vrx*B.Nx+Vry*B.Ny+Vrz*B.Nz
            
            dUx=Vrx*dt
            dUy=Vry*dt
            dUz=Vrz*dt
            
            Vnx=Vdotted*B.Nx
            Vny=Vdotted*B.Ny
            Vnz=Vdotted*B.Nz
            
            Vtx=Vrx-Vnx
            Vty=Vry-Vny
            Vtz=Vrz-Vnz
            
            
            #print(index,dUx,dUy,dUz)
            Udotted=dUx*B.Nx+dUy*B.Ny+dUz*B.Nz
            
            dUnx=B.Nx*Udotted
            dUny=B.Ny*Udotted
            dUnz=B.Nz*Udotted
            
            

#            dUsx=dUx-dUnx
#            dUsy=dUy-dUny
#            dUsz=dUz-dUnz    

            dUsx=Vtx*dt
            dUsy=Vty*dt
            dUsz=Vtz*dt
            
            U=-(B.L-l)
            
            dUnx=B.Nx*U
            dUny=B.Ny*U
            dUnz=B.Nz*U
            
            B.Unx=dUnx
            B.Uny=dUny
            B.Unz=dUnz
            
            
            
            B.Usx-=dUsx+Utx1+Utx2
            B.Usy-=dUsy+Uty1+Uty2
            B.Usz-=dUsz+Utz1+Utz2
            if B.tYpe==12:
                B.Usx=-((B.X0)-(P1p.x-P2p.x))
#                B.Usy=0
#                B.Usz=0
#            else:
#                B.Usx-=dUsx
#                B.Usy-=dUsy
#                B.Usz-=dUsz
#            B.Usx-=dUsx
#            B.Usy-=dUsy
#            B.Usz-=dUsz
            
#            dmg=1
#            if B.tYpe==12:
##                e=(B.Usx*B.Usx+B.Usy*B.Usy+B.Usz*B.Usz)**0.5
##                e=(B.Usx**2+B.Usy**2+B.Usz**2)**0.5
##                e=abs(B.Usx)
##                e=abs((B.X0)-(P1p.x-P2p.x))
#                e=abs(B.X0-(P1p.x-P2p.x))
##                e=((B.Unx*B.Unx)/(B.L*B.L)+(B.Uny*B.Uny)/(B.L*B.L)+(B.Unz*B.Unz)/(B.L*B.L))**0.5+((B.Usx*B.Usx)/(B.L*B.L)+(B.Usy*B.Usy)/(B.L*B.L)+(B.Usz*B.Usz)/(B.L*B.L))**0.5 #damage causing strain (linear combination of normal and shear strain)
#                if e>B.MaxK:
#                    B.MaxK=e
#                if B.MaxK>BProp[B.tYpe,e0]:
##                    B.D=1-(B.MaxK-BProp[B.tYpe,e0])/(BProp[B.tYpe,ef]-BProp[B.tYpe,e0])
#                    B.D=1-(BProp[B.tYpe,ef]*(B.MaxK-BProp[B.tYpe,e0]))/(B.MaxK*(BProp[B.tYpe,ef]-BProp[B.tYpe,e0]))
##                    if B.D-D>0.000005:
##                        B.D-=0.000005
##                    else:
##                        B.D=D
##                    B.D=1-(BProp[B.tYpe,ef]*(B.MaxK-BProp[B.tYpe,e0]))/(B.MaxK*(BProp[B.tYpe,ef]-BProp[B.tYpe,e0]))
##                    B.D=(BProp[B.tYpe,e0]/B.MaxK)*math.exp(-(B.MaxK-BProp[B.tYpe,e0])/BProp[B.tYpe,ef])
#                    
##                    if B.MaxK>BProp[B.tYpe,ef]:
##                        B.D=0
#                    if B.D<0:
#                        B.D=0
#            B.D=0
                        
    #        if n>0: # if bond in tension, this is a problem
            dmg=B.D
#            dmg=1
            Wrx=(P2p.Wx-P1p.Wx)
            Wry=(P2p.Wy-P1p.Wy)
            Wrz=(P2p.Wz-P1p.Wz)
    
            dMrx=dmg*DMn*Wrx
            dMry=dmg*DMn*Wry
            dMrz=dmg*DMn*Wrz
            
            dFnx=-dmg*DFn*Vnx
            dFny=-dmg*DFn*Vny
            dFnz=-dmg*DFn*Vnz
           
            dFtx=-dmg*DFn*Vtx
            dFty=-dmg*DFn*Vty
            dFtz=-dmg*DFn*Vtz
            
            dWx=Wrx*dt
            dWy=Wry*dt
            dWz=Wrz*dt
            
            Wdotted=dWx*B.Nx+dWy*B.Ny+dWz*B.Nz
            
            dWnx=B.Nx*Wdotted
            dWny=B.Ny*Wdotted
            dWnz=B.Nz*Wdotted
            
            dWsx=dWx-dWnx
            dWsy=dWy-dWny
            dWsz=dWz-dWnz
            
            B.Wnx+=dWnx
            B.Wny+=dWny
            B.Wnz+=dWnz
            
            B.Wsx+=dWsx
            B.Wsy+=dWsy
            B.Wsz+=dWsz
            
            Fnx=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unx
            Fny=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Uny
            Fnz=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unz
            
            Fsx=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usx
            Fsy=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usy
            Fsz=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usz
            #print(index,B.Unx,B.Usx,B.Uny,B.Usy,B.Unz,B.Usz,)
            
            Mnx=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,J]*B.Wnx
            Mny=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,J]*B.Wny
            Mnz=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,J]*B.Wnz
            
            Msx=-dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*B.Wsx
            Msy=-dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*B.Wsy
            Msz=-dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,I]*B.Wsz
                
            Fx=Fnx+Fsx-dFnx-dFtx
            Fy=Fny+Fsy-dFny-dFty
            Fz=Fnz+Fsz-dFnz-dFtz
#            if Prt[B.P1].tYpe !=3:
            cuda.atomic.add(Fx_All,B.P1,Fx)
            cuda.atomic.add(Fy_All,B.P1,Fy)
            cuda.atomic.add(Fz_All,B.P1,Fz)
#            if Prt[B.P2].tYpe !=3:
            cuda.atomic.add(Fx_All,B.P2,-Fx)
            cuda.atomic.add(Fy_All,B.P2,-Fy)
            cuda.atomic.add(Fz_All,B.P2,-Fz)
            
            
            Mx=(B.Ny*Pprop[Prt[B.P1].tYpe,rp])*(Fsz-dFtz)-(B.Nz*Pprop[Prt[B.P1].tYpe,rp])*(Fsy-dFty)
            My=(B.Nz*Pprop[Prt[B.P1].tYpe,rp])*(Fsx-dFtx)-(B.Nx*Pprop[Prt[B.P1].tYpe,rp])*(Fsz-dFtz)
            Mz=(B.Nx*Pprop[Prt[B.P1].tYpe,rp])*(Fsy-dFty)-(B.Ny*Pprop[Prt[B.P1].tYpe,rp])*(Fsx-dFtx)
    
            cuda.atomic.add(Mx_All,B.P1,-Mnx-Msx+Mx-dMrx)
            cuda.atomic.add(My_All,B.P1,-Mny-Msy+My-dMry)
            cuda.atomic.add(Mz_All,B.P1,-Mnz-Msz+Mz-dMrz)

            cuda.atomic.add(Mx_All,B.P2,Mnx+Msx+Mx-dMrx)
            cuda.atomic.add(My_All,B.P2,Mny+Msy+My-dMry)
            cuda.atomic.add(Mz_All,B.P2,Mnz+Msz+Mz-dMrz)
            
            Bnds[index]=B

@cuda.jit #(debug=True)
def Integrator(Prt,Fx_All,Fy_All,Fz_All,Mx_All,My_All,Mz_All):
    #this is the integrator, i am using a verlet leapfrog scheme which i understand might be unstable for small dt due to the single precision nature of the data
    # using the euler scheme would require me to store the previous timesteps acceleration though
    Pprop=cuda.const.array_like(Pprp)
    d_dT=cuda.const.array_like(dTT)
    mass=0
    Ip=1
#    rp=2
    damp=3
    BCondition=4
    Bx=5
    By=6
    Bz=7
    BWCondition=8
    BWx=9
    BWy=10
    BWz=11
#    DCn=12
#    DCt=13
    dt=d_dT[0]
    index=cuda.grid(1)
    if index<Prt.shape[0]:
#        Bnds[pos].tYpe=p1type*10+p2type
        P=Prt[index]
        Fxp=Fx_All[index]#-Pprop[P.tYpe,DCn]*P.Vx
        Fyp=Fy_All[index]#-Pprop[P.tYpe,DCn]*P.Vy
        Fzp=Fz_All[index]#-Pprop[P.tYpe,DCn]*P.Vz
        Mxp=Mx_All[index]#-Pprop[P.tYpe,DCt]*P.Wx
        Myp=My_All[index]#-Pprop[P.tYpe,DCt]*P.Wy
        Mzp=Mz_All[index]#-Pprop[P.tYpe,DCt]*P.Wz
        #accelerations
        #V=math.sqrt(P.Vx*P.Vx+P.Vy*P.Vy+P.Vz*P.Vz)
        dampfactor=0.07             
       
        accx=Fxp/Pprop[P.tYpe,mass] #new var
        accy=Fyp/Pprop[P.tYpe,mass] #new var
        accz=Fzp/Pprop[P.tYpe,mass] #new var
        
        estVx=P.Vx+accx*dt/2.
        estVy=P.Vy+accy*dt/2.
        estVz=P.Vz+accz*dt/2.
#        
        dFx=dampfactor*math.fabs(Fxp)
        dFy=dampfactor*math.fabs(Fyp)
        dFz=dampfactor*math.fabs(Fzp)
        
        if estVx>0:
            dFx=-dFx
        if estVy>0:
            dFy=-dFy
        if estVz>0:
            dFz=-dFz
#        
        accx=(Fxp+dFx)/Pprop[P.tYpe,mass] #new var
        accy=(Fyp+dFy)/Pprop[P.tYpe,mass] #new var
        accz=(Fzp+dFz)/Pprop[P.tYpe,mass] #new var
        #update positions
#        if P.tYpe ==0:
        P.x+=dt*P.Vx + 0.5*accx*dt*dt*Pprop[P.tYpe,BCondition]
        P.y+=dt*P.Vy + 0.5*accy*dt*dt*Pprop[P.tYpe,BCondition]
        P.z+=dt*P.Vz + 0.5*accz*dt*dt*Pprop[P.tYpe,BCondition]
#        else:
#            P.x+=dt*P.Vx + 0.5*accx*dt*dt
#            P.y+=dt*P.Vy + 0.5*accy*dt*dt
#            P.z+=dt*P.Vz + 0.5*accz*dt*dt
            
        #velocity update
        P.Vx+=0.5*dt*(accx+P.ax)
        P.Vy+=0.5*dt*(accy+P.ay)
        P.Vz+=0.5*dt*(accz+P.az)
       
        #boundary condition applications
        P.Vx=Pprop[P.tYpe,BCondition]*P.Vx+Pprop[P.tYpe,Bx] 
        P.Vy=Pprop[P.tYpe,BCondition]*P.Vy+Pprop[P.tYpe,By]
        P.Vz=Pprop[P.tYpe,BCondition]*P.Vz+Pprop[P.tYpe,Bz]
        if P.tYpe==0:
            P.Vx=0
        #store previous step acceleration
        P.ax=accx
        P.ay=accy
        P.az=accz
        
        #Rotation/Orientation, i should probably use quaternions here but i havent worked out the maths yet
        accx=Mxp/Pprop[P.tYpe,Ip] 
        accy=Myp/Pprop[P.tYpe,Ip] 
        accz=Mzp/Pprop[P.tYpe,Ip] 
        
        estVx=P.Wx+accx*dt/2
        estVy=P.Wy+accy*dt/2
        estVz=P.Wz+accz*dt/2
#        
        dFx=dampfactor*math.fabs(Mxp)
        dFy=dampfactor*math.fabs(Myp)
        dFz=dampfactor*math.fabs(Mzp)
#        
        if estVx>0:
            dFx=-Pprop[P.tYpe,damp]*math.fabs(Mxp)
        if estVy>0:
            dFy=-Pprop[P.tYpe,damp]*math.fabs(Myp)
        if estVz>0:
            dFz=-Pprop[P.tYpe,damp]*math.fabs(Mzp)
                    
        accx=(Mxp+dFx)/Pprop[P.tYpe,Ip]  #new var
        accy=(Myp+dFy)/Pprop[P.tYpe,Ip]  #new var
        accz=(Mzp+dFz)/Pprop[P.tYpe,Ip]  #new var
        
#        P.Wx+=dt*accx
#        P.Wy+=dt*accy
#        P.Wz+=dt*accz
        
        P.Wx+=dt*accx*Pprop[P.tYpe,BWCondition]
        P.Wy+=dt*accy*Pprop[P.tYpe,BWCondition]
        P.Wz+=dt*accz*Pprop[P.tYpe,BWCondition]
        
        P.Wx=Pprop[P.tYpe,BWCondition]*P.Wx+Pprop[P.tYpe,BWx] 
        P.Wy=Pprop[P.tYpe,BWCondition]*P.Wy+Pprop[P.tYpe,BWy]
        P.Wz=Pprop[P.tYpe,BWCondition]*P.Wz+Pprop[P.tYpe,BWz]
        #update particle and set all forces back to zero
        Prt[index]=P
        Fx_All[index]=0.0
        Fy_All[index]=0.0
        Fz_All[index]=0.0
        Mx_All[index]=0.0
        My_All[index]=0.0
        Mz_All[index]=0.0
#        if P.tYpe==2:
#            Fx_All[index]=2000000.0


@cuda.jit #(debug=True)
def IntegratorLocked(Prt,Fx_All,Fy_All,Fz_All,Mx_All,My_All,Mz_All):
    #this is the integrator, i am using a verlet leapfrog scheme which i understand might be unstable for small dt due to the single precision nature of the data
    # using the euler scheme would require me to store the previous timesteps acceleration though
    Pprop=cuda.const.array_like(Pprp)
    d_dT=cuda.const.array_like(dTT)
    mass=0
    Ip=1
#    rp=2
    damp=3
    BCondition=4
    Bx=5
    By=6
    Bz=7
    BWCondition=8
    BWx=9
    BWy=10
    BWz=11
#    DCn=12
#    DCt=13
    dt=d_dT[0]
    index=cuda.grid(1)
    if index<Prt.shape[0]:
#        Bnds[pos].tYpe=p1type*10+p2type
        P=Prt[index]
        Fxp=Fx_All[index]#-Pprop[P.tYpe,DCn]*P.Vx
        Fyp=Fy_All[index]#-Pprop[P.tYpe,DCn]*P.Vy
        Fzp=Fz_All[index]#-Pprop[P.tYpe,DCn]*P.Vz
        Mxp=Mx_All[index]#-Pprop[P.tYpe,DCt]*P.Wx
        Myp=My_All[index]#-Pprop[P.tYpe,DCt]*P.Wy
        Mzp=Mz_All[index]#-Pprop[P.tYpe,DCt]*P.Wz
        #accelerations
        #V=math.sqrt(P.Vx*P.Vx+P.Vy*P.Vy+P.Vz*P.Vz)
        dampfactor=0.8             
       
        accx=Fxp/Pprop[P.tYpe,mass] #new var
        accy=Fyp/Pprop[P.tYpe,mass] #new var
        accz=Fzp/Pprop[P.tYpe,mass] #new var
        
        estVx=P.Vx+accx*dt/2.
        estVy=P.Vy+accy*dt/2.
        estVz=P.Vz+accz*dt/2.
#        
        dFx=dampfactor*math.fabs(Fxp)
        dFy=dampfactor*math.fabs(Fyp)
        dFz=dampfactor*math.fabs(Fzp)
        
        if estVx>0:
            dFx=-dFx
        if estVy>0:
            dFy=-dFy
        if estVz>0:
            dFz=-dFz
#        
        accx=(Fxp+dFx)/Pprop[P.tYpe,mass] #new var
        accy=(Fyp+dFy)/Pprop[P.tYpe,mass] #new var
        accz=(Fzp+dFz)/Pprop[P.tYpe,mass] #new var
        #update positions
        if P.tYpe==0 or P.tYpe==3:
            P.x+=dt*P.Vx + 0.5*accx*dt*dt*Pprop[P.tYpe,BCondition]
            P.y+=dt*P.Vy + 0.5*accy*dt*dt#*Pprop[P.tYpe,BCondition]
            P.z+=dt*P.Vz + 0.5*accz*dt*dt#*Pprop[P.tYpe,BCondition]
        else:
            P.x+=dt*P.Vx + 0.5*accx*dt*dt
            P.y+=dt*P.Vy + 0.5*accy*dt*dt#*Pprop[P.tYpe,BCondition]
            P.z+=dt*P.Vz + 0.5*accz*dt*dt#*Pprop[P.tYpe,BCondition]
        
        #velocity update
        P.Vx+=0.5*dt*(accx+P.ax)
        P.Vy+=0.5*dt*(accy+P.ay)
        P.Vz+=0.5*dt*(accz+P.az)
       
        #boundary condition applications
        if P.tYpe==0 or P.tYpe==3:
            P.Vx=Pprop[P.tYpe,BCondition]*P.Vx-3*Pprop[P.tYpe,Bx] 
            P.Vy=Pprop[P.tYpe,BCondition]*P.Vy+Pprop[P.tYpe,By]
            P.Vz=Pprop[P.tYpe,BCondition]*P.Vz+Pprop[P.tYpe,Bz]
#        if P.tYpe==3 or P.tYpe==0:
#            P.Vx=0 
#            P.Vy=0
#            P.Vz=0
        
        #store previous step acceleration
        P.ax=accx
        P.ay=accy
        P.az=accz
        
        #Rotation/Orientation, i should probably use quaternions here but i havent worked out the maths yet
        accx=Mxp/Pprop[P.tYpe,Ip] 
        accy=Myp/Pprop[P.tYpe,Ip] 
        accz=Mzp/Pprop[P.tYpe,Ip] 
        
        estVx=P.Wx+accx*dt/2
        estVy=P.Wy+accy*dt/2
        estVz=P.Wz+accz*dt/2
#        
        dFx=dampfactor*math.fabs(Mxp)
        dFy=dampfactor*math.fabs(Myp)
        dFz=dampfactor*math.fabs(Mzp)
#        
        if estVx>0:
            dFx=-Pprop[P.tYpe,damp]*math.fabs(Mxp)
        if estVy>0:
            dFy=-Pprop[P.tYpe,damp]*math.fabs(Myp)
        if estVz>0:
            dFz=-Pprop[P.tYpe,damp]*math.fabs(Mzp)
                    
        accx=(Mxp+dFx)/Pprop[P.tYpe,Ip]  #new var
        accy=(Myp+dFy)/Pprop[P.tYpe,Ip]  #new var
        accz=(Mzp+dFz)/Pprop[P.tYpe,Ip]  #new var
        
#        P.Wx+=dt*accx
#        P.Wy+=dt*accy
#        P.Wz+=dt*accz
        
        P.Wx+=dt*accx*Pprop[P.tYpe,BWCondition]
        P.Wy+=dt*accy*Pprop[P.tYpe,BWCondition]
        P.Wz+=dt*accz*Pprop[P.tYpe,BWCondition]
        
        P.Wx=Pprop[P.tYpe,BWCondition]*P.Wx+Pprop[P.tYpe,BWx] 
        P.Wy=Pprop[P.tYpe,BWCondition]*P.Wy+Pprop[P.tYpe,BWy]
        P.Wz=Pprop[P.tYpe,BWCondition]*P.Wz+Pprop[P.tYpe,BWz]
        #update particle and set all forces back to zero
        Prt[index]=P
        Fx_All[index]=0.0
        Fy_All[index]=0.0
        Fz_All[index]=0.0
        Mx_All[index]=0.0
        My_All[index]=0.0
        Mz_All[index]=0.0

@cuda.jit
def ContactHandlerSimple(Prt,Bnds,Fx_All,Fy_All,Fz_All,Mx_All,My_All,Mz_All):
    BProp=cuda.const.array_like(Bprp)
    A=0
    I=1
    J=2
    Kn=3
    Kt=4
    e0=8
    ef=9
    Pprop=cuda.const.array_like(Pprp)
    mass=0
#    Ip=1
    rp=2
    d_dT=cuda.const.array_like(dTT)
    dt=d_dT[0]
    index=cuda.grid(1)
    if index <Bnds.shape[0]:
        B=Bnds[index]
        if B.P1!=B.P2:
            P1p=Prt[B.P1]
            P2p=Prt[B.P2]
            
            Nxo=B.Nx
            Nyo=B.Ny
            Nzo=B.Nz        
            
            l=math.sqrt((P1p.x-P2p.x)*(P1p.x-P2p.x)+(P1p.y-P2p.y)*(P1p.y-P2p.y)+(P1p.z-P2p.z)*(P1p.z-P2p.z))
            if l==0: l=1
                    
            B.Nx=(P2p.x-P1p.x)/l 
            B.Ny=(P2p.y-P1p.y)/l 
            B.Nz=(P2p.z-P1p.z)/l 
            
            Nxx=Nyo*B.Nz-Nzo*B.Ny
            Nyy=Nzo*B.Nx-Nxo*B.Nz
            Nzz=Nxo*B.Ny-Nyo*B.Nx
            
            Utx1=-(B.Usy*Nzz-B.Usz*Nyy)
            Uty1=-(B.Usz*Nxx-B.Usx*Nzz)
            Utz1=-(B.Usx*Nyy-B.Usy*Nxx)
            
            rotdotted=dt/2*((P1p.Wx+P2p.Wx)*B.Nx+(P1p.Wy+P2p.Wy)*B.Ny+(P1p.Wz+P2p.Wz)*B.Nz)
            rx=rotdotted*B.Nx
            ry=rotdotted*B.Ny
            rz=rotdotted*B.Nz
            
            Utx2=-(B.Usy*rz-B.Usz*ry)
            Uty2=-(B.Usz*rx-B.Usx*rz)
            Utz2=-(B.Usx*ry-B.Usy*rx)
            
            DFn=0.9*math.sqrt(BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*2*Pprop[P1p.tYpe,mass])
            DMn=0.01*2*Pprop[Prt[B.P1].tYpe,rp]**2*math.sqrt(BProp[B.tYpe,Kt]*2*Pprop[P1p.tYpe,mass]/5)
            
            Cox=(P1p.x+P2p.x)/2.0
            Coy=(P1p.y+P2p.y)/2.0
            Coz=(P1p.z+P2p.z)/2.0
            #print(index,P1p.Wx,P1p.Wy,P1p.Wz)
            Vrx=(P2p.Vx+(P2p.Wy*(Coz-P2p.z)-P2p.Wz*(Coy-P2p.y)))-(P1p.Vx+(P1p.Wy*(Coz-P1p.z)-P1p.Wz*(Coy-P1p.y)))
            Vry=(P2p.Vy+(P2p.Wz*(Cox-P2p.x)-P2p.Wx*(Coz-P2p.z)))-(P1p.Vy+(P1p.Wz*(Cox-P1p.x)-P1p.Wx*(Coz-P1p.z)))
            Vrz=(P2p.Vz+(P2p.Wx*(Coy-P2p.y)-P2p.Wy*(Cox-P2p.x)))-(P1p.Vz+(P1p.Wx*(Coy-P1p.y)-P1p.Wy*(Cox-P1p.x)))
            
            Vdotted=Vrx*B.Nx+Vry*B.Ny+Vrz*B.Nz
            
            dUx=Vrx*dt
            dUy=Vry*dt
            dUz=Vrz*dt
            
            Vnx=Vdotted*B.Nx
            Vny=Vdotted*B.Ny
            Vnz=Vdotted*B.Nz
            
            Vtx=Vrx-Vnx
            Vty=Vry-Vny
            Vtz=Vrz-Vnz
                        
            Udotted=dUx*B.Nx+dUy*B.Ny+dUz*B.Nz
            
            U=-(B.L-l)
            
            B.Unx=B.Nx*U
            B.Uny=B.Ny*U
            B.Unz=B.Nz*U
            
            dUnx=B.Nx*Udotted
            dUny=B.Ny*Udotted
            dUnz=B.Nz*Udotted
            
            dUsx=dUx-dUnx
            dUsy=dUy-dUny
            dUsz=dUz-dUnz    
   
#            B.Usx-=dUsx+Utx1+Utx2
#            B.Usy-=dUsy+Uty1+Uty2
#            B.Usz-=dUsz+Utz1+Utz2
            
            B.Usx-=dUsx
            B.Usy-=dUsy
            B.Usz-=dUsz
            
#            dmg=1
    
            Wrx=(P2p.Wx+P1p.Wx)
            Wry=(P2p.Wy+P1p.Wy)
            Wrz=(P2p.Wz+P1p.Wz)
    
            
            
            dmg=1
            if B.tYpe==12 or B.tYpe==13:
#                e=abs(abs(B.X0)-abs(P1p.x-P2p.x))
                e=(B.Usx**2+B.Usy**2+B.Usz**2)**0.5
#                e=((B.Unx*B.Unx)/(B.L*B.L)+(B.Uny*B.Uny)/(B.L*B.L)+(B.Unz*B.Unz)/(B.L*B.L))**0.5+((B.Usx*B.Usx)/(B.L*B.L)+(B.Usy*B.Usy)/(B.L*B.L)+(B.Usz*B.Usz)/(B.L*B.L))**0.5 #damage causing strain (linear combination of normal and shear strain)
                if e>B.MaxK:
                    B.MaxK=e
                if B.MaxK>BProp[B.tYpe,e0]:
#                    B.D=1-(B.MaxK-BProp[B.tYpe,e0])/(BProp[B.tYpe,ef]-BProp[B.tYpe,e0])
                    B.D=1-(BProp[B.tYpe,ef]*(B.MaxK-BProp[B.tYpe,e0]))/(B.MaxK*(BProp[B.tYpe,ef]-BProp[B.tYpe,e0]))
#                    B.D=(BProp[B.tYpe,e0]/B.MaxK)*math.exp(-(B.MaxK-BProp[B.tYpe,e0])/BProp[B.tYpe,ef])
                    
#                    if B.MaxK>BProp[B.tYpe,ef]:
#                        B.D=0
                    if B.D<0:
                        B.D=0
#            B.D=0
                        
    #        if n>0: # if bond in tension, this is a problem
            dmg=B.D
            
            dMrx=dmg*DMn*Wrx
            dMry=dmg*DMn*Wry
            dMrz=dmg*DMn*Wrz
            
            dFtx=-dmg*DFn*Vtx
            dFty=-dmg*DFn*Vty
            dFtz=-dmg*DFn*Vtz
            
            dFnx=-dmg*DFn*Vnx
            dFny=-dmg*DFn*Vny
            dFnz=-dmg*DFn*Vnz
            
            Fnx=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unx
            Fny=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Uny
            Fnz=dmg*BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unz
           
            Fsx=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usx
            Fsy=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usy
            Fsz=-dmg*BProp[B.tYpe,Kt]*BProp[B.tYpe,A]*B.Usz
                
            Fx=Fnx+Fsx-dFnx-dFtx
            Fy=Fny+Fsy-dFny-dFty
            Fz=Fnz+Fsz-dFnz-dFtz
            

            
            cuda.atomic.add(Fx_All,B.P1,Fx)
            cuda.atomic.add(Fy_All,B.P1,Fy)
            cuda.atomic.add(Fz_All,B.P1,Fz)
              
            cuda.atomic.add(Fx_All,B.P2,-Fx)
            cuda.atomic.add(Fy_All,B.P2,-Fy)
            cuda.atomic.add(Fz_All,B.P2,-Fz)
            
            Mx=(B.Ny*Pprop[Prt[B.P1].tYpe,rp])*(Fsz-dFtz)-(B.Nz*Pprop[Prt[B.P1].tYpe,rp])*(Fsy-dFty)
            My=(B.Nz*Pprop[Prt[B.P1].tYpe,rp])*(Fsx-dFtx)-(B.Nx*Pprop[Prt[B.P1].tYpe,rp])*(Fsz-dFtz)
            Mz=(B.Nx*Pprop[Prt[B.P1].tYpe,rp])*(Fsy-dFty)-(B.Ny*Pprop[Prt[B.P1].tYpe,rp])*(Fsx-dFtx)
            
            cuda.atomic.add(Mx_All,B.P1,Mx-dMrx)
            cuda.atomic.add(My_All,B.P1,My-dMry)
            cuda.atomic.add(Mz_All,B.P1,Mz-dMrz)

            cuda.atomic.add(Mx_All,B.P2,Mx-dMrx)
            cuda.atomic.add(My_All,B.P2,My-dMry)
            cuda.atomic.add(Mz_All,B.P2,Mz-dMrz)
            

            Bnds[index]=B
               
@cuda.jit
def SampleBnds(Bnds,C,Prt):
    BProp=cuda.const.array_like(Bprp)
    A=0
    Kn=3
    Kt=1e9
    index=cuda.grid(1)
    if index<len(Bnds):
        B=Bnds[index]
        if B.tYpe==12:
#            Fnx=BProp[B.tYpe,Kn]*BProp[B.tYpe,A]*B.Unx
            Fsx=Kt*BProp[B.tYpe,A]*B.Usx*B.D
            cuda.atomic.add(C,0,abs(Fsx))
        if B.tYpe==12:
            P1p=Prt[B.P1]
            P2p=Prt[B.P2]
#            l=math.sqrt((P1p.x-P2p.x)*(P1p.x-P2p.x)+(P1p.y-P2p.y)*(P1p.y-P2p.y)+(P1p.z-P2p.z)*(P1p.z-P2p.z))
#            cuda.atomic.add(C,1,B.Unx*B.Unx)
            
            
#            cuda.atomic.add(C,2,1)
            U=(B.Usx**2+B.Usy**2+B.Usz**2)**0.5
#                U=abs(B.Usx)
#                cuda.atomic.add(C,3,U)
            cuda.atomic.max(C,2,U)
            cuda.atomic.add(C,3,1)
            if P1p.x>30 or P2p.x>30:
                d=abs(B.X0-(P1p.x-P2p.x))
                cuda.atomic.max(C,1,d)
#        if B.tYpe==12:
            cuda.atomic.add(C,4,1-B.D)
#            cuda.atomic.add(C,5,1)
            if B.D==0:
                cuda.atomic.add(C,5,1)
                
                

@cuda.jit
def Reset(C):
    C[0]=0
    C[1]=0
    C[2]=0
    C[3]=0
    C[4]=0
    C[5]=0
#    C[6]=0
#    C[7]=0
    C[9]=0
    
            
@cuda.jit
def SamplePrt(CC,Prt,Fx,Fy):
    index=cuda.grid(1)
    if index<len(Prt):
        P=Prt[index]
        if P.tYpe==3:
#            cuda.atomic.add(CC,0,CC[1]*Fx[index]+CC[2]*Fy[index])
            cuda.atomic.add(CC,0,Fx[index])
            cuda.atomic.add(CC,3,P.x)

@cuda.jit
def ResetCC(CC):
    CC[0]=0
    CC[3]=0
        
        
