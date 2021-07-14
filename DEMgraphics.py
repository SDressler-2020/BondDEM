# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 21:06:19 2018

@author: jodie
"""

import numpy as np
import math
from numba import cuda, int16, float32, from_dtype, jit
import numba as nb
import csv
from tkinter import *

class Ball:
    def __init__(self, size, x, y,numberID):
        self.x=x
        self.y=y
        self.size = size
        self.numberID=numberID
        self.shape = canvas.create_oval(x, y, x+self.size, y+self.size, fill='red')
        self.speedx = 0
        self.speedy = 0
    def update(self,mag,maxvari,minvari,V,X):
        vari=np.linalg.norm(V[self.numberID])
        minvari=0
        xx=0
        if maxvari>1e-20:
            xx=((maxvari-vari)/(maxvari-minvari))
        xx=int((1-xx)*255)
        r = xx
        g = 100 
        canvas.itemconfig(self.shape,fill=rgb(r,g,0))
        self.speedx=(15+mag*X[self.numberID][0]-self.x)/0.001#this locates the ball, division by 10 to scale the velocity division by 900 (the zoom factor) would be as is, division by 10 is 90x exaggeration
        self.speedy=(450+mag*X[self.numberID][1]-self.y)/0.001
        canvas.move(self.shape, self.speedx, self.speedy)
        self.x=15+mag*X[self.numberID][0]
        self.y=450+mag*X[self.numberID][1]

class Lines:
    def __init__(self, mag,starts, end, numberID, P1, P2,P1size,P2size):
        self.starts=starts
        self.end=end
        self.cstarts=[0,0]
        self.cend=[0,0]
        self.cstarts[0]=15+mag*starts[0]+mag*P1size
        self.cstarts[1]=450+mag*starts[1]+mag*P1size
        self.cend[0]=15+mag*end[0]+mag*P2size
        self.cend[1]=450+mag*end[1]+mag*P2size
        self.numberID=numberID
        self.P1=P1
        self.P2=P2
        self.shape=canvas.create_line(self.cstarts[0],self.cstarts[1],self.cend[0],self.cend[1],width=1)
    def update(self,mag,minvari,maxvari,X,Fn,N):
#        vari=np.linalg.norm(Fn[self.numberID])
#        xx=0
#        d=np.dot(Fn[self.numberID],N[self.numberID].T)
        color="blue"
#        if d>0:
#            color="red"
#            xx=((maxvari-vari)/maxvari)
#        if d<0:
#            xx=((minvari-vari)/minvari)
#        xx=abs(int((1-xx)*10))
        canvas.itemconfig(self.shape,width=0.5,fill=color)
        self.cstarts[0]=self.cstarts[0]+(mag*X[self.P1][0]-mag*self.starts[0])/0.001
        self.cstarts[1]=self.cstarts[1]+(mag*X[self.P1][1]-mag*self.starts[1])/0.001
        self.cend[0]=self.cend[0]+(mag*X[self.P2][0]-mag*self.end[0])/0.001
        self.cend[1]=self.cend[1]+(mag*X[self.P2][1]-mag*self.end[1])/0.001
        canvas.coords(self.shape,self.cstarts[0],self.cstarts[1],self.cend[0],self.cend[1])
        self.starts[0]=X[self.P1][0]#this locates the ball, division by 10 to scale the velocity division by 900 (the zoom factor) would be as is, division by 10 is 90x exaggeration
        self.starts[1]=X[self.P1][1]
        self.end[0]=X[self.P2][0]#this locates the ball, division by 10 to scale the velocity division by 900 (the zoom factor) would be as is, division by 10 is 90x exaggeration
        self.end[1]=X[self.P2][1]
        


#colours
def rgb(r, g, b):
    return "#%s%s%s" % tuple([hex(c)[2:].rjust(2, "0") for c in (r, g, b)])

#def objgen(Particles,Pprop):
#    mag=15000
#    ball_list = []
#    i=Particles.shape[0]
#    for j in range(0,i):
#        if Particles[j]['z']==0:
#            ball_list.append(Ball(mag*2*Pprop[Particles[j]['tYpe']]['r'],15+mag*Particles[j]['x'],450+mag*Particles[j]['y'],j))
#    return ball_list

#mass=0
#Ip=1
##rp=2
WIDTH = 2400
HEIGHT = 900
tk = Tk()
canvas = Canvas(tk, width=WIDTH, height=HEIGHT, bg="black")
tk.title("Drawing")
canvas.pack()


def objgen(Particles,Pprop):
    mag=40
    ball_list = []
    i=Particles.shape[0]
    for j in range(0,i):
        if abs(Particles[j]['z'])<0.25:
            ball_list.append(Ball(mag*2*Pprop[Particles[j]['tYpe'],2],15+mag*Particles[j]['x'],450+mag*Particles[j]['y'],j))
    return ball_list

#def Linegen(Bonds,Particles,Pprop):
#    Line_list = []
#    mag=15000
#    i=Bonds.shape[0]
#    for j in range(0,i):
#        if Particles[Bonds[j]['P1']]['z']==0 and Particles[Bonds[j]['P2']]['z']==0:
#            Line_list.append(Lines(mag,[Particles[Bonds[j]['P1']]['x'],Particles[Bonds[j]['P1']]['y']],[Particles[Bonds[j]['P2']]['x'],Particles[Bonds[j]['P2']]['y']],j,Bonds[j]['P1'],Bonds[j]['P2'],Pprop[Particles[Bonds[j]['P1']]['tYpe']]['r'],Pprop[Particles[Bonds[j]['P2']]['tYpe']]['r']))
#    return Line_list

def Linegen(Bonds,Particles,Pprop):
    Line_list = []
    mag=40
    i=Bonds.shape[0]
    for j in range(0,i):
        if abs(Particles[Bonds[j]['P1']]['z'])<0.5 and abs(Particles[Bonds[j]['P2']]['z'])<0.5:
            Line_list.append(Lines(mag,[Particles[Bonds[j]['P1']]['x'],Particles[Bonds[j]['P1']]['y']],[Particles[Bonds[j]['P2']]['x'],Particles[Bonds[j]['P2']]['y']],j,Bonds[j]['P1'],Bonds[j]['P2'],Pprop[Particles[Bonds[j]['P1']]['tYpe'],2],Pprop[Particles[Bonds[j]['P2']]['tYpe'],2]))
    return Line_list


def updateGraphics(Prt,ball_list):
    minvari=0
    maxvari=0
    mag=40
    X=[]
    V=[]
    for x in Prt:
        X.append([x['x'],x['y'],x['z']])
        V.append([x['Vx'],x['Vy'],x['Vz']])
        if np.linalg.norm(np.array([x['Vx'],x['Vy'],x['Vz']]))>maxvari: #find maximum velocity, there is probably a more elegant way to do this
            maxvari=np.linalg.norm(np.array([x['Vx'],x['Vy'],x['Vz']]))
    V=np.array(V)
    X=np.array(X)
    for ball in ball_list:
        ball.update(mag,maxvari,minvari,V,X) #update graphics
    tk.update()
    
def updateLineGraphics(Bnd,Prt,Line_list):
    minvari=0
    maxvari=0
    mag=40
    X=[]
    Fn=[]
    N=[]
    for x in Prt:
        X.append([x['x'],x['y'],x['z']])
    for b in Bnd:
        Fn.append([1,1,1])
#        Fn.append([b['Fnx'],b['Fny'],b['Fnz']])
        N.append([b['Nx'],b['Ny'],b['Nz']])
#        d=np.dot(np.array([b['Fnx'],b['Fny'],b['Fnz']]),np.array([b['Nx'],b['Ny'],b['Nz']]))
        d=1
        if d>maxvari:
            maxvari=d
        if d<minvari:
            minvari=d

    X=np.array(X)
    Fn=np.array(Fn)
    N=np.array(N)
    for line in Line_list:
        line.update(mag,minvari,maxvari,X,Fn,N) #update graphics
    tk.update()