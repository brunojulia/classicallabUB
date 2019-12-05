
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from physystem import *
from phi import *

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty,ListProperty,NumericProperty,StringProperty
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle,Color,Ellipse,Line
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.popup import Popup
import pickle
import os
import time
from kivy.config import Config
Config.set('kivy','window_icon','ub.png')


class savewindow(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)   
    
class loadwindow(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    
    

class main(BoxLayout):
    
    charge = 1.
    
    particles = np.array([])
    
    plot_texture = ObjectProperty()
    
    hist = Figure()   
    histax = hist.add_subplot(111)     
    histax.set_xlabel('v')
    histax.set_xlim([0,1])
    histax.set_ylim([0,25])
    histcanvas = FigureCanvasKivyAgg(hist)
    
    acuhist = Figure()   
    acuhistax = acuhist.add_subplot(111)     
    acuhistax.set_xlabel('v')
    acuhistax.set_xlim([0,1])
    acuhistax.set_ylim([0,25])
    acuhistcanvas = FigureCanvasKivyAgg(acuhist)
        
    Vacu = np.array([])
    MBacu = np.zeros(100)
    acucounter = 0

    
    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.time = 0.
        self.T = 120
        self.dt = 0.01
        self.speedindex = 3
        self.change_speed()
        self.running = False
        self.paused = False
        self.ready = False
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)
        self.previewlist = []
        self.progress = 0.
        
        self.histbox.add_widget(self.histcanvas)
        self.acuhistbox.add_widget(self.acuhistcanvas)
        
        self.V0 = 0.01 #eV
        self.R = 3.405 #A
        self.L = 250. #A
        self.M = 0.04 #kg/mol
                  
    def update_pos(self,touch):
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale

        if(self.menu.current_tab.text == 'Particles'):
            self.x0slider.value = x
            self.y0slider.value = y
            
    def update_angle(self,touch):
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale
        xdif = x-self.x0slider.value
        ydif = y-self.y0slider.value
        if(np.abs(xdif)<0.01):
            if(ydif>0):
                angle = np.pi/2.
            else:
                angle = -np.pi/2.
        elif(xdif > 0 and ydif > 0):
            angle = np.arctan((ydif)/(xdif))
        elif(xdif < 0 and ydif < 0):
            angle = np.arctan((ydif)/(xdif)) + np.pi
        elif(xdif < 0 and ydif > 0):
            angle = np.arctan((ydif)/(xdif)) + np.pi
        else:
            angle = np.arctan((ydif)/(xdif)) + 2*np.pi
        if(np.abs(x) < 100. and np.abs(y) < 100.):
            if(self.partmenu.current_tab.text == 'Single'):
                self.thetasslider.value = int(round(angle*(180/np.pi),0))


    def add_particle_list(self):
        self.stop()
        if(self.partmenu.current_tab.text == 'Single'):
            vx = self.vsslider.value * np.cos(self.thetasslider.value*(np.pi/180.))
            vy = self.vsslider.value * np.sin(self.thetasslider.value*(np.pi/180.))
            self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([self.x0slider.value,self.y0slider.value])/self.R,np.array([vx,vy]),2))
            
            self.previewlist.append('Single')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,vx,vy])
        elif(self.partmenu.current_tab.text == 'Random Lattice'):
            n = int(self.nrslider.value)
            x,y = np.linspace(-self.L/2*0.8,self.L/2*0.8,n),np.linspace(-self.L/2*0.8,self.L/2*0.8,n)
            vmax = 10
            temp = 2.5
#            vx,vy = (np.random.ranf(n**2)-0.5),(np.random.ranf(n**2)-0.5)
#            vx,vy = 0.8*np.ones(n**2)*np.meshgrid(np.sign(x),np.sign(x))[0].flatten() , 0.5*np.ones(n**2)*np.meshgrid(np.sign(y),np.sign(y))[0].flatten()
            
            temp = 3.
            theta = np.random.ranf(n**2)*2*np.pi
            vx,vy = 0.5*np.cos(theta),0.5*np.sin(theta)
            
            vcm = np.array([np.sum(vx),np.sum(vy)])/n**2
            kin = np.sum(vx**2+vy**2)/n**2
            
            vx = (vx-vcm[0])*np.sqrt(2*temp/kin)
            vy = (vy-vcm[1])*np.sqrt(2*temp/kin)
            k = 0
            for i in range(0,n):
                for j in range(0,n):
                    self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([x[i],y[j]])/self.R,np.array([vx[k],vy[k]]),2))
                
                    self.previewlist.append('Single')
                    self.previewlist.append([x[i],y[j],vx[k]*self.R,vy[k]*self.R])
                    k += 1
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
        
        V = np.sqrt(vx**2+vy**2)
        self.histax.set_xlabel('v')
        self.histax.set_xlim([0,V.max()])
        self.histax.set_ylim([0,1]) 
            
        self.histax.hist(V,bins=np.arange(0, V.max() + 1, 1),density=True)
        self.histcanvas.draw()
        
            
    def reset_particle_list(self):
        self.stop()
        self.particles = np.array([])
        self.previewlist = []
        
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
    
    def playcompute(self):
        if(self.ready==False):
            self.statuslabel.text = 'Computing...'
            Clock.schedule_once(self.computation)

        elif(self.ready==True):
            if(self.running==False):
                self.timer = Clock.schedule_interval(self.animate,0.04)
                self.running = True
                self.paused = False
            elif(self.running==True):
                pass
            
    def computation(self,*args):
        print('---Computation Start---')

        start = time.time()
        self.s = PhySystem(self.particles,[self.V0,self.R,self.L/self.R])
        self.s.solveverlet(self.T,self.dt)
        print('---Computation End---')
        print('Exec time = ',time.time() - start)
        self.ready = True
        self.pcbutton.background_normal = 'Icons/play.png'
        self.pcbutton.background_down = 'Icons/playb.png'
        self.statuslabel.text = 'Ready'
        
        np.savetxt('Kenergy.dat',self.s.K,fmt='%10.5f')
        np.savetxt('Uenergy.dat',self.s.U,fmt='%10.5f')
        np.savetxt('Tenergy.dat',self.s.K + self.s.U,fmt='%10.5f')
        np.savetxt('Temps.dat',self.s.T,fmt='%10.5f')
        
        
        
    def pause(self):
        if(self.running==True):
            self.paused = True
            self.timer.cancel()
            self.running = False
        else:
            pass
        
    def stop(self):
        self.pause()
        self.paused = False
        self.time = 0
        self.plotbox.canvas.clear()
        
    
        
    def change_speed(self):
        sl = [1,2,5,10]
        if(self.speedindex == len(sl)-1):
            self.speedindex = 0
        else:
            self.speedindex += 1
        self.speed = sl[self.speedindex]
        self.speedbutton.text = str(self.speed)+'x'
    
    def save(self,path,name,comp=False):
        
        savedata = np.array([self.s,self.T,self.dt,self.L,self.previewlist])
        with open(os.path.join(path,name+'.dat'),'wb') as file:
            pickle.dump(savedata,file)
        self.dismiss_popup()
    
    def savepopup(self):
        content = savewindow(save = self.save, cancel = self.dismiss_popup)
        self._popup = Popup(title='Save File', content = content, size_hint=(1,1))
        self._popup.open()
    
    def load(self,path,name,demo=False):
        self.stop()
        with open(os.path.join(path,name[0]),'rb') as file:
            savedata = pickle.load(file)
        
        self.s = savedata[0]
        self.T = savedata[1]
        self.dt = savedata[2]
        self.L = savedata[3]
        self.previewlist = savedata[4]
        
        
        self.ready = True
        self.pcbutton.background_normal = 'Icons/play.png'
        self.pcbutton.background_down = 'Icons/playb.png'
        self.statuslabel.text = 'Ready'
        print('Loaded simulation {} with computation'.format(name))
        if(demo==False):
            self.dismiss_popup()
    
    def loadpopup(self):
        content = loadwindow(load = self.load, cancel = self.dismiss_popup)
        self._popup = Popup(title='Load File', content = content, size_hint=(1,1))
        self._popup.open()
        
    def plotpopup(self):
        self.eplot = Figure()
        t = np.arange(self.dt,self.T+self.dt,self.dt)
        ax = self.eplot.add_subplot(111)
        
        ax.plot(t,self.s.K,'r-',label = 'Kinetic Energy')
        ax.plot(t,self.s.U,'b-',label = 'Potential Energy')
        ax.plot(t,self.s.K+self.s.U,'g-',label = 'Total Energy')
#        plt.plot(t,self.s.Kmean,'g-',label = 'Mean Kinetic Energy')
        ax.legend(loc=1)
        ax.set_xlabel('t')
        
        self.ecanvas = FigureCanvasKivyAgg(self.eplot)
        content = self.ecanvas
        self._popup = Popup(title ='Energy conservation',content = content, size_hint=(0.9,0.9))
        self._popup.open()
    
    def dismiss_popup(self):
        self._popup.dismiss()
        
                
            
            
            
            
    def timeinversion(self):
#        TO FIX
        if(self.ready==True):
            self.pause()
            t = self.time
            self.stop()
            reversedpart = []
            reversedconds = []
            reversedpreview = []
            
            for p in self.particles:
                reversedpart.append(Particle(self.massslider.value,self.charge,dt))
                reversedconds.append([p.trax(t),p.tray(t),-p.travx(t),-p.travy(t)])
                reversedpreview.append('Single')
                reversedpreview.append([p.trax(t),p.tray(t),-p.travx(t),-p.travy(t)])
                
            self.particles = reversedpart
            self.init_conds = reversedconds
            self.previewlist = reversedpreview
            
            self.ready = False
#            self.pcbutton.text = "Compute"
            self.pcbutton.background_normal = 'Icons/compute.png'
            self.pcbutton.background_down = 'Icons/computeb.png'
            self.statuslabel.text = 'Not Ready'
        else:
            pass
        
    
    def preview(self,interval):
        if(self.running == False and self.paused == False):
            if(self.menu.current_tab.text == 'Particles'):
                if(self.partmenu.current_tab.text == 'Single'):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scale = b/self.L
                    self.plotbox.canvas.clear()
                    
                    vx = self.vsslider.value * np.cos(self.thetasslider.value*(np.pi/180.))
                    vy = self.vsslider.value * np.sin(self.thetasslider.value*(np.pi/180.))
                    
                    with self.plotbox.canvas:
                        Color(1.0,0.5,0.0)
                        Ellipse(pos=(self.x0slider.value*scale+w/2.-self.R*scale/2.,self.y0slider.value*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                        Line(points=[self.x0slider.value*scale+w/2.,self.y0slider.value*scale+h/2.,vx*scale+w/2.+self.x0slider.value*scale,vy*scale+w/2.+self.y0slider.value*scale])
                else:
                    self.plotbox.canvas.clear()
                    
                    
            else:
                self.plotbox.canvas.clear()
                 
                
            with self.plotbox.canvas:
                for i in range(0,len(self.previewlist),2):
                    if(self.previewlist[i] == 'Single'):
                        x0 = self.previewlist[i+1][0]
                        y0 = self.previewlist[i+1][1]
                        vx0 = self.previewlist[i+1][2]
                        vy0 = self.previewlist[i+1][3]
                        
                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scale = b/self.L
                        
                        Color(0.0,0.0,1.0)
                        Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                        Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])
               
    def animate(self,interval):
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        self.plotbox.canvas.clear()
        
        N = self.s.particles.size
        i = int(self.time/self.dt)
        delta = 1./self.dt
        with self.plotbox.canvas:
            for j in range(0,N): 
                Color(1.0,0.0,0.0)
                Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
        
        self.time += interval*self.speed
        self.progressbar.value = (self.time/self.T)*100
        
        self.acucounter += 1
        
        if(self.plotmenu.current_tab.text == 'Momentum'):
            if(i%100 == 0 or i == 0 or True):
                vs = np.linspace(0,self.s.V.max(),100)
                
                self.histax.clear()
                self.histax.set_xlabel('v')
                self.histax.set_xlim([0,self.s.V.max()])
                self.histax.set_ylim([0,np.ceil(self.s.MB.max())])
                
            
                self.histax.hist(self.s.V[i,:],bins=np.arange(0, self.s.V.max() + 1, 1),density=True)
                self.histax.plot(vs,self.s.MB[i,:],'r-')
                self.histcanvas.draw()
        
        
        if(self.plotmenu.current_tab.text == 'Acu' and self.time>self.T/2.):
            vs = np.linspace(0,self.s.V.max(),100)
            
            
            self.acuhistax.clear()
            self.acuhistax.set_xlabel('v')
            self.acuhistax.set_xlim([0,self.s.V.max()])
            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])
            
        
            self.acuhistax.hist(self.s.Vacu[int((i-int((self.T/self.dt)/2))/delta)],bins=np.arange(0, self.s.V.max() + 0.5, 0.5),density=True)
            self.acuhistax.plot(vs,self.s.MBacu[int((i-int((self.T/self.dt)/2))/delta)],'r-')
            self.acuhistcanvas.draw()
            
#        if(self.plotmenu.current_tab.text == 'Acu'):
#            vs = np.linspace(0,self.s.V.max(),100)
#            
#            
#            self.acuhistax.clear()
#            self.acuhistax.set_xlabel('v')
#            self.acuhistax.set_xlim([0,self.s.V.max()])
#            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])
#            
#        
#            self.acuhistax.hist(self.Vacu,bins=np.arange(0, self.s.V.max() + 0.5, 0.5),density=True)
#            self.acuhistax.plot(vs,self.MBacu,'r-')
#            self.acuhistcanvas.draw()
            
#        if(self.time>self.T/2. and self.acucounter == int(0.4/self.dt)):
#            print('hola')
#            self.Vacu = np.append(self.Vacu,self.s.V[i,:])
#            Temp = np.sum(self.Vacu**2)/(self.Vacu.size - 2)
#            self.MBacu = (vs/(Temp)*np.exp(-vs**2/(2*Temp)))    
#            print(self.Vacu.shape)
            
        if(self.acucounter >= int(1./self.dt)):
            self.acucounter = 0 
        
        if(self.time >= self.T):
            self.time = 0.
            self.Vacu = np.array([])

    

            
class intsimApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    intsimApp().run()