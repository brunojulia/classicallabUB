# -----------------------------------------------------------
# intsim.py
# Requires physystem.py and intsim.kv to run
#
# Developed by Arnau Jurado Romero as part of a 'Practicas en Empresa'
# internship at QuantumLabUB, supervised by Bruno Juliá Díaz and
# Montserrat Guilleumas Morell
# 
# For any questions feel free to email me at:
# arnau.jurado.romero@gmail.com
# -----------------------------------------------------------


"""
intsim app running with kivy using the intsim.kv file.

The numerical method used in this program is found in the physystem.py code. 
This code contains contains the UI and the logic necesary to run it, 
as well as the definition of some variables like the units and size of the box.

The functions are written roughly in the order of the kv file.


Here I will explain some parts of the code that appear repeteadly or that 
are too extensive to explain.

-----Drawing-----
The BoxLayout object plotbox is where the drawing of the particles takes 
place, all kivy widgets have a canvas where you can draw shapes 
(technically is all the same canvas but that is not important right now).
For drawing I use the Ellipse and Line shapes, see the kivy documentation 
for its arguments and details.

Because we are translating a physical system's coordinates to kivy 
coordinates we first need to find the appropiate scale factors. 
The plotbox (and our physical system) is square, if you take a look at 
the .kv file you will find that the plotbox the height as the size of the square, 
that's because most screens are more wide than tall so we use the 
maximum space possible. In the code below I have generalized
the drawing to any aspect ratios but the .kv file still needs to be changed sadly.

For finding the scale factor we will divide the size by the lenght of 
the physical box, that's what the -scale = b/self.L- line does, where b is the size
of the plotbox:

w = self.plotbox.size[0]
h = self.plotbox.size[1]
b = min(w,h)

So when drawing we only need the multiply the physical coordinates of 
our point by the scale factor. We also need to keep in mind that kivy coordinates
start at the bottom left of the screen so we need to substract w/2 or h/2 
(which are both b/2) to the coordinates. Moreover, the Ellipse shape
also is referenced the bottom left of the shape (because it is a children of Rectangle) 
so we also need to substract the radius. With this information
you should understand the line:

Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

Also remember that we are working in reduced units (see doc for details) 
so we need to multiply by the unit of lenght (self.R) to have the coordinates 
in the same units as the box size (in this case angstroms).

-----Animation and flow control-----
Upon start, the __init__ function is called which sets up some parameters 
and flags as well as initializing the plots.

previewtimer is also started. This timer runs constatly and calls the 
preview function and checks if the animation is running or paused 
(controlled by the running and paused flags repectively). If the animation 
is not running or paused then it draws 
the particles using the information from previewlist.

The computation process is split into two functions (playcompute and 
computation). playcompute checks if the computation is already done, 
using the ready flag; if it is then the timer "timer" (sorry for the bad name) 
starts which calls animation. 
If the computation is not done then the computation function is called. 
These processes also change the values of the flags running, paused and ready.

-----Saving and loading-----
The save and load buttons call their respective pop-ups, these popups contain 
two buttons (and a text input in the case of save). One
of the buttons calls the save or load functions, which perform the saving 
or loading action itself and then closes the pop-up by calling dismiss_popup.
The other button labeled "Cancel" simply closes the pop-up.
Right now if you try to save without having performed a computation the program will crash.

-----Future improvents to be done-----
Unless the Verlet algorythm is optimized (which I'm sure it can improve 
but I don't know by how much) then the computations are too long
to be performed 'live'. The interface should be streamlined to load demos
 because the computation process would be way to lenghty.

The interface was adapted from my other program, 2dsim.py, and I didn't 
have enough time to clean up left-overs from that interface (like the 
time inversion button and the single particle addition mode).

Lastly, I would like to remark that this program can be repurposed
to other physical systems simply by changing the interaction potential
(to a gravitational one for example). This would also mean that less
particles are required to showcase interesting phenomena and so 
the computation time will be shorter.


"""




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from physystem import *

#Kivy imports
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.graphics import Rectangle,Color,Ellipse,Line #Used to draw
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Used to do matplotlib plots
from kivy.clock import Clock
from kivy.uix.popup import Popup
#Other imports
import pickle #For saving
import os # For saving too
import time #To check computation time

#This two lines should set the icon of the application to an ub logo
#but only works rarely
from kivy.config import Config
Config.set('kivy','window_icon','ub.png')


#Definition of the save and load windows popups
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
    
    #Definition for the matplotlib figures of the histograms
    #For some reason the xlabel is ignored, should look into that.
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
        
    #These are for a different method of accumulation (see comments animation function)
    Vacu = np.array([])
    MBacu = np.zeros(100)
    acucounter = 0

    
    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.time = 0.
        #Here you can modify the time of computation and the step
        self.T = 540
        self.dt = 0.01

        #Initialization of the speed button
        self.speedindex = 3
        self.change_speed()

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.ready = False #Checks if computation is done
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)#Always, runs, shows previews
        self.previewlist = []
        self.progress = 0.
        
        #Initialization of histogram plots
        self.histbox.add_widget(self.histcanvas)
        self.acuhistbox.add_widget(self.acuhistcanvas)
        
        #Here you can modify the units of the simulation as well as the size of the box.
        self.V0 = 0.01 #eV
        self.R = 3.405 #A
        self.L = 200. #A
        self.M = 0.04 #kg/mol

    def update_pos(self,touch):
        """This function updates the position parameters
        when you click the screen"""

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
        """This function sets the theta angle of the
        single particle addition mode (which is not really used
        for this program) when clicking and dragging"""

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

        self.stop() #I stop the simultion to avoid crashes

        #Check in which mode the user is
        if(self.partmenu.current_tab.text == 'Single'):
            vx = self.vsslider.value * np.cos(self.thetasslider.value*(np.pi/180.))
            vy = self.vsslider.value * np.sin(self.thetasslider.value*(np.pi/180.))

            self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([self.x0slider.value,self.y0slider.value])/self.R,np.array([vx,vy]),2))
            
            self.previewlist.append('Single')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,vx,vy])
        elif(self.partmenu.current_tab.text == 'Random Lattice'):
            #Initialization process for the lattice, see documentation for the operations in this part

            n = int(self.nrslider.value)
            x,y = np.linspace(-self.L/2*0.8,self.L/2*0.8,n),np.linspace(-self.L/2*0.8,self.L/2*0.8,n)
            vmax = 10
            temp = 2.5
            
            temp = 3.
            theta = np.random.ranf(n**2)*2*np.pi
            vx,vy = 0.5*np.cos(theta),0.5*np.sin(theta)
            
            vcm = np.array([np.sum(vx),np.sum(vy)])/n**2
            kin = np.sum(vx**2+vy**2)/(n**2)
            
            vx = (vx-vcm[0])*np.sqrt(2*temp/kin)
            vy = (vy-vcm[1])*np.sqrt(2*temp/kin)
            k = 0
            for i in range(0,n):
                for j in range(0,n):
                    self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([x[i],y[j]])/self.R,np.array([vx[k],vy[k]]),2))
                
                    self.previewlist.append('Single')
                    self.previewlist.append([x[i],y[j],vx[k]*self.R,vy[k]*self.R])
                    k += 1

        #This block of code is present at different points in the program
        #It updates the ready flag and changes the icons for compute/play button and the status label.
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
        #Empties particle list
        self.stop()
        self.particles = np.array([])
        self.previewlist = []
        
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
    
    def playcompute(self):
        #If ready is false it starts the computation, if true starts the animation
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
        #Computation process
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
        
        #This also saves the temperatures and energies to files
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
        #This simply cicles the sl list with the speed multipliers, self.speed is later
        #used to speed up the animation
        sl = [1,2,5,10]
        if(self.speedindex == len(sl)-1):
            self.speedindex = 0
        else:
            self.speedindex += 1
        self.speed = sl[self.speedindex]
        self.speedbutton.text = str(self.speed)+'x'
    
    #Saving and loading processes and popups, the kivy documentation
    #has a good explanation on the usage of the filebrowser widget and the
    #process of creating popups in general.
    def save(self,path,name):
        #I put all the relevant data in a numpy array and save it with pickle
        #The order is important for the loading process.
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
        """This plotpopu show the energy plots on a popup when the giant 'Energy' button
        on the UI is pressed, this was originally and experiment and I ran out of time to 
        change it. It should be done like the histograms and embed the FigureCanvasKivyAgg in
        the UI directly"""
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
        """This function comes from the other program and is linked to the time inversion button.
        Right now the button won't do anything because I have emptied this function. If you delete this
        Function the program will crash if you press the buttons. If you ask why I haven't deleted
        the button is because it would mess up the aspect ratio of the icons"""
        pass
        
    
    def preview(self,interval):
        """Draws the previews of the particles when the animation is not running or before adding
        the preview of the lattice mode before adding is not programmed (mainly because it is a random process)"""
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
        """Draw all the particles for the animation"""

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        self.plotbox.canvas.clear()
        
        N = self.s.particles.size
        n = int(self.T/self.dt)
        i = int(self.time/self.dt)
        delta = 5./self.dt

        with self.plotbox.canvas:
            for j in range(0,N): 
                Color(1.0,0.0,0.0)
                Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
        
        self.time += interval*self.speed #Here is where speed accelerates animation
        self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.
        
        self.acucounter += 1
        
        if(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
            vs = np.linspace(0,self.s.V.max(),100)
            
            self.histax.clear()
            self.histax.set_xlabel('v')
            self.histax.set_xlim([0,self.s.V.max()])
            self.histax.set_ylim([0,np.ceil(self.s.MB.max())])
            
        
            self.histax.hist(self.s.V[i,:],bins=np.arange(0, self.s.V.max() + 1, 1),density=True)
            self.histax.plot(vs,self.s.MB[i,:],'r-')
            self.histcanvas.draw()
        
        
        if(self.plotmenu.current_tab.text == 'Acu' and self.time>40.): #Accumulated momentum histogram
            vs = np.linspace(0,self.s.V.max(),100)
            
            
            self.acuhistax.clear()
            self.acuhistax.set_xlabel('v')
            self.acuhistax.set_xlim([0,self.s.V.max()])
            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])
            
        
            self.acuhistax.hist(self.s.Vacu[int((i-int(n - (40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
            self.acuhistax.plot(vs,self.s.MBacu[int((i-int(n - (40./self.dt)))/delta)],'r-')
            self.acuhistcanvas.draw()
                

        """This block of code is for building the accumulated histograms as the animation progresses, this
        is extremely slow and will slowdown the program, I will leave it if you want to take a look at it."""


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
#            
#        if(self.acucounter >= int(1./self.dt)):
#            self.acucounter = 0 
#        
#        if(self.time >= self.T):
#            self.time = 0.
#            self.Vacu = np.array([])

    

            
class intsimApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    intsimApp().run()