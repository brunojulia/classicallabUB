# -----------------------------------------------------------
# particlesinabox.py
# Requires physystem.py and particlesinabox.kv to run
# -----------------------------------------------------------

"""
Edited and modified by Jofre Vallès Muns, March 2020
from the initial code by Arnau Jurado Romero.

The comments without specifying nothing are from Arnau, and the comments
like "JV: ..." are made by Jofre.
"""

"""
particlesinabox app running with kivy using the particlesinabox.kv file.

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

#This two lines set the icon of the application to an ub logo
#JV: We then "config" the default window size
#JV: We have to put these lines first in the code in order to work, why? "Kivy things my friend"
from kivy.config import Config
Config.set('kivy','window_icon','Icons/ub.png')
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '700')

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
from kivy.graphics import Rectangle,Color,Ellipse,Line,InstructionGroup #Used to draw, JV: InstructionGroup to save the drawing of the brownian trace in a group
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Used to do matplotlib plots
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.core.window import Window #JV: used to change the background color
#Other imports
import pickle #For saving
import os # For saving too
import time #To check computation time


#JV: Definition of the save, the load and the advanced settings windows popups
class savewindow(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class loadwindow(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class settingswindow(FloatLayout):
    change_settings = ObjectProperty(None)
    cancel = ObjectProperty(None)


class main(BoxLayout):

    charge = 1.

    particles = np.array([])

    plot_texture = ObjectProperty()

    #Definition for the matplotlib figures of the histograms

    #JV: Moment histogram
    hist = Figure()
    histax = hist.add_subplot(111, xlabel='v', ylabel = 'Number of particles relative')
    histax.set_xlim([0,1])
    histax.set_ylim([0,25])
    hist.subplots_adjust(0.125,0.19,0.9,0.9) #JV: We ajust the subplot to see whole axis and their labels
    histax.yaxis.labelpad = 10 #JV: We ajust the labels to our interest
    histax.xaxis.labelpad = -0.5
    histcanvas = FigureCanvasKivyAgg(hist)

    #JV: Acomulated moment histogram
    acuhist = Figure()
    acuhistax = acuhist.add_subplot(111, xlabel='v', ylabel = 'Number of particles relative')
    acuhistax.set_xlim([0,1])
    acuhistax.set_ylim([0,25])
    acuhist.subplots_adjust(0.125,0.19,0.9,0.9)
    acuhistax.yaxis.labelpad = 10
    acuhistax.xaxis.labelpad = -0.5
    acuhistcanvas = FigureCanvasKivyAgg(acuhist)

    #JV: Energy (Sub)Plot
    enplot = Figure()
    enplotax = enplot.add_subplot(111, xlabel='t', ylabel = 'Energy')
    enplotax.set_xlim([0,60]) #JV: This initial value should change if we change the total time of computation
    enplotax.set_ylim([0,25])
    enplot.subplots_adjust(0.125,0.19,0.9,0.9)
    enplotax.yaxis.labelpad = 10
    enplotax.xaxis.labelpad = -0.5
    enplotcanvas = FigureCanvasKivyAgg(enplot)

    #JV: Xpos plot
    extraplot = Figure()
    extraplotax = extraplot.add_subplot(111, xlabel='t', ylabel= "Entropy")
    extraplotax.set_xlim([0,60]) #JV: This initial value should change if we change the total time of computation
    extraplotax.set_ylim([0,25])
    extraplot.subplots_adjust(0.125,0.19,0.9,0.9)
    extraplotax.yaxis.labelpad = 10
    extraplotax.xaxis.labelpad = -0.5
    extraplotcanvas = FigureCanvasKivyAgg(extraplot)

    #These are for a different method of accumulation (see comments animation function)
    Vacu = np.array([])
    MBacu = np.zeros(100)
    acucounter = 0

    #JV: To change the background color from the simulation canvas, check also the kivy file for more
    Window.clearcolor = (0, 0, 0, 1)

    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.time = 0.
        #Here you can modify the time of computation and the step
        self.T = self.timeslider.value
        self.dt = 0.01

        #Initialization of the speed button
        self.speedindex = 3
        self.change_speed()

        #JV: This variable saves the number of steps it draws the plots. If it's 1, it will update each step of time, etc. Smaller numbers makes less "fps" in the simulation
        self.update_plots = 5

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.ready = False #Checks if computation is done
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)#Always, runs, shows previews. Crida la funció preview cada 0.04 segons
        self.previewlist = []
        self.progress = 0.
        self.our_submenu = 'Random Lattice' #JV: We start at the "Random Lattice" submenu
        self.our_menu = 'In a box' #JV: We start at "In a box" menu. This two variables will store in which submenu/menu we are
        self.n = 1 #JV: Modify this value if at the start you want to show more than one particle in the simulation (as a default value)
        self.n1 = 1 #JV: Modify this value if at the start you want to show more than one particle in the subsystem 1 in the subsystems submenu (as a default value)
        self.n2 = 1 #JV: Same for the second subsystem
        self.nbig = 1 #JV: Number of initial big particles in the "Brownian" submenu
        self.nsmall = 4 #JV: Number of inicial small particles in the "Brownian" submenu

        self.Rbig = 4*3.405 #JV: We define the initial value of the radius for the big particle in the "Brownian" submenu

        self.wallpos = 0 #JV: We define the initial value of the position of the wall in the "Walls" menu
        self.holesize = 20 #JV: We define the initial value of the size of the hole (we need to change the kivy file if we change this value, go to the parameters of this slider and change the "value")
        self.wallwidth = 2 #JV: We define the initial value of the width of the wall

        #Initialization of the plots
        self.histbox.add_widget(self.histcanvas)
        self.acuhistbox.add_widget(self.acuhistcanvas)
        self.enplotbox.add_widget(self.enplotcanvas)
        self.extraplotbox.add_widget(self.extraplotcanvas)

        self.mass = 1 #JV: change this if you want to change the initial mass

        #Here you can modify the units of the simulation as well as the size of the box.
        self.V0 = 0.01 #eV
        self.R = 3.405 #A
        self.L = 200. #A
        self.M = 0.04 #kg/mol

        #JV: This group will contain the two lines that form the wall in the "Walls" menu
        self.obj = InstructionGroup()
        self.point1 = []
        self.point2 = []

        self.obj2 = InstructionGroup() #JV: This group will contain the trace of the big particle in the "Brownian" submenu

        #JV: We create a list that contains our submenus, it will help us when we have to modify them. Update this if you add more submenus.
        self.submenu_list = [self.rlmenu,self.sbsmenu,self.brwmenu]
        self.submenu_list_str = ["Random Lattice","Subsystems","Brownian"] #JV: And a list with the names of each submenu

        #JV: We do the same for the menus. Update this if you add more menus.
        self.menu_list = [self.inaboxmenu,self.freemenu,self.wallmenu]
        self.menu_list_str = ["In a box","Free!","Walls"] #JV: And a list with the names of each menu

        #JV: We make this so the simulation starts with one particle instead of 0, that could lead to some errors
        self.add_particle_list(False)

    def update_pos(self,touch):
        """This function updates the position parameters
        when you click the screen"""

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale


    def on_touch_Slider(self):
        """JV: This function is evaluated whenever Slider is clicked, we will want to add a particle
        if the number chosen particles is changed, if it isn't, we will understand it as an accidental
        click (which often happens), so we don't mess the computation"""

        #JV: Update the computation time
        if(self.timeslider.value == self.T):
            pass
        else:
            self.T = self.timeslider.value
            self.add_particle_list(False)

        #JV: Conditions for "In a box" menu
        if(self.our_menu == self.menu_list_str[0]):
            if(self.our_submenu == 'Random Lattice'):
                if(self.nrslider.value == self.n):
                    pass
                else:
                    self.n = self.nrslider.value
                    self.add_particle_list(False)
            elif(self.our_submenu == 'Subsystems'):
                if(self.n1slider.value == self.n1 and self.n2slider.value == self.n2):
                      pass
                else:
                      self.n1 = self.n1slider.value
                      self.n2 = self.n2slider.value
                      self.add_particle_list(False)
            elif(self.our_submenu == 'Brownian'):
                if(self.nbigslider.value == self.nbig and self.nsmallslider.value == self.nsmall):
                    pass
                else:
                    self.nbig = self.nbigslider.value
                    self.nsmall = self.nsmallslider.value
                    self.add_particle_list(False)
        #JV: Conditions for "Free!" menu
        elif(self.our_menu == self.menu_list_str[1]):
            if(self.our_submenu== 'Random Lattice'):
                if(self.nrslider2.value == self.n):
                    pass
                else:
                    self.n = self.nrslider2.value
                    self.add_particle_list(False)
            elif(self.our_submenu == 'Subsystems'):
                if(self.n1slider2.value == self.n1 and self.n2slider2.value == self.n2):
                      pass
                else:
                      self.n1 = self.n1slider2.value
                      self.n2 = self.n2slider2.value
                      self.add_particle_list(False)
            elif(self.our_submenu == 'Brownian'):
                if(self.nbigslider2.value == self.nbig and self.nsmallslider2.value == self.nsmall):
                    pass
                else:
                    self.nbig = self.nbigslider2.value
                    self.nsmall = self.nsmallslider2.value
                    self.add_particle_list(False)
        #JV: Conditions for "Walls" menu
        elif(self.our_menu == self.menu_list_str[2]):
            if(self.our_submenu == 'Random Lattice'):
                if(self.nrslider3.value == self.n):
                    pass
                else:
                    self.n = self.nrslider3.value
                    self.add_particle_list(False)
            elif(self.our_submenu == 'Subsystems'):
                if(self.n1slider3.value == self.n1 and self.n2slider3.value == self.n2):
                      pass
                else:
                      self.n1 = self.n1slider3.value
                      self.n2 = self.n2slider3.value
                      self.add_particle_list(False)
            elif(self.our_submenu == 'Brownian'):
                if(self.nbigslider3.value == self.nbig and self.nsmallslider3.value == self.nsmall):
                    pass
                else:
                    self.nbig = self.nbigslider3.value
                    self.nsmall = self.nsmallslider3.value
                    self.add_particle_list(False)

            if(self.wallslider.value == self.wallpos):
                pass
            else:
                self.wallpos = self.wallslider.value
                self.add_particle_list(False)

            if(self.holeslider.value == self.holesize):
                pass
            else:
                self.holesize = self.holeslider.value
                self.add_particle_list(False)

    def on_touch_Submenu(self):
        """JV: Similar to the previous function, this function is evaluated when the submenu buttons
        are clicked, we will want to erase the previous particles that are in the screen and change them
        to the ones that we want for the new submenu"""

        #JV: Conditions for each menu
        for i in range (len(self.menu_list)):
            #JV: 0 corresponds to "In a box", 1 to "Free!", 2 to "Walls"
            if(self.menu.current_tab.text == self.menu_list_str[i]):
                if(self.menu_list[i].current_tab.text  == 'Random Lattice' and not(self.our_submenu == 'Random Lattice')):
                    self.stop()
                    self.our_submenu = 'Random Lattice'
                    self.extraplottab.text = "Entropy"
                    self.extraplotax.clear()
                    self.extraplotax.set_xlabel('t')
                    self.extraplotax.set_ylabel('Entropy')
                    self.extraplotcanvas.draw()
                    self.add_particle_list(False)
                elif(self.menu_list[i].current_tab.text  == 'Subsystems' and not(self.our_submenu == 'Subsystems')):
                    self.stop()
                    self.our_submenu = 'Subsystems'
                    self.extraplottab.text = "Entropy"
                    self.extraplotax.clear()
                    self.extraplotax.set_xlabel('t')
                    self.extraplotax.set_ylabel('Entropy')
                    self.extraplotcanvas.draw()
                    self.add_particle_list(False)
                elif(self.menu_list[i].current_tab.text  == 'Brownian' and not(self.our_submenu == 'Brownian')):
                    self.stop()
                    self.our_submenu = 'Brownian'
                    self.extraplotax.clear()
                    self.extraplotax.set_xlabel('t')
                    self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$')
                    self.extraplotcanvas.draw()
                    self.extraplottab.text = "<|x(t)-x(0)|^2>"
                    self.add_particle_list(False)
                else:
                    pass

    def on_touch_Menu(self):
        """JV: Similar to the previous functions, this function is evaluated when the menu buttons
        are clicked, we will want to erase the previous particles that are in the screen and reset them to
        the ones we want in the new menu"""

        if(self.menu.current_tab.text == 'In a box' and not(self.our_menu == 'In a box')):
            self.stop()
            self.our_menu = 'In a box'
            self.add_particle_list(False)
        elif(self.menu.current_tab.text == 'Free!' and not(self.our_menu == 'Free!')):
            self.stop()
            self.our_menu = 'Free!'
            self.add_particle_list(False)
        elif(self.menu.current_tab.text == 'Walls' and not(self.our_menu == 'Walls')):
            self.stop()
            self.our_menu = 'Walls'
            self.add_particle_list(False)
        else:
            pass

        #JV: Now we update the submenu so we go to the submenu that we were previously, not the same that we were but in a different menu
        self.on_touch_Submenu()

    def add_particle_list(self,inversion):

        self.stop() #I stop the simultion to avoid crashes

        self.reset_particle_list();

        #JV: We check the part of the submenu that we are
        if(self.our_submenu == 'Random Lattice'):
            if(self.our_menu == "In a box"):
                self.n = int(self.nrslider.value)
            elif(self.our_menu == "Free!"):
                self.n = int(self.nrslider2.value)
            elif(self.our_menu == "Walls"):
                self.n = int(self.nrslider3.value)

            if(inversion == False):
                self.inversion = False
                if(self.our_menu == "Walls"):
                    x,y = np.linspace(-self.L/2*0.9,self.wallpos-self.L*0.05,self.n),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n)
                else:
                    x,y = np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n)

                temp = 3.
                theta = np.random.ranf(self.n**2)*2*np.pi
                vx,vy = 0.5*np.cos(theta),0.5*np.sin(theta)

                vcm = np.array([np.sum(vx),np.sum(vy)])/self.n**2
                kin = np.sum(vx**2+vy**2)/(self.n**2)

                if(self.n == 1): #JV: To avoid problems, if we only have one particle it will not obey that the velocity of the center of mass is 0
                    vx = vx*np.sqrt(2*temp/kin)
                    vy = vy*np.sqrt(2*temp/kin)
                else:
                    vx = (vx-vcm[0])*np.sqrt(2*temp/kin)
                    vy = (vy-vcm[1])*np.sqrt(2*temp/kin)

            else:
                self.inversion = True
                i = int(self.T/self.dt)

                x = self.s.X[i,:]*self.R
                y = self.s.Y[i,:]*self.R
                vx = -self.s.VX[i,:]
                vy = -self.s.VY[i,:]

            k = 0
            for i in range(0,self.n):
                for j in range(0,self.n):
                    #JV: In "particles" we have the positions and velocities in reduced units (the velocities are already transformed,
                    # but for the positions we need to include the scale factor)
                    if(inversion == False):
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x[i],y[j]])/self.R,np.array([vx[k],vy[k]]),2))

                        #JV: In this new array we will have the positions and velocities in the physical units (Angstrom,...)
                        self.previewlist.append([x[i],y[j],vx[k]*self.R,vy[k]*self.R])
                    else:
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x[k],y[k]])/self.R,np.array([vx[k],vy[k]]),2))
                        self.previewlist.append([x[k],y[k],vx[k]*self.R,vy[k]*self.R])

                    k += 1

        elif(self.our_submenu == 'Subsystems'):
            if(self.our_menu == "In a box"):
                self.n1 = int(self.n1slider.value)
                self.n2 = int(self.n2slider.value)
            elif(self.our_menu == "Free!"):
                self.n1 = int(self.n1slider2.value)
                self.n2 = int(self.n2slider2.value)

            if(inversion == False):
                self.inversion = False
                if(self.our_menu == "Walls"):
                    x1,y1 = np.linspace(-self.L/2*0.9,self.wallpos-self.L*0.05,self.n1),np.linspace(self.L/2*0.9,self.L/2*0.1,self.n1)
                    x2,y2 = np.linspace(-self.L/2*0.9,self.wallpos-self.L*0.05,self.n2),np.linspace(-self.L/2*0.9,-self.L/2*0.1,self.n2)
                else:
                    x1,y1 = np.linspace(-self.L/2*0.9,-self.L/2*0.1,self.n1),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n1)
                    x2,y2 = np.linspace(self.L/2*0.1,self.L/2*0.9,self.n2),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n2)

                temp1 = 1
                theta1 = np.random.ranf(self.n1**2)*2*np.pi
                vx1,vy1 = 0.5*np.cos(theta1),0.5*np.sin(theta1)
                vcm1 = np.array([np.sum(vx1),np.sum(vy1)])/self.n1**2
                kin1 = np.sum(vx1**2+vy1**2)/(self.n1**2)

                if(self.n1 == 1): #JV: As previous, if there's only one particle his cm velocity will not be 0
                    vx1 = vx1*np.sqrt(2*temp1/kin1)
                    vy1 = vy1*np.sqrt(2*temp1/kin1)
                else:
                    vx1 = (vx1-vcm1[0])*np.sqrt(2*temp1/kin1)
                    vy1 = (vy1-vcm1[1])*np.sqrt(2*temp1/kin1)

                temp2 = 3
                theta2 = np.random.ranf(self.n2**2)*2*np.pi
                vx2,vy2 = 0.5*np.cos(theta2),0.5*np.sin(theta2)
                vcm2 = np.array([np.sum(vx2),np.sum(vy2)])/self.n2**2
                kin2 = np.sum(vx2**2+vy2**2)/(self.n2**2)

                if(self.n2 == 1):
                    vx2 = vx2*np.sqrt(2*temp2/kin2)
                    vy2 = vy2*np.sqrt(2*temp2/kin2)
                else:
                    vx2 = (vx2-vcm2[0])*np.sqrt(2*temp2/kin2)
                    vy2 = (vy2-vcm2[1])*np.sqrt(2*temp2/kin2)

            else:
                self.inversion = True
                i = int(self.T/self.dt)

                x = self.s.X[i,:]*self.R
                y = self.s.Y[i,:]*self.R
                vx = -self.s.VX[i,:]
                vy = -self.s.VY[i,:]

            k = 0
            for i in range(0,self.n1):
                for j in range(0,self.n1):
                    if(inversion == False):
                        #JV: In "particles" we have the positions and velocities in kivy units (the velocities are already transformed,
                        # but for the positions we need to include the scale factor)
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x1[i],y1[j]])/self.R,np.array([vx1[k],vy1[k]]),2))

                        #JV: In this new array we will have the positions and velocities in the physical units (Angstrom,...)
                        self.previewlist.append([x1[i],y1[j],vx1[k]*self.R,vy1[k]*self.R])
                    else:
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x[k],y[k]])/self.R,np.array([vx[k],vy[k]]),2))
                        self.previewlist.append([x[k],y[k],vx[k]*self.R,vy[k]*self.R])

                    k += 1

            k = 0
            for i in range(0,self.n2):
                for j in range(0,self.n2):
                    if(inversion == False):
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x2[i],y2[j]])/self.R,np.array([vx2[k],vy2[k]]),2))
                        self.previewlist.append([x2[i],y2[j],vx2[k]*self.R,vy2[k]*self.R])
                    else:
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x[k+self.n1**2],y[k+self.n1**2]])/self.R,np.array([vx[k+self.n1**2],vy[k+self.n1**2]]),2))
                        self.previewlist.append([x[k+self.n1**2],y[k+self.n1**2],vx[k+self.n1**2]*self.R,vy[k+self.n1**2]*self.R])

                    k += 1

        elif(self.our_submenu == 'Brownian'):
            if(self.our_menu == "In a box"):
                self.nbig = int(self.nbigslider.value)
                self.nsmall = int(self.nsmallslider.value)
            elif(self.our_menu == "Free!"):
                self.nbig = int(self.nbigslider2.value)
                self.nsmall = int(self.nsmallslider2.value)

            if(inversion == False):
                self.inversion = False
                #JV: corresponding to the small particles variables
                if(self.our_menu == "Walls"):
                    x,y = np.linspace(-self.L/2*0.9,self.wallpos-self.L*0.05,self.nsmall),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.nsmall)
                else:
                    x,y = np.linspace(-self.L/2*0.9,self.L/2*0.9,self.nsmall),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.nsmall)

                temp = 4.
                theta = np.random.ranf(self.nsmall**2)*2*np.pi
                vx,vy = 0.5*np.cos(theta),0.5*np.sin(theta)

                vcm = np.array([np.sum(vx),np.sum(vy)])/self.nsmall**2
                kin = np.sum(vx**2+vy**2)/(self.nsmall**2)

                vx = (vx-vcm[0])*np.sqrt(2*temp/kin)
                vy = (vy-vcm[1])*np.sqrt(2*temp/kin)

                #JV: now for the big particle(s):
                if(self.our_menu == "Walls"):
                    xbig,ybig = (-(self.L/2-self.wallpos)/2,0)
                else:
                    xbig,ybig = (0,0)

                vxbig,vybig = (0,0)

            else:
                self.inversion = True
                i = int(self.T/self.dt)

                x = self.s.X[i,:]*self.R
                y = self.s.Y[i,:]*self.R
                vx = -self.s.VX[i,:]
                vy = -self.s.VY[i,:]


            k = 0
            for i in range(0,self.nsmall):
                for j in range(0,self.nsmall):
                    if(inversion == False):
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x[i],y[j]])/self.R,np.array([vx[k],vy[k]]),2))
                        self.previewlist.append([x[i],y[j],vx[k]*self.R,vy[k]*self.R])
                    else:
                        self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([x[k],y[k]])/self.R,np.array([vx[k],vy[k]]),2))
                        self.previewlist.append([x[k],y[k],vx[k]*self.R,vy[k]*self.R])

                    k += 1

            k = 0
            for i in range(0, self.nbig):
                for j in range(0,self.nbig):
                    if(inversion == False):
                        #JV: Because we want the supose that all the particles have the same density, and we are on a 2D field, we include the (self.Rbig/self.R)**2 factor on the mass
                        self.particles = np.append(self.particles,particle(self.mass*((self.Rbig/self.R)**2),self.charge,self.Rbig/self.R,np.array([xbig,ybig])/self.R,np.array([vxbig,vybig]),2))
                        self.previewlist.append([xbig,ybig,vxbig*self.R,vybig*self.R])
                    else:
                        self.particles = np.append(self.particles,particle(self.mass*((self.Rbig/self.R)**2),self.charge,self.Rbig/self.R,np.array([x[k+self.nsmall**2],y[k+self.nsmall**2]])/self.R,np.array([vx[k+self.nsmall**2],vy[k+self.nsmall**2]]),2))
                        self.previewlist.append([x[k+self.nsmall**2],y[k+self.nsmall**2],vx[k+self.nsmall**2]*self.R,vy[k+self.nsmall**2]*self.R])

                    k += 1

        #This block of code is present at different points in the program
        #It updates the ready flag and changes the icons for compute/play button and the status label.
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'

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
                self.timer = Clock.schedule_interval(self.animate,0.05)
                self.running = True
                self.paused = False
            elif(self.running==True):
                pass

    def computation(self,*args):
        #Computation process
        print("")
        print("")
        print('---Computation Start---')

        start = time.time()

        #JV: We create a PhySystem class by passing the array of particles and the physical units of the simulation as arguments
        #JV: self.wallpos, self.holesize, self.wallwidth are in Angtroms, so we need to divide it by self.R to have it in reduced units (the units we work in physystem)
        self.s = PhySystem(self.particles,[self.V0,self.R,self.L/self.R,self.our_menu,self.our_submenu,self.n1,self.n2,self.wallpos/self.R,self.holesize/self.R,self.wallwidth/self.R])

        #JV: We put the +10 because we want to genarate more values than the ones we will show, we do that to be able to stop the simulation when it ends, to avoid problems
        #JV: It's not an elegant solution, but it works just fine. Check in the future, we could correct this maybe changing how the time steps work
        #JV: If we modify as we said in the previous line, change the +10 in the part of energy representation in the animate() function
        self.s.solveverlet(self.T+10,self.dt)

        print('---Computation End---')
        print('Exec time = ',time.time() - start)
        print("")

        last = len(self.s.U)-1 #JV: This variable stores the index of the last variable in this arrays, so we can access it easier
        print("Initial energy: ",self.s.K[0]+self.s.U[0],"Final energy: ",self.s.U[last]+self.s.K[last])
        print("Relative increment of energy: ", (abs((self.s.K[0]+self.s.U[0])-(self.s.U[last]+self.s.K[last]))/(self.s.K[0]+self.s.U[0]))*100,"%")

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
        self.obj2.clear()
        self.plotbox.canvas.clear()
        self.acucounter = 0

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
        savedata = np.array([self.s,self.T,self.dt,self.L,self.previewlist,self.our_menu,self.our_submenu,self.n1,self.n2,self.nsmall,self.wallpos,self.holesize,self.Rbig])
        with open(os.path.join(path,name+'.dat'),'wb') as file:
            pickle.dump(savedata,file)
        self.dismiss_popup()

    def savepopup(self):
        content = savewindow(save = self.save, cancel = self.dismiss_popup)
        self._popup = Popup(title='Save File', content = content, size_hint=(1,1))
        self._popup.open()

    def advanced_settings(self):
        content = settingswindow(change_settings = self.change_settings, cancel = self.dismiss_popup)
        self._popup = Popup(title='Advanced Settings', content = content, size_hint = (1,1))
        self._popup.content.rbig_slider.value = int(self.Rbig/self.R)
        self._popup.content.boxlength_slider.value = self.L
        self._popup.content.dt_slider.value = self.dt
        self._popup.open()

    def change_settings(self):
        if(self._popup.content.rbig_slider.value != self.Rbig or self._popup.content.boxlength_slider.value != self.L or self._popup.content.dt_slider.value != self.dt):
            self.Rbig = self._popup.content.rbig_slider.value * self.R
            self.L = self._popup.content.boxlength_slider.value
            self.dt = self._popup.content.dt_slider.value
            self.add_particle_list(False)
        else:
            pass

    def load(self,path,name,demo=False):
        self.stop()
        with open(os.path.join(path,name[0]),'rb') as file:
            savedata = pickle.load(file)

        self.s = savedata[0]
        self.T = savedata[1]
        self.dt = savedata[2]
        self.L = savedata[3]
        self.previewlist = savedata[4]
        self.our_menu = savedata[5] #JV: To know in which menu corresponds the simulation
        self.our_submenu = savedata[6]
        self.n1 = savedata[7]
        self.n2 = savedata[8]
        self.nsmall = savedata[9]
        self.wallpos = savedata[10]
        self.holesize = savedata[11]
        self.Rbig = savedata[12]

        self.timeslider.value = self.T
        self.n = int(np.sqrt(self.s.particles.size))

        #JV: And now we set the state of the corresponding menu to "pressed". Update this if you add more menus or submenus
        if(self.our_menu == "In a box"):
            self.menu.switch_to(self.inaboxtab)

        elif(self.our_menu == "Free!"):
            self.menu.switch_to(self.freetab)

        elif(self.our_menu == "Walls"):
            self.menu.switch_to(self.walltab)
            self.wallslider.value = self.wallpos #JV: Now we need to update also the specific sliders of this menu
            self.holeslider.value = self.holesize

          #JV: Now we do the same for the submenu buttons, and we update the sliders to the actual simulation parameter
        if(self.our_submenu == "Random Lattice"):
            self.extraplotax.clear()
            self.extraplotax.set_xlabel('t')
            self.extraplotax.set_ylabel('Entropy')
            self.extraplotcanvas.draw()
            self.extraplottab.text = "Entropy"

            if(self.our_menu == "In a box"):
                self.inaboxmenu.switch_to(self.rlmenu)
                self.nrslider.value = int(np.sqrt(self.s.particles.size))

            elif(self.our_menu == "Free!"):
                self.freemenu.switch_to(self.rlmenu2)
                self.nrslider2.value = int(np.sqrt(self.s.particles.size))

            elif(self.our_menu == "Walls"):
                self.wallmenu.switch_to(self.rlmenu3)
                self.nrslider3.value = int(np.sqrt(self.s.particles.size))

        elif(self.our_submenu == "Subsystems"):

            if(self.our_menu == "In a box"):
                self.inaboxmenu.switch_to(self.sbsmenu)
                self.n1slider.value = self.n1
                self.n2slider.value = self.n2

            elif(self.our_menu == "Free!"):
                self.freemenu.switch_to(self.sbsmenu2)
                self.n1slider2.value = self.n1
                self.n2slider2.value = self.n2

            elif(self.our_menu == "Walls"):
                self.wallmenu.switch_to(self.sbsmenu3)
                self.n1slider3.value = self.n1
                self.n2slider3.value = self.n2

        elif(self.our_submenu == "Brownian"):
            self.extraplotax.clear()
            self.extraplotax.set_xlabel('t')
            self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$')
            self.extraplotcanvas.draw()
            self.extraplottab.text = "<|x(t)-x(0)|^2>"

            if(self.our_menu == "In a box"):
                self.inaboxmenu.switch_to(self.brwmenu)
                self.nbigslider.value = int(np.sqrt(self.s.particles.size - self.nsmall**2))
                self.nsmallslider.value = self.nsmall

            elif(self.our_menu == "Free!"):
                self.freemenu.switch_to(self.brwmenu2)
                self.nbigslider2.value = int(np.sqrt(self.s.particles.size - self.nsmall**2))
                self.nsmallslider2.value = self.nsmall

            elif(self.our_menu == "Walls"):
                self.wallmenu.switch_to(self.brwmenu3)
                self.nbigslider3.value = int(np.sqrt(self.s.particles.size - self.nsmall**2))
                self.nsmallslider3.value = self.nsmall

        self.ready = True
        self.pcbutton.background_normal = 'Icons/play.png'
        self.pcbutton.background_down = 'Icons/playb.png'
        self.statuslabel.text = 'Ready'
        print("")
        print("")
        print('Loaded simulation {} with computation'.format(name))

        last = len(self.s.U)-1 #JV: This variable stores the index of the last variable in this arrays, so we can access it easier
        print("Initial energy: ",self.s.K[0]+self.s.U[0],"Final energy: ",self.s.U[last]+self.s.K[last])
        print("Relative increment of energy: ", (abs((self.s.K[0]+self.s.U[0])-(self.s.U[last]+self.s.K[last]))/(self.s.K[0]+self.s.U[0]))*100,"%")
        if(demo==False):
            self.dismiss_popup()

    def loadpopup(self):
        content = loadwindow(load = self.load, cancel = self.dismiss_popup)
        self._popup = Popup(title='Load File', content = content, size_hint=(1,1))
        self._popup.open()



#    def plotpopup(self):
#        """This plotpopu show the energy plots on a popup when the giant 'Energy' button
#        on the UI is pressed, this was originally and experiment and I ran out of time to
#        change it. It should be done like the histograms and embed the FigureCanvasKivyAgg in
#        the UI directly"""
#        self.eplot = Figure()
#        t = np.arange(self.dt,self.T+self.dt,self.dt)
#        ax = self.eplot.add_subplot(111)
#
#        ax.plot(t,self.s.K,'r-',label = 'Kinetic Energy')
#        ax.plot(t,self.s.U,'b-',label = 'Potential Energy')
#        ax.plot(t,self.s.K+self.s.U,'g-',label = 'Total Energy')
##        plt.plot(t,self.s.Kmean,'g-',label = 'Mean Kinetic Energy')
#        ax.legend(loc=1)
#        ax.set_xlabel('t')
#
#        self.ecanvas = FigureCanvasKivyAgg(self.eplot)
#        content = self.ecanvas
#        self._popup = Popup(title ='Energy conservation',content = content, size_hint=(0.9,0.9))
#        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()



    def timeinversion(self):
        """JV: This funcion is the one responsible of calling the others functions when we want to make
        the time inversions, allowing us to compute the path of the particles when inverting its velocity,
        so we can see if the particles reach the first initial state. We change the time of computation so
        it reaches to the initial positions and then it stops."""

        #JV: We have this condition so you are only able to compute this time inversion if you haven't done it before (in this simulation)
        # Doing this we make sure the user doesn't enter in a loop of time inversions that are not what we want ( :) )
        if(self.inversion == False):
            self.T = self.time
            self.stop()
            self.timeslider.value = self.T

            self.add_particle_list(True)
        pass


    def preview(self,interval):
        """Draws the previews of the particles when the animation is not running or before adding
        the preview of the lattice mode before adding is not programmed (mainly because it is a random process)"""

        if(self.running == False and self.paused == False):
            #JV: We can add conditions when there will be different menus, now it's not needed
            self.plotbox.canvas.clear()
            with self.plotbox.canvas:
                if(self.our_submenu == 'Random Lattice'):
                    for i in range(0,len(self.previewlist),1):
                        x0 = self.previewlist[i][0]
                        y0 = self.previewlist[i][1]
                        vx0 = self.previewlist[i][2]
                        vy0 = self.previewlist[i][3]

                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scale = b/self.L

                        Color(0.34,0.13,1.0)
                        Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                        Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                elif(self.our_submenu == 'Subsystems'):
                    for i in range(0,len(self.previewlist),1):
                        if (i < (self.n1)**2):
                            x0 = self.previewlist[i][0]
                            y0 = self.previewlist[i][1]
                            vx0 = self.previewlist[i][2]
                            vy0 = self.previewlist[i][3]

                            w = self.plotbox.size[0]
                            h = self.plotbox.size[1]
                            b = min(w,h)
                            scale = b/self.L

                            Color(0.34,0.13,1.0)
                            Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                            Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                        else:
                            x0 = self.previewlist[i][0]
                            y0 = self.previewlist[i][1]
                            vx0 = self.previewlist[i][2]
                            vy0 = self.previewlist[i][3]

                            w = self.plotbox.size[0]
                            h = self.plotbox.size[1]
                            b = min(w,h)
                            scale = b/self.L

                            Color(0.035,0.61,0.17)
                            Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                            Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                elif(self.our_submenu == 'Brownian'):
                    for i in range(0,len(self.previewlist),1):
                        if (i < (self.nsmall)**2):
                            x0 = self.previewlist[i][0]
                            y0 = self.previewlist[i][1]
                            vx0 = self.previewlist[i][2]
                            vy0 = self.previewlist[i][3]

                            w = self.plotbox.size[0]
                            h = self.plotbox.size[1]
                            b = min(w,h)
                            scale = b/self.L

                            Color(0.34,0.13,1.0)
                            Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                            Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                        else:
                            x0 = self.previewlist[i][0]
                            y0 = self.previewlist[i][1]
                            vx0 = self.previewlist[i][2]
                            vy0 = self.previewlist[i][3]

                            w = self.plotbox.size[0]
                            h = self.plotbox.size[1]
                            b = min(w,h)
                            scale = b/self.L

                            Color(0.035,0.61,0.17)
                            Ellipse(pos=(x0*scale+w/2.-self.Rbig*scale/2.,y0*scale+h/2.-self.Rbig*scale/2.),size=(self.Rbig*scale,self.Rbig*scale))
                            Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                if(self.our_menu == "Walls"):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scale = b/self.L

                    self.obj = InstructionGroup()
                    self.plotbox.canvas.remove(self.obj)

                    #JV: We need to multiply by scale to transform from Angstrom units to Kivy units, that ajust depending on the resulution
                    self.obj.add(Color(0.37,0.01,0.95))
                    self.obj.add(Rectangle(pos=(w/2+self.wallpos*scale-self.wallwidth*scale/2,0), size = (self.wallwidth*scale,h/2 - self.holesize*scale/2)))
                    self.obj.add(Rectangle(pos=(w/2+self.wallpos*scale-self.wallwidth*scale/2,h/2+self.holesize*scale/2), size = (self.wallwidth*scale,h/2 - self.holesize*scale/2)))
                    self.plotbox.canvas.add(self.obj)


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

        if(self.our_menu == "Walls"):
            self.plotbox.canvas.add(self.obj)

        if(self.our_submenu == 'Random Lattice'):
            with self.plotbox.canvas:
                for j in range(0,N):
                    Color(1.0,0.0,0.0)
                    Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

            self.time += interval*self.speed #Here is where speed accelerates animation
            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.

            self.acucounter += 1

            if(self.plotmenu.current_tab.text == 'Energy'): #JV: instantaneous energy graphic
                #JV: The +10 is because we actually compute 10 units of time more than we represent, check computation() fucntion for more detail

                if(self.acucounter % self.update_plots == 0):
                    t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                    self.enplotax.clear()
                    self.enplotax.set_xlabel('t')
                    self.enplotax.set_ylabel('Energy')

                    self.enplotax.set_xlim([0,self.T])
                    self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])

                    #JV: We make the red line a little wider so we can see it and doesn't get hidden (in some cases) by the total energy
                    self.enplotax.plot(t[0:i],self.s.K[0:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                    self.enplotax.plot(t[0:i],self.s.U[0:i],'b-',label = 'Potential Energy')
                    self.enplotax.plot(t[0:i],self.s.K[0:i]+self.s.U[0:i],'g-',label = 'Total Energy')

                    self.enplotax.legend(loc=7)

                    self.enplotcanvas.draw()

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram

                if(self.acucounter % self.update_plots == 0):
                    vs = np.linspace(0,self.s.V.max()+0.5,100) #JV: The +0.5 is because we want to see the whole last possible bar

                    self.histax.clear()
                    self.histax.set_xlabel('v')
                    self.histax.set_ylabel('Number of particles relative')
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    self.histax.set_ylim([0,np.ceil(self.s.MB.max())])

                    self.histax.hist(self.s.V[i,:],bins=np.arange(0,self.s.V.max()+1, 0.334),rwidth=0.75,density=True,color=[0.0,0.0,1.0])
                    self.histax.text(vs[np.argmax(self.s.MB[i,:])-int(len(vs)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(vs,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()


            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                self.acuhistax.clear() #JV: We clean the graphic although we don't draw anything yet (to clean anything left in a previous simulation)
                if(self.time > 40.):

                    if(self.acucounter % self.update_plots == 0):
                        vs = np.linspace(0,self.s.V.max()+0.5,100)

                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                        self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])

                        #print(i,n,delta,int((i-int((40./self.dt)))/delta),len(self.s.Vacu))
                        self.acuhistax.hist(self.s.Vacu[int((i-int((40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
                        self.acuhistax.plot(vs,self.s.MBacu[int((i-int((40./self.dt)))/delta)],'r-')

                        self.acuhistcanvas.draw()

            elif(self.plotmenu.current_tab.text == 'Entropy'):
                if(self.acucounter % self.update_plots == 0):
                    t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                    self.extraplotax.clear()
                    self.extraplotax.set_xlabel('t')
                    self.extraplotax.set_ylabel('Entropy')

                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([self.s.entropy.min(),self.s.entropy.max()+self.s.entropy.max()*0.05])
                    self.extraplotax.plot(t[0:i],self.s.entropy[0:i],'g-')

                    self.extraplotcanvas.draw()

        elif(self.our_submenu == 'Subsystems'):
            with self.plotbox.canvas:
                for j in range(0,(self.n1)**2+(self.n2)**2):
                    if(j < self.n1**2):
                        Color(0.32,0.86,0.86)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                    else:
                        Color(0.43,0.96,0.16)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

            self.time += interval*self.speed #Here is where speed accelerates animation
            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.

            self.acucounter += 1

            #JV: Here we do the graphics for this submenu, check the comments from the previous submenu for more info
            #JV: We will draw the two subsystems in two diferent colors
            if(self.plotmenu.current_tab.text == 'Energy'):

                if(self.acucounter % self.update_plots == 0):
                    t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                    self.enplotax.clear()
                    self.enplotax.set_xlabel('t')
                    self.enplotax.set_ylabel('Energy')

                    self.enplotax.set_xlim([0,self.T])
                    self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])

                    self.enplotax.plot(t[0:i],self.s.K[0:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                    self.enplotax.plot(t[0:i],self.s.U[0:i],'b-',label = 'Potential Energy')
                    self.enplotax.plot(t[0:i],self.s.K[0:i]+self.s.U[0:i],'g-',label = 'Total Energy')

                    self.enplotax.legend(loc=7)

                    self.enplotcanvas.draw()

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram

                if(self.acucounter % self.update_plots == 0):
                    vs = np.linspace(0,self.s.V.max()+0.5,100) #JV: The +0.5 is because we want to see the whole last possible bar

                    self.histax.clear()
                    self.histax.set_xlabel('v')
                    self.histax.set_ylabel('Number of particles relative')
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    self.histax.set_ylim([0,np.ceil(self.s.MB.max())])

                    self.histax.hist([self.s.V[i,0:self.n1**2],self.s.V[i,self.n1**2:self.n1**2+self.n2**2]],bins=np.arange(0, self.s.V.max() + 1, 0.334),rwidth=0.8,density=True,color=[[0.32,0.86,0.86],[0.43,0.96,0.16]])
                    #JV: We now plot the texts that show the temperature of each types of particles
#                    self.histax.text(vs[np.argmax(self.s.MB1[i,:])],self.s.MB1[i,:].max()+self.s.MB1[i,:].max()*0.05,"T1 = "+str(np.round(self.s.T1[i],decimals=3)), fontsize=15, color = "blue", alpha = 0.75)
#                    self.histax.text(vs[np.argmax(self.s.MB2[i,:])],self.s.MB2[i,:].max()-self.s.MB2[i,:].max()*0.05-0.075,"T2 = "+str(np.round(self.s.T2[i],decimals=3)), fontsize=15, color = "green", alpha = 0.75)
                    self.histax.text(self.s.V.max()-2.5,0.8,"T1 = "+str(np.round(self.s.T1[i],decimals=3)), fontsize=15, color = "blue", alpha = 0.75)
                    self.histax.text(self.s.V.max()-1,0.8,"T2 = "+str(np.round(self.s.T2[i],decimals=3)), fontsize=15, color = "green", alpha = 0.75)
                    self.histax.text(vs[np.argmax(self.s.MB[i,:])-int(len(vs)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(vs,self.s.MB1[i,:],'b-', alpha = 0.5)
                    self.histax.plot(vs,self.s.MB2[i,:],'g-', alpha = 0.5)
                    self.histax.plot(vs,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()


            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                self.acuhistax.clear()
                if(self.time > 40.):

                    if(self.acucounter % self.update_plots == 0):
                        vs = np.linspace(0,self.s.V.max()+0.5,100)

                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                        self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])


                        self.acuhistax.hist(self.s.Vacu[int((i-int((40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
                        self.acuhistax.plot(vs,self.s.MBacu[int((i-int((40./self.dt)))/delta)],'r-')

                        self.acuhistcanvas.draw()

            elif(self.plotmenu.current_tab.text == 'Entropy'):
                if(self.acucounter % self.update_plots == 0):
                    t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                    self.extraplotax.clear()
                    self.extraplotax.set_xlabel('t')
                    self.extraplotax.set_ylabel('Entropy')

                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([self.s.entropy.min(),self.s.entropy.max()+self.s.entropy.max()*0.05])
                    self.extraplotax.plot(t[0:i],self.s.entropy[0:i],'g-')

                    self.extraplotcanvas.draw()

        elif(self.our_submenu == 'Brownian'):
            with self.plotbox.canvas:
                for j in range(0,(self.nsmall)**2+(self.nbig)**2):
                    if(j < self.nsmall**2):
                        Color(0.32,0.86,0.86)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                    else:
#                        Color(0.43,0.96,0.16)
#                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.Rbig*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.Rbig*scale/2.),size=(self.Rbig*scale,self.Rbig*scale))
#                        if(abs(self.s.X[i,j]-self.s.X[i-10,j]) < 0.5*self.L/self.R and abs(self.s.Y[i,j]-self.s.Y[i-10,j]) < 0.5*self.L/self.R):
#                            self.plotbox.canvas.add(self.obj2)
#                            self.points.append((self.s.X[i,j])*scale*self.R+w/2.)
#                            self.points.append((self.s.Y[i,j])*scale*self.R+h/2.)
#                            self.obj2.add(Color(0.43,0.96,0.16))
#                            self.obj2.add(Line(points=self.points,width = 1.5))
#                        else:
#                            print("hey")
#                            self.points.append((self.s.X[i,j])*scale*self.R+w/2.)
#                            self.points.append((self.s.Y[i,j])*scale*self.R+h/2.)
#                        print(self.s.Y[i,j]-self.s.Y[i-10,j], self.L/self.R)

                        Color(0.43,0.96,0.16)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.Rbig*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.Rbig*scale/2.),size=(self.Rbig*scale,self.Rbig*scale))
                        #JV: These lines inside the condition make the trace of the big particle. We will only draw it if we are not in the "timeinversion" mode
                        if(self.inversion == False):
                            #if(abs(self.s.X[i,j]-self.s.X[i-1,j]) < 0.3*self.L and abs(self.s.Y[i,j]-self.s.Y[i-1,j]) < 0.3*self.L):
                            self.plotbox.canvas.add(self.obj2)
#                            self.points.append((self.s.X[i,j])*scale*self.R+w/2.)
#                            self.points.append((self.s.Y[i,j])*scale*self.R+h/2.)
                            self.obj2.add(Color(0.43,0.96,0.16))
                            self.obj2.add(Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.,(self.s.Y[i,j])*scale*self.R+h/2.),size=(self.Rbig*scale/10,self.Rbig*scale/10)))
    #                        print(self.s.Y[i,j]-self.s.Y[i-1,j], self.L/self.R)


            self.time += interval*self.speed #Here is where speed accelerates animation
            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.

            self.acucounter += 1

            if(self.plotmenu.current_tab.text == 'Energy'):
                if(self.acucounter % self.update_plots == 0):
                    t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                    self.enplotax.clear()
                    self.enplotax.set_xlabel('t')
                    self.enplotax.set_ylabel('Energy')

                    self.enplotax.set_xlim([0,self.T])
                    self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])

                    self.enplotax.plot(t[0:i],self.s.K[0:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                    self.enplotax.plot(t[0:i],self.s.U[0:i],'b-',label = 'Potential Energy')
                    self.enplotax.plot(t[0:i],self.s.K[0:i]+self.s.U[0:i],'g-',label = 'Total Energy')

                    self.enplotax.legend(loc=7)

                    self.enplotcanvas.draw()

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.acucounter % self.update_plots == 0):
                    vs = np.linspace(0,self.s.V.max()+0.5,100) #JV: The +0.5 is because we want to see the whole last possible bar

                    self.histax.clear()
                    self.histax.set_xlabel('v')
                    self.histax.set_ylabel('Number of particles relative')
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    self.histax.set_ylim([0,np.ceil(self.s.MB.max())])

                    self.histax.hist([self.s.V[i,0:self.nsmall**2],self.s.V[i,self.nsmall**2:self.nsmall**2+self.nbig**2]],bins=np.arange(0, self.s.V.max() + 1, 0.334),rwidth=0.75,density=True,color=[[0.32,0.86,0.86],[0.43,0.96,0.16]])
                    self.histax.text(vs[np.argmax(self.s.MB[i,:])-int(len(vs)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(vs,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()


            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                self.acuhistax.clear()
                if(self.time > 40.):

                    if(self.acucounter % self.update_plots == 0):
                        vs = np.linspace(0,self.s.V.max()+0.5,100)

                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                        self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])


                        self.acuhistax.hist(self.s.Vacu[int((i-int((40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
                        self.acuhistax.plot(vs,self.s.MBacu[int((i-int((40./self.dt)))/delta)],'r-')

                        self.acuhistcanvas.draw()

            elif(self.plotmenu.current_tab.text == '<|x(t)-x(0)|^2>'):
                if(self.acucounter % self.update_plots == 0):
                    t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                    self.extraplotax.clear()
                    self.extraplotax.set_xlabel('t')
                    self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$')


                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([0,self.s.X2.max()+self.s.X2.max()*0.05])
                    self.extraplotax.plot(t[0:i],self.s.X2[0:i],'g-')

                    self.extraplotcanvas.draw()


        if(self.our_menu == "Walls"):
            w = self.plotbox.size[0]
            h = self.plotbox.size[1]
            b = min(w,h)
            scale = b/self.L

            self.obj = InstructionGroup()
            self.plotbox.canvas.remove(self.obj)

            #JV: We need to multiply by scale to transform from Angstrom units to Kivy units, that ajust depending on the resulution
            self.obj.add(Color(0.37,0.01,0.95))
            self.obj.add(Rectangle(pos=(w/2+self.wallpos*scale-self.wallwidth*scale/2,0), size = (self.wallwidth*scale,h/2 - self.holesize*scale/2)))
            self.obj.add(Rectangle(pos=(w/2+self.wallpos*scale-self.wallwidth*scale/2,h/2+self.holesize*scale/2), size = (self.wallwidth*scale,h/2 - self.holesize*scale/2)))
            self.plotbox.canvas.add(self.obj)

        #JV: Check if the animations has arrived at the end of the performance, if it has, it will stop
        if(i >= n):
            if(self.inversion == True):
                self.previewlist = []
                if(self.our_submenu == "Random Lattice"):
                    x = self.s.X[i,:]*self.R
                    y = self.s.Y[i,:]*self.R
                    vx = -self.s.VX[i,:]
                    vy = -self.s.VY[i,:]


                    for k in range(0,self.n**2):
                        self.previewlist.append([x[k],y[k],vx[k]*self.R,vy[k]*self.R])

                elif(self.our_submenu == "Subsystems"):
                    x = self.s.X[i,:]*self.R
                    y = self.s.Y[i,:]*self.R
                    vx = -self.s.VX[i,:]
                    vy = -self.s.VY[i,:]

                    for k in range(0,self.n1**2+self.n2**2):
                        self.previewlist.append([x[k],y[k],vx[k]*self.R,vy[k]*self.R])

                elif(self.our_submenu == "Brownian"):
                    x = self.s.X[i,:]*self.R
                    y = self.s.Y[i,:]*self.R
                    vx = -self.s.VX[i,:]
                    vy = -self.s.VY[i,:]

                    for k in range(0,self.nsmall**2+self.nbig**2):
                        self.previewlist.append([x[k],y[k],vx[k]*self.R,vy[k]*self.R])

            self.stop()


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




class particlesinaboxApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    particlesinaboxApp().run()