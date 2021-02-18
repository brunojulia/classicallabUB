# -----------------------------------------------------------
# particlesinabox.py
# Requires physystem.py and particlesinabox.kv to run
# -----------------------------------------------------------

"""
Edited and modified by Jofre Vallès Muns, March 2020 - February 2021
from the initial code by Arnau Jurado Romero.

The comments without specifying nothing are from Arnau, and the comments
like "JV: ..." are made by Jofre.

First of all, if you want to understand the theory behind this program and
know the basic workflow of it, I suggest you to have a look at the PDF
document that comes with it, in which we sum up how does this program work and
how it was made.

The sections of this program are Simulation, Demo and Game.

Simulation is the place where you can compute your own simulations with plenty
of freedom. In Demo you can see the evolution of the saved and already computed
simulations and have also a brief explanation of what is happening. Game is a
section that is only in the code, it does not appear if you open the program and
that is because it does not work as fluid as we wanted, so it is an unpolished
idea that is interesting to further develop but for now, we do not show it to
the user.
"""


"""
JV: Even though the program has changed a lot, we still include this brief
explanation made by Arnau Jurado, in which it explains the basic coding structure
of this program:


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
Config.set('kivy','window_icon','icons/icona.png')
Config.set('graphics', 'width', '1300')
Config.set('graphics', 'height', '750')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from physystem import *
from numba import jit

#Kivy imports
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout

from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.screenmanager import FadeTransition, SlideTransition

from kivy.uix.label import Label
from kivy.uix.slider import Slider
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
    change_units = ObjectProperty(None)
    cancel = ObjectProperty(None)

class warningwindow(FloatLayout):
    cancel = ObjectProperty(None)


class SimulationScreen(Screen):
    charge = 1.
    particles = np.array([])

    plot_texture = ObjectProperty()

    #Definition for the matplotlib figures of the histograms

    #JV: Moment histogram
    hist = Figure()
    histax = hist.add_subplot(111, xlabel='v', ylabel = 'Number of particles')
    histax.set_xlim([0,1])
    histax.set_ylim([0,25])
    hist.subplots_adjust(0.125,0.19,0.9,0.9) #JV: We ajust the subplot to see whole axis and their labels
    histax.yaxis.labelpad = 10 #JV: We ajust the labels to our interest
    histax.xaxis.labelpad = -0.5
    histcanvas = FigureCanvasKivyAgg(hist)

    #JV: Acomulated moment histogram
    acuhist = Figure()
    acuhistax = acuhist.add_subplot(111, xlabel='v', ylabel = 'Number of particles')
    acuhistax.set_xlim([0,1])
    acuhistax.set_ylim([0,25])
    acuhist.subplots_adjust(0.125,0.19,0.9,0.9)
    acuhistax.yaxis.labelpad = 10
    acuhistax.xaxis.labelpad = -0.5
    acuhistcanvas = FigureCanvasKivyAgg(acuhist)

    #JV: Energy (Sub)Plot
    enplot = Figure()
    enplotax = enplot.add_subplot(111, xlabel='t', ylabel = 'Energy')
    enplotax.set_xlim([0,1])
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

    def transition_SM(self):
        """JV: Screen transition: from 'simulation' to 'menu'"""
        self.stop()
        self.manager.transition = FadeTransition()
        self.manager.current = 'menu'


    def __init__(self, **kwargs):
        super(SimulationScreen, self).__init__(**kwargs)

    def s_pseudo_init(self):
        """
        JV: Here we initialize some functions and some initial values. Some of this values can be changed inside the simulation
         on the "Advanced Settings" menu. (Go to advanced_settings() for more info)
        """
        self.time = 0.
        #Here you can modify the initial time of computation and the step
        self.T = self.timeslider.value
        self.dt = 0.01 #JV: Because the time of computation is a number that ends with 0 or 5 (we can change this in the kivy file), we
        # need a dt that is multiple of 5 (0,01;0,015;...), if we don't do this we get some strange errors (...)

        #Initialization of the speed button
        self.speedindex = 3
        self.change_speed()

        #JV: Initial values of the temperatures of the particles (temp1) and the second type of particles for the subsystems menu
        self.temp1 = 3
        self.temp2 = 2

        self.compact = 1 #JV: If this variable is increased, the particles will be close to each other, instead of ocupying the whole space they will get into a "corner"

        #JV: This number is the steps it takes in the animation, to draws the plots. If it's 1, it will update each step of time, etc.
        self.update_plots = 5
        #JV: The momentum plot can be slower than the other plots, so we it is possible to only update it every certain steps inside the plotting() function. If 1 it will update like the other plots.
        self.update_momentum = 3
        self.histtype = "stepfilled" #JV: The type of histogram. {'bar', 'barstacked', 'step', 'stepfilled'}. "bar" seems to be the most appealing visually, while "step" runs much faster
        self.counter = 0 #JV: The counter of this "plotting" steps
        #JV: This counters store the last plotting point, so we can print the ones that are missing (one for the energy plot and the other for the entropy)
        self.prev_i_en = 0
        self.prev_i_ex = 0
        self.simulation = False #JV: This bool will be true when we will have calculated/loaded a simulation
        self.simulation_type = "" #JV: This variable will store the type of simulation that we have calculated

        self.si_units = False #JV: We start the simulation in reduced units, but the user is able to change this to SIUnits in the "Advanced Settings" Pop-Up

        #JV: Here we save the maximum, the minimum and the step values of the slider in the advanced-settings pop-up because we will need it. If you update the values in the Kivy file update this values aswell.
        self.dt_slider_min = 0.0005
        self.dt_slider_max = 0.02
        self.dt_slider_step = 0.0005

        #JV: And we do the same for the temperatures. Change if you modify this values in the Kivy file.
        self.temp_slider_min = 1
        self.temp_slider_max = 5
        self.temp_slider_step = 1

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.ready = False #Checks if computation is done
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)#Always, runs, shows previews. Calls the preview() function every 0.04 seconds
        self.previewlist = []
        self.progress = 0.
        self.our_submenu = 'Random Lattice' #JV: We start at the "Random Lattice" submenu
        self.our_menu = 'In a box' #JV: We start at "In a box" menu. This two variables will store in which submenu/menu we are
        self.n = 3 #JV: Modify this value if at the start you want to show more than one particle in the simulation (as a default value)
        self.n1 = 2 #JV: Modify this value if at the start you want to show more than one particle in the subsystem 1 in the subsystems submenu (as a default value)
        self.n2 = 2 #JV: Same for the second subsystem
        self.nbig = 1 #JV: Number of initial big particles in the "Brownian" submenu
        self.nsmall = 4 #JV: Number of inicial small particles in the "Brownian" submenu

        self.Rbig = 4*3.405 #JV: We define the initial value of the radius for the big particle in the "Brownian" submenu

        self.wallpos = 0 #JV: We define the initial value of the position of the wall in the "Walls" menu
        self.holesize = 20 #JV: We define the initial value of the size of the hole (we need to change the kivy file if we change this value, go to the parameters of this slider and change the "value")
        self.wallwidth = 1 #JV: We define the initial value of the width of the wall

        #Initialization of the plots
        self.histbox.add_widget(self.histcanvas)
        self.acuhistbox.add_widget(self.acuhistcanvas)
        self.enplotbox.add_widget(self.enplotcanvas)
        self.extraplotbox.add_widget(self.extraplotcanvas)

        self.mass = 1 #JV: change this if you want to change the initial mass

        #Here you can modify the units of the simulation as well as the size of the box.
        self.V0 = 0.0103 #eV
        self.R = 3.405 #A
        self.L = 200. #A
        self.M = 0.04 #kg/mol
        # self.m = 6.6323 * 10^-26 #kg

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
        """This function updates the position parameters when you click the screen"""
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
                if(self.nsmallslider.value == self.nsmall):
                    pass
                else:
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
                if(self.nsmallslider2.value == self.nsmall):
                    pass
                else:
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
                if(self.nsmallslider3.value == self.nsmall):
                    pass
                else:
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
                    self.setplotlimits(self.simulation)
                    self.extraplottab.text = "Entropy"
                    self.add_particle_list(False)
                elif(self.menu_list[i].current_tab.text  == 'Subsystems' and not(self.our_submenu == 'Subsystems')):
                    self.stop()
                    self.our_submenu = 'Subsystems'
                    self.setplotlimits(self.simulation)
                    self.extraplottab.text = "Entropy"
                    self.add_particle_list(False)
                elif(self.menu_list[i].current_tab.text  == 'Brownian' and not(self.our_submenu == 'Brownian')):
                    self.stop()
                    self.our_submenu = 'Brownian'
                    self.setplotlimits(self.simulation)
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

        #JV: We do the same as the stop() function but we do not include the update on the plots because that slows the program.
        self.pause()
        self.paused = False
        self.time = 0
        self.obj2.clear()
        self.plotbox.canvas.clear()
        self.progressbar.value = 0 #JV: We set the progress bar to 0
        #JV: Restart the counters
        self.acucounter = 0
        self.prev_i_en = 0
        self.prev_i_ex = 0

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
                    x,y = np.linspace(-self.L/2*0.95,self.wallpos-self.L*0.03,self.n),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n)
                else:
                    x,y = np.linspace(-self.L/2*0.9/self.compact,self.L/2*0.9/self.compact,self.n),np.linspace(-self.L/2*0.9/self.compact,self.L/2*0.9/self.compact,self.n)

                #JV: Physystem.py works in reduced units, so we need to make sure that we are sending the values in this units
                if(self.si_units):
                    temp = int(np.round((self.temp1 / 119.53),0))
                else:
                    temp = self.temp1

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
                    x1,y1 = np.linspace(-self.L/2*0.95,self.wallpos-self.L*0.03,self.n1),np.linspace(self.L/2*0.9,self.L/2*0.1,self.n1)
                    x2,y2 = np.linspace(-self.L/2*0.95,self.wallpos-self.L*0.03,self.n2),np.linspace(-self.L/2*0.9,-self.L/2*0.1,self.n2)
                else:
                    x1,y1 = np.linspace(-self.L/2*0.9,-self.L/2*0.1,self.n1),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n1)
                    x2,y2 = np.linspace(self.L/2*0.1,self.L/2*0.9,self.n2),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n2)

                if(self.si_units):
                    temp1 = int(np.round((self.temp1 / 119.53),0))
                else:
                    temp1 = self.temp1

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

                if(self.si_units):
                    temp2 = int(np.round((self.temp2 / 119.53),0))
                else:
                    temp2 = self.temp2

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
                self.nsmall = int(self.nsmallslider.value)
            elif(self.our_menu == "Free!"):
                self.nsmall = int(self.nsmallslider2.value)

            if(inversion == False):
                self.inversion = False
                #JV: corresponding to the small particles variables
                if(self.our_menu == "Walls"):
                    x,y1 = np.linspace(-self.L/2*0.9,self.wallpos-self.L*0.05,self.nsmall),np.linspace(self.L/2*0.9,self.L/2*0.12,int(self.nsmall/2))
                    y2 = np.linspace(-self.L/2*0.9,-self.L/2*0.12,int(self.nsmall/2))
                    y = np.append(y1,y2)
                else:
                    x1,y1 = np.linspace(-self.L/2*0.9,-self.L/2*0.1,int(self.nsmall/2)),np.linspace(-self.L/2*0.9,-self.L/2*0.1,int(self.nsmall/2))
                    x2,y2 = np.linspace(self.L/2*0.9,self.L/2*0.1,int(self.nsmall/2)),np.linspace(self.L/2*0.9,self.L/2*0.1,int(self.nsmall/2))
                    x = np.append(x1,x2)
                    y = np.append(y1,y2)
                    # x,y = np.linspace(-self.L/2*0.95,self.L/2*0.95,self.nsmall),np.linspace(-self.L/2*0.95,self.L/2*0.95,self.nsmall)

                if(self.si_units):
                    temp = int(np.round((self.temp1 / 119.53),0))
                else:
                    temp = self.temp1

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
        self.pcbutton.background_normal = 'icons/compute.png'
        self.pcbutton.background_down = 'icons/computeb.png'
        self.statuslabel.text = 'Not Ready'

    def reset_particle_list(self):
        #Empties particle list
        self.particles = np.array([])
        self.previewlist = []
        self.ready = False
        self.pcbutton.background_normal = 'icons/compute.png'
        self.pcbutton.background_down = 'icons/computeb.png'
        self.statuslabel.text = 'Not Ready'

    def playcompute(self):
        #If ready is false it starts the computation, if true starts the animation
        if(self.ready==False):
            self.statuslabel.text = 'Computing...'
            Clock.schedule_once(self.computation)

        elif(self.ready==True):
            if(self.running==False):
                time.sleep(0.5)
                self.timer = Clock.schedule_interval(self.animate,0.04)
                self.timer2 = Clock.schedule_interval(self.plotting, 0.04*self.update_plots)
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

        #JV: We put the +20 because we want to genarate more values than the ones we will show, we do that to be able to stop the simulation when it ends, to avoid problems
        #JV: It's not an elegant solution, but it works just fine. Check in the future, we could correct this maybe changing how the time steps work
        #JV: If we modify as we said in the previous line, change the +10 in the part of energy representation in the animate() function
        if(self.si_units):
            #JV: Physystem.py works with reduced units, so we need to transform this values if we are using currently the SI Units
            self.s.solveverlet(int(5 * np.round(self.T /(2.16 * 5)))+20,float(5 * np.round(self.dt /(2.16 * 5), 4)))
        else:
            self.s.solveverlet(self.T+20,self.dt)

        print('---Computation End---')
        print('Exec time = ',time.time() - start)
        print("")

        last = len(self.s.U)-1 #JV: This variable stores the index of the last variable in this arrays, so we can access it easier
        print("Initial energy: ",self.s.K[0]+self.s.U[0],"Final energy: ",self.s.U[last]+self.s.K[last])
        print("Relative increment of energy: ", (abs((self.s.K[0]+self.s.U[0])-(self.s.U[last]+self.s.K[last]))/(self.s.K[0]+self.s.U[0]))*100,"%")

        self.ready = True
        self.pcbutton.background_normal = 'icons/play.png'
        self.pcbutton.background_down = 'icons/playb.png'
        self.statuslabel.text = 'Ready'

        #This also saves the temperatures and energies to files
        np.savetxt('k_energy.dat',self.s.K,fmt='%10.5f')
        np.savetxt('u_energy.dat',self.s.U,fmt='%10.5f')
        np.savetxt('t_energy.dat',self.s.K + self.s.U,fmt='%10.5f')
        np.savetxt('temps.dat',self.s.T,fmt='%10.5f')

        #JV: We now check if we have to change the units of the calculations, so we get the values in the correct desired units
        if(self.si_units):
            #JV: Now this energy values are in aJ (atto-Joules, 10^-18 J)
            self.s.K = np.array(self.s.K) * self.V0 * 16.022 #JV: This value is to convert from eV to aJ
            self.s.U = np.array(self.s.U) * self.V0 * 16.022

            #JV: The time values are already transformed

            #JV: The velocity values are in m/s
            self.s.V = np.array(self.s.V) * 155.42 #JV: We get this number from sqrt(self.V0/self.m) in SI Units
            self.s.Vacu = np.array(self.s.Vacu) * 155.42

            #JV: The temperature values are in Kelvin
            self.s.T = np.array(self.s.T) * 119.53 #JV: We get this value from self.V0/kB (Boltzmann-constant)

            if(self.our_submenu == "Subsystems"):
                self.s.T1 = np.array(self.s.T1) * 119.53
                self.s.T2 = np.array(self.s.T2) * 119.53

            if(self.simulation_type != "Brownian"):
                #JV: The entropy in aJ/K (10^-18 J/K)
                self.s.entropy = np.array(self.s.entropy) * 16.022/119.53
            else:
                #JV: The distance^2 value now is in Angstrom^2 (10^-10 m)^2
                self.s.X2 = np.array(self.s.X2) * (self.R)**2

        self.simulation = True #JV: This bool is True if we have a simulation calculated/loaded
        self.simulation_type = self.our_submenu #JV: This variable stores the type of simulation that we have calculated
        self.setplotlimits(simulation = self.simulation) #JV: We call the function that will update the limits of our (matplotlib) plots


    def pause(self):
        if(self.running==True):
            self.paused = True
            self.timer.cancel()
            self.timer2.cancel()
            self.running = False
        else:
            pass

    def stop(self):
        self.pause()
        self.paused = False
        self.time = 0
        self.obj2.clear()
        self.plotbox.canvas.clear()
        self.progressbar.value = 0 #JV: We set the progress bar to 0
        self.setplotlimits(simulation = self.simulation) #JV: We don't know if we have calculated/loaded a simulation yet
        #JV: Restart the counters
        self.acucounter = 0
        self.prev_i_en = 0
        self.prev_i_ex = 0

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
        info_img = "icons/gray_demo.png" #JV: This image is just a gray plain color. If you want to show an info image, change this to the path of this image (e.g. "saves_img/Subsys_1.png")
        # color gris per defecte = "saves_img/Subsys_1.png"
        savedata = np.array([self.s,self.T,self.dt,self.L,self.previewlist,self.our_menu,self.our_submenu,self.n1,self.n2,self.nsmall,self.wallpos,self.holesize,self.Rbig,self.temp1,self.temp2,self.compact,self.si_units,info_img])
        with open(os.path.join(path,name+'.dat'),'wb') as file:
            pickle.dump(savedata,file)
        self.dismiss_popup()

    def savepopup(self):
        content = savewindow(save = self.save, cancel = self.dismiss_popup)
        self._popup = Popup(title='Save File', content = content, size_hint=(1,1))
        self._popup.open()

    def advanced_settings(self):
        content = settingswindow(change_settings = self.change_settings, change_units = self.change_units, cancel = self.dismiss_advanced_settings_popup)
        self._popup = Popup(title='Advanced Settings', content = content, size_hint = (0.6,1), auto_dismiss=False)

        #JV: We display the actual values of these parameters
        if(self.si_units):
            self._popup.content.si_button.state = "down"
            self._popup.content.reduced_button.state = "normal"

            #JV: We now display the dt settings in the correct SI Units
            self._popup.content.dt_slider.min = float(np.round(self.dt_slider_min * 2.16, 3)) #JV: Go to change_units() to understand the "2.16" factor
            self._popup.content.dt_slider.max = float(np.round(self.dt_slider_max * 2.16, 3))
            self._popup.content.dt_slider.step = float(np.round(self.dt_slider_step * 2.16, 3))
            self._popup.content.dt_units_label.text = "(·10^-12 s)"

            #JV: We display the correct temperature units
            self._popup.content.temp1_slider.min = float(np.round(self.temp_slider_min * 119.53, 0)) #JV: Go to change_units() to understand the "119.53" factor
            self._popup.content.temp1_slider.max = float(np.round(self.temp_slider_max * 119.53, 0))
            self._popup.content.temp1_slider.step = float(np.round(self.temp_slider_step * 119.53, 0))

            self._popup.content.temp2_slider.min = float(np.round(self.temp_slider_min * 119.53, 0))
            self._popup.content.temp2_slider.max = float(np.round(self.temp_slider_max * 119.53, 0))
            self._popup.content.temp2_slider.step = float(np.round(self.temp_slider_step * 119.53, 0))

            self._popup.content.units_temp1.text = " K"
            self._popup.content.units_temp2.text = " K"
        else:
            self._popup.content.si_button.release = "normal"
            self._popup.content.reduced_button.state = "down"

            self._popup.content.dt_slider.min = self.dt_slider_min
            self._popup.content.dt_slider.max = self.dt_slider_max
            self._popup.content.dt_slider.step = self.dt_slider_step

            self._popup.content.temp1_slider.min = self.temp_slider_min
            self._popup.content.temp1_slider.max = self.temp_slider_max
            self._popup.content.temp1_slider.step = self.temp_slider_step

            self._popup.content.temp2_slider.min = self.temp_slider_min
            self._popup.content.temp2_slider.max = self.temp_slider_max
            self._popup.content.temp2_slider.step = self.temp_slider_step

            self._popup.content.dt_units_label.text = ""
            self._popup.content.units_temp1.text = ""
            self._popup.content.units_temp2.text = ""

        self._popup.content.rbig_slider.value = int(self.Rbig/self.R)
        self._popup.content.boxlength_slider.value = int(self.L)
        self._popup.content.dt_slider.value = float(self.dt)
        self._popup.content.temp1_slider.value = int(self.temp1)
        self._popup.content.temp2_slider.value = int(self.temp2)
        self._popup.content.compact_slider.value = self.compact
        self._popup.open()

    def change_settings(self):
        #JV: We change the parameters if there is some change
        if(self._popup.content.rbig_slider.value != int(self.Rbig/self.R) or self._popup.content.boxlength_slider.value != self.L or self._popup.content.dt_slider.value != self.dt or self._popup.content.temp1_slider.value != self.temp1 or self._popup.content.temp2_slider.value != self.temp2 or self._popup.content.compact_slider.value != self.compact):
            self.Rbig = self._popup.content.rbig_slider.value * self.R
            self.L = self._popup.content.boxlength_slider.value
            self.dt = self._popup.content.dt_slider.value
            self.temp1 = self._popup.content.temp1_slider.value
            self.temp2 = self._popup.content.temp2_slider.value
            self.compact = self._popup.content.compact_slider.value
            self.add_particle_list(False)
        if(self._popup.content.si_button.state == "down" and self.si_units == False):
            self.si_units = True
            self.change_units()
        elif(self._popup.content.reduced_button.state == "down" and self.si_units == True):
            self.si_units = False
            self.change_units()
        else:
            pass


    def change_units(self):
        """JV: This function updates the units of the simulation. Remember that in reduced units, we use self.V0 (the attractive well of the particle)
         as the unit of energy, self.R (the distance of hard sphere of the particle) as the unit of length and self.m (the mass of the particle) as the
         unit of mass. To get the SI Units we only need to do operations with this mentioned values."""
        if(self.si_units): #JV: We want the simulation in SI Units, so we update the corresponding values
            if(self.simulation): #JV: If we have loaded a simulation, then we update its values aswell
                 #JV: Now this energy values are in aJ (atto-Joules, 10^-18 J)
                self.s.K = np.array(self.s.K) * self.V0 * 16.022 #JV: This value is to convert from eV to aJ
                self.s.U = np.array(self.s.U) * self.V0 * 16.022

                #JV: The time values are in ps (pico-seconds, 10^-12 s)
                self.T = int(self.T * 2.16) #JV: We get this number doing self.R*sqrt(self.m/self.V0) in SI Units
                self.timeslider.min = int(self.timeslider.min * 2.16)
                self.timeslider.max = int(self.timeslider.max * 2.16)
                self.timeslider.value = self.T
                self.dt = np.round(self.dt * 2.16, 4)

                #JV: The velocity values are in m/s
                self.s.V = np.array(self.s.V) * 155.42 #JV: We get this number from sqrt(self.V0/self.m) in SI Units
                self.s.Vacu = np.array(self.s.Vacu) * 155.42

                #JV: The temperature values are in Kelvin
                self.s.T = np.array(self.s.T) * 119.53 #JV: We get this value from self.V0/kB (Boltzmann-constant)
                self.temp1 = np.round(self.temp1*119.53, 0)
                self.temp2 = np.round(self.temp2*119.53, 0)

                if(self.simulation_type == "Subsystems"):
                    self.s.T1 = np.array(self.s.T1) * 119.53
                    self.s.T2 = np.array(self.s.T2) * 119.53

                if(self.simulation_type != "Brownian"):
                    #JV: The entropy in aJ/K (10^-18 J/K)
                    self.s.entropy = np.array(self.s.entropy) * 16.022/119.53
                else:
                    #JV: The distance^2 value now is in Angstrom^2 (10^-10 m)^2
                    self.s.X2 = np.array(self.s.X2) * (self.R)**2
            else:
                #JV: Time values in ps (pico-seconds, 10^-12 s)
                self.T = int(self.T * 2.16)
                self.timeslider.min = int(self.timeslider.min * 2.16)
                self.timeslider.max = int(self.timeslider.max * 2.16)
                self.timeslider.value = self.T
                self.dt = float(np.round(self.dt * 2.16, 4))
                self.temp1 = np.round(self.temp1*119.53, 0)
                self.temp2 = np.round(self.temp2*119.53, 0)

            self.stop()
        else: #JV: To go from SI Units to reduced Units, we only have to divide by the factors in the previous condition
            if(self.simulation):
                 #JV: Now this energy values are in the potencial-attractive well of the particle
                self.s.K = np.array(self.s.K) / (self.V0 * 16.022) #JV: This value is to convert from eV to aJ
                self.s.U = np.array(self.s.U) / (self.V0 * 16.022)

                #JV: Time values
                self.T = int(5 * np.round(self.T /(2.16 * 5))) #JV: We get this number doing self.R*sqrt(self.m/self.V0) in SI Units. We multiply and divide by 5 so we get a number multiple of 5, this helps when we will have to compute
                self.timeslider.min = int(5 * np.round(self.timeslider.min /(2.16 * 5)))
                self.timeslider.max = int(5 * np.round(self.timeslider.max /(2.16 * 5)))
                self.timeslider.value = self.T
                self.dt = float(5 * np.round(self.dt /(2.16 * 5), 4))

                #JV: Velocity values
                self.s.V = np.array(self.s.V) / 155.42 #JV: We get this number from sqrt(self.V0/self.m) in SI Units
                self.s.Vacu = np.array(self.s.Vacu) / 155.42


                #JV: Temperature values
                self.s.T = np.array(self.s.T) / 119.53 #JV: We get this value from self.V0/kB (Boltzmann-constant)
                self.temp1 = int(np.round((self.temp1 / 119.53),0))
                self.temp2 = int(np.round((self.temp2 / 119.53),0))

                if(self.simulation_type == "Subsystems"):
                    self.s.T1 = np.array(self.s.T1) / 119.53
                    self.s.T2 = np.array(self.s.T2) / 119.53

                if(self.simulation_type != "Brownian"):
                    #JV: The entropy in aJ/K (10^-18 J/K)
                    self.s.entropy = np.array(self.s.entropy) * 119.53/16.022
                else:
                    #JV: The distance^2 value
                    self.s.X2 = np.array(self.s.X2) / (self.R)**2
            else:
                self.T = int(5 * np.round(self.T /(2.16 * 5))) #JV: We get this number doing self.R*sqrt(self.m/self.V0) in SI Units
                self.timeslider.min = int(5 * np.round(self.timeslider.min /(2.16 * 5)))
                self.timeslider.max = int(5 * np.round(self.timeslider.max /(2.16 * 5)))
                self.timeslider.value = self.T
                self.dt = float(5 * np.round(self.dt /(2.16 * 5), 4))
                self.temp1 = int(np.round((self.temp1 / 119.53),0))
                self.temp2 = int(np.round((self.temp2 / 119.53),0))
            self.stop()

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
        self.temp1 = savedata[13]
        self.temp2 = savedata[14]
        self.compact = savedata[15]
        self.si_units = savedata[16]

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
                self.nsmallslider.value = self.nsmall

            elif(self.our_menu == "Free!"):
                self.freemenu.switch_to(self.brwmenu2)
                self.nsmallslider2.value = self.nsmall

            elif(self.our_menu == "Walls"):
                self.wallmenu.switch_to(self.brwmenu3)
                self.nsmallslider3.value = self.nsmall

        self.ready = True
        self.pcbutton.background_normal = 'icons/play.png'
        self.pcbutton.background_down = 'icons/playb.png'
        self.statuslabel.text = 'Ready'
        print("")
        print("")
        print('Loaded simulation {} with computation'.format(name))

        last = len(self.s.U)-1 #JV: This variable stores the index of the last variable in this arrays, so we can access it easier
        print("Initial energy: ",self.s.K[0]+self.s.U[0],"Final energy: ",self.s.U[last]+self.s.K[last])
        print("Relative increment of energy: ", (abs((self.s.K[0]+self.s.U[0])-(self.s.U[last]+self.s.K[last]))/(self.s.K[0]+self.s.U[0]))*100,"%")
        if(demo==False):
            self.dismiss_popup()

        self.simulation = True #JV: This bool is True if we have a simulation calculated/loaded
        self.simulation_type = self.our_submenu #JV: This variable stores the type of simulation that we have calculated
        self.setplotlimits(simulation = self.simulation) #JV: Reset and adapt the matplotlib plots


    def loadpopup(self):
        content = loadwindow(load = self.load, cancel = self.dismiss_popup)
        self._popup = Popup(title='Load File', content = content, size_hint=(1,1))
        self._popup.open()


    def dismiss_popup(self):
        #JV: Closes the Pop-Up
        self._popup.dismiss()

    def dismiss_advanced_settings_popup(self, button_pressed):
        #JV: Closes the Pop-Up in the advanced-settings window
        if(button_pressed == "si"):
            self._popup.content.si_button.state = "down"
            self._popup.content.reduced_button.state = "normal"
        if(button_pressed == "red"):
            self._popup.content.si_button.release = "normal"
            self._popup.content.reduced_button.state = "down"

        self.change_settings()
        self._popup.dismiss()


    def timeinversion(self):
        """JV: This funcion is the one responsible of calling the others functions when we want to make
        the time inversions, allowing us to compute the path of the particles when inverting its velocity,
        so we can see if the particles reach the first initial state. We change the time of computation so
        it reaches to the initial positions and then it stops."""

        #JV: We have this condition so you are only able to compute this time inversion if you haven't done it before (in this simulation)
        # Doing this we make sure the user doesn't enter in a loop of time inversions that are not what we want ( :) )
        if(self.inversion == False and self.time > 5.):
            if(self.si_units):
                self.T = self.time * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
            else:
                self.T = self.time
            self.stop()
            self.timeslider.value = self.T

            self.add_particle_list(True)
        pass

    def plotting(self,interval):
        """JV: This function is the responsible of updating the plots.

        I think that the MB curve needs some ajust, I'm not sure it works properly'
        """
        i = int(self.time/self.dt)
        pause = 0.001
        n_bins = 15 #JV: The number of bins that we will be plotting in the histogram

        if(self.si_units):
            delta = 5*2.16/self.dt
        else:
            delta = 5./self.dt

        if(self.our_submenu == 'Random Lattice'):
            if(self.plotmenu.current_tab.text == 'Energy'): #JV: instantaneous energy graphic
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.U[self.prev_i_en:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i]+self.s.U[self.prev_i_en:i],'g-',label = 'Total Energy')
                self.enplotcanvas.draw()
                self.prev_i_en = i
                plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.counter % self.update_momentum == 0):
                    self.histax.cla()
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 0.5*len(self.s.V[i,:])
                    self.histax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.histax.set_xlabel('v (m/s)')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+50)/n_bins
                        hist,bins = np.histogram(self.s.V[i,:], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        center = (bins[:-1] + bins[1:]) / 2
                        self.histax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.histax.set_xlabel('v')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+1)/n_bins
                        hist,bins = np.histogram(self.s.V[i,:], bins=np.arange(0,self.s.V.max()+1,bin_width))
                        center = (bins[:-1] + bins[1:]) / 2
                        self.histax.bar(center, hist, align='center', width=bin_width*0.8)
                    self.s.MB[i,:] = self.s.MB[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.
                    if(self.si_units):
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3))+" K", fontsize=15, color = "red", alpha = 0.85)
                    else:
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(self.vsplot,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                if(self.si_units):
                    wait_time = 30. * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
                else:
                    wait_time = 30.
                if(self.time > wait_time):
                    self.acuhistax.cla()
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 1
                    self.acuhistax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.acuhistax.set_xlabel('v (m/s)')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    self.acuhistcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Entropy'):
                self.extraplotax.plot(self.tplot[self.prev_i_ex:i],self.s.entropy[self.prev_i_ex:i],'g-')
                self.extraplotcanvas.draw()
                self.prev_i_ex = i
                plt.pause(pause)

        elif(self.our_submenu == 'Subsystems'):
            #JV: Here we do the graphics for this submenu, check the comments from the previous submenu for more info
            #JV: We will draw the two subsystems in two diferent colors
            if(self.plotmenu.current_tab.text == 'Energy'):
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.U[self.prev_i_en:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i]+self.s.U[self.prev_i_en:i],'g-',label = 'Total Energy')
                self.enplotcanvas.draw()
                self.prev_i_en = i
                plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.counter % self.update_momentum == 0):
                    self.histax.cla()
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 0.3*len(self.s.V[i,:])
                    self.histax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.histax.set_xlabel('v (m/s)')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+50)/int(n_bins*0.7)
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.n1**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.n1**2:self.n1**2+self.n2**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
                        self.histax.bar(center2, hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16])
                    else:
                        self.histax.set_xlabel('v')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+1)/int(n_bins*0.7)
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.n1**2], bins=np.arange(0,self.s.V.max()+1,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.n1**2:self.n1**2+self.n2**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
                        self.histax.bar(center2, hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16])
#                    self.hist.subplots_adjust(0.125,0.19,0.9,0.9)
#                    self.histax.yaxis.labelpad = 10
#                    self.histax.xaxis.labelpad = -0.5
                    # self.histax.hist([self.s.V[i,0:self.n1**2], self.s.V[i,self.n1**2:self.n1**2+self.n2**2]], bins=np.arange(0, self.s.V.max() + 1, 0.67), rwidth=0.8, density=True, color=[[0.32,0.86,0.86],[0.43,0.96,0.16]], histtype = self.histtype)
                    self.s.MB[i,:] = self.s.MB[i,:] * y_lim #JV: The MB curves go from 0 to 1, so we need to adapt them to our plot.
                    self.s.MB1[i,:] = self.s.MB1[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.
                    self.s.MB2[i,:] = self.s.MB2[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.

                    #JV: We now plot the texts that show the temperature of each types of particles
                    if(self.si_units):
                        self.histax.text(self.vsplot[35],y_lim*0.8,"T1 = "+str(np.round(self.s.T1[i],decimals=3))+" K", fontsize = 12, color = "blue", alpha = 0.75)
                        self.histax.text(self.vsplot[70],y_lim*0.8,"T2 = "+str(np.round(self.s.T2[i],decimals=3))+" K", fontsize = 12, color = "green", alpha = 0.75)
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3))+" K", fontsize = 12, color = "red", alpha = 0.85)
                    else:
                        self.histax.text(self.vsplot[45],y_lim*0.8,"T1 = "+str(np.round(self.s.T1[i],decimals=3)), fontsize = 12, color = "blue", alpha = 0.75)
                        self.histax.text(self.vsplot[75],y_lim*0.8,"T2 = "+str(np.round(self.s.T2[i],decimals=3)), fontsize = 12, color = "green", alpha = 0.75)
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize = 12, color = "red", alpha = 0.85)
                    self.histax.plot(self.vsplot,self.s.MB1[i,:],'b-', alpha = 0.5)
                    self.histax.plot(self.vsplot,self.s.MB2[i,:],'g-', alpha = 0.5)
                    self.histax.plot(self.vsplot,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                if(self.si_units):
                    wait_time = 30. * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
                else:
                    wait_time = 30.
                if(self.time > wait_time):
                    self.acuhistax.cla()
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 1
                    self.acuhistax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.acuhistax.set_xlabel('v (m/s)')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    self.acuhistcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Entropy'):
                self.extraplotax.plot(self.tplot[self.prev_i_ex:i],self.s.entropy[self.prev_i_ex:i],'g-')
                self.extraplotcanvas.draw()
                self.prev_i_ex = i
                plt.pause(pause)

        elif(self.our_submenu == 'Brownian'):
            if(self.plotmenu.current_tab.text == 'Energy'):
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.U[self.prev_i_en:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i]+self.s.U[self.prev_i_en:i],'g-',label = 'Total Energy')
                self.enplotcanvas.draw()
                self.prev_i_en = i
                plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.counter % self.update_momentum == 0):
                    self.histax.cla()
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 0.6*len(self.s.V[i,:])
                    self.histax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.histax.set_xlabel('v (m/s)')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+50)/n_bins
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.nsmall**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.nsmall**2:self.nsmall**2+self.nbig**2], bins=1)
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center2, y_lim*hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16],alpha=0.8)
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
                        # self.histax.hist(self.s.V[i,:],bins=np.arange(0,self.s.V.max()+10, 100),density=True,color=[0.0,0.0,1.0], histtype = self.histtype)
                    else:
                        self.histax.set_xlabel('v')
                        self.histax.set_ylabel('Number of particles')
                        # self.histax.hist(self.s.V[i,:],bins=np.arange(0,self.s.V.max()+1, 0.334),rwidth=0.75,density=True,color=[0.0,0.0,1.0], histtype = self.histtype)
                        bin_width = (self.s.V.max()+1)/n_bins
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.nsmall**2], bins=np.arange(0,self.s.V.max()+1,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.nsmall**2:self.nsmall**2+self.nbig**2])
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center2, y_lim*hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16],alpha=0.8)
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
#                    self.hist.subplots_adjust(0.125,0.19,0.9,0.9)
#                    self.histax.yaxis.labelpad = 10
#                    self.histax.xaxis.labelpad = -0.5
                    self.s.MB[i,:] = self.s.MB[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.
                    if(self.si_units):
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3))+" K", fontsize=15, color = "red", alpha = 0.85)
                    else:
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(self.vsplot,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()
                    plt.pause(pause)

                    # self.histax.hist([self.s.V[i,0:self.nsmall**2],self.s.V[i,self.nsmall**2:self.nsmall**2+self.nbig**2]],bins=np.arange(0, self.s.V.max() + 1, 0.334),rwidth=0.75,density=True,color=[[0.32,0.86,0.86],[0.43,0.96,0.16]], histtype = self.histtype)
                    # self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)

            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                if(self.si_units):
                    wait_time = 30. * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
                else:
                    wait_time = 30.
                if(self.time > wait_time):
#                    vs = np.linspace(0,self.s.V.max()+0.5,100)
                    self.acuhistax.cla()
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 1
                    self.acuhistax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.acuhistax.set_xlabel('v (m/s)')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                        # self.acuhistax.hist(self.s.Vacu[int((i-int((wait_time/(2.16*self.dt))))/delta)],bins=np.arange(0, self.s.V.max() + 10, 70),density=True, histtype = self.histtype)
                    else:
                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                        # self.acuhistax.hist(self.s.Vacu[int((i-int((wait_time/self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.35),density=True, histtype = self.histtype)
                    # self.acuhistax.plot(self.vsplot,self.s.MBacu[int((i-int((wait_time/self.dt)))/delta)]*y_lim,'r-')
                    self.acuhistcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == '<|x(t)-x(0)|^2>'):
#                t = np.arange(self.dt,self.T+10+self.dt,self.dt)
                self.extraplotax.plot(self.tplot[self.prev_i_ex:i],self.s.X2[self.prev_i_ex:i],'g-')
                self.extraplotcanvas.draw()
                self.prev_i_ex = i
                plt.pause(pause)

        self.counter += 1

    def setplotlimits(self,simulation):
        """JV: This function updates the limits of the matplotlib plots. If there is a simulation loaded, it will adapt the plots limits to it."""

        #JV: Energy plot
        self.enplotax.cla()
        if(self.si_units): #JV: If true, then we will add the units on the labels
            self.enplotax.set_xlabel('t (·$10^{-12}$ s)')
            self.enplotax.set_ylabel('Energy (·$10^{-18}$ J)')
        else:
            self.enplotax.set_xlabel('t')
            self.enplotax.set_ylabel('Energy')
        self.enplot.subplots_adjust(0.125,0.19,0.9,0.9)
        self.enplotax.yaxis.labelpad = 10
        self.enplotax.xaxis.labelpad = -0.5

        #JV: Momentum plot
        self.histax.cla()
        if(self.si_units):
            self.histax.set_xlabel('v (m/s)')
            self.histax.set_ylabel('Number of particles')
        else:
            self.histax.set_xlabel('v')
            self.histax.set_ylabel('Number of particles')
        self.hist.subplots_adjust(0.125,0.19,0.9,0.9)
        self.histax.yaxis.labelpad = 10
        self.histax.xaxis.labelpad = -0.5

        #JV: Accumulated momentum
        self.acuhistax.cla()
        if(self.si_units):
            self.acuhistax.set_xlabel('v (m/s)')
            self.acuhistax.set_ylabel('Number of particles relative')
        else:
            self.acuhistax.set_xlabel('v')
            self.acuhistax.set_ylabel('Number of particles relative')
        self.acuhist.subplots_adjust(0.125,0.19,0.9,0.9)
        self.acuhistax.yaxis.labelpad = 10
        self.acuhistax.xaxis.labelpad = -0.5

        #JV: Entropy/
        self.extraplotax.cla()
        if(self.our_submenu == 'Brownian'):
            if(self.si_units):
                self.extraplotax.set_xlabel('t (·$10^{-12}$ s)')
                self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$ (·$10^{-20}$ $m^2$)')
            else:
                self.extraplotax.set_xlabel('t')
                self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$')
        else:
            if(self.si_units):
                self.extraplotax.set_xlabel('t (·$10^{-12}$ s)')
                self.extraplotax.set_ylabel('Entropy (·$10^{-18}$ J/K)')
            else:
                self.extraplotax.set_xlabel('t')
                self.extraplotax.set_ylabel('Entropy')

        self.extraplot.subplots_adjust(0.125,0.19,0.9,0.9)
        self.extraplotax.yaxis.labelpad = 10
        self.extraplotax.xaxis.labelpad = -0.5

        if(simulation):
            n = int(self.T/self.dt)
            self.tplot = np.arange(self.dt,self.T+20+self.dt,self.dt)
            self.vsplot = np.linspace(0,self.s.V.max()+0.5,100)

            #JV: Energy plot
            self.enplotax.set_xlim([0,self.T])
            #JV: We make the red line a little wider so we can see it and doesn't get hidden (in some cases) by the total energy
            self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])
            self.enplotax.plot(0,self.s.K[0],'r-',label = 'Kinetic Energy', linewidth = 2.2)
            self.enplotax.plot(0,self.s.U[0],'b-',label = 'Potential Energy')
            self.enplotax.plot(0,self.s.K[0]+self.s.U[0],'g-',label = 'Total Energy')
            self.enplotax.legend(loc=7)

            #JV: Momentum plot
            self.histax.set_xlim([0,self.s.V.max()+0.5])
            self.histax.set_ylim([0,0.6*len(self.s.V[0,:])])

            #JV: Accumulated momentum
            self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])

            #JV: Entropy/
            if(self.our_submenu == 'Brownian'):
                if(self.simulation_type == "Brownian"):
                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([0,self.s.X2.max()+self.s.X2.max()*0.05])
                else:
                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([0,5])
            else:
                if(self.simulation_type != "Brownian"):
                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([self.s.entropy.min(),self.s.entropy.max()+self.s.entropy.max()*0.05])
                else:
                    self.extraplotax.set_xlim([0,self.T])
                    self.extraplotax.set_ylim([0,5])
        else:
            #JV: Energy plot
            self.enplotax.set_xlim([0,self.T])
            self.enplotax.set_ylim([0,25])

            #JV: Momentum plot
            self.histax.set_xlim([0,6])
            self.histax.set_ylim([0,4])

            #JV: Accumulated momentum
            self.acuhistax.set_xlim([0,6])
            self.acuhistax.set_ylim([0,1])

            #JV: Entropy/
            self.extraplotax.set_xlim([0,self.T])
            self.extraplotax.set_ylim([0,5])

        #JV: Now draw this changes in the plot
        self.enplotcanvas.draw()
        self.histcanvas.draw()
        self.acuhistcanvas.draw()
        self.extraplotcanvas.draw()

    def preview(self,interval):
        """Draws the previews of the particles when the animation is not running or before adding
        the preview of the lattice mode before adding is not programmed (mainly because it is a random process)"""

        if(self.running == False and self.paused == False):
            #JV: We can add conditions if we need it for different menus (like we do for the "walls" menu)
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

        if(self.our_menu == "Walls"):
            self.plotbox.canvas.add(self.obj)

        if(self.our_submenu == 'Random Lattice'):
            with self.plotbox.canvas:
                for j in range(0,N):
                    Color(1.0,0.0,0.0)
                    Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

            #Here is where speed accelerates animation
            if(self.si_units):
                self.time += interval*self.speed*2.16 #JV: Go to change_units() to understand the "2.16" factor"
            else:
                self.time += interval*self.speed

            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.
            self.acucounter += 1

        elif(self.our_submenu == 'Subsystems'):
            with self.plotbox.canvas:
                for j in range(0,(self.n1)**2+(self.n2)**2):
                    if(j < self.n1**2):
                        Color(0.32,0.86,0.86)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                    else:
                        Color(0.43,0.96,0.16)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

            #Here is where speed accelerates animation
            if(self.si_units):
                self.time += interval*self.speed*2.16 #JV: Go to change_units() to understand the "2.16" factor"
            else:
                self.time += interval*self.speed

            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.
            self.acucounter += 1

        elif(self.our_submenu == 'Brownian'):
            with self.plotbox.canvas:
                for j in range(0,(self.nsmall)**2+(self.nbig)**2):
                    if(j < self.nsmall**2):
                        Color(0.32,0.86,0.86)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                    else:
                        Color(0.43,0.96,0.16)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.Rbig*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.Rbig*scale/2.),size=(self.Rbig*scale,self.Rbig*scale))

                        #JV: These lines inside the condition make the trace of the big particle. We will only draw it if we are not in the "timeinversion" mode.
                        #JV: THIS FEATURE IS NOT ACTIVE, NEEDS UPDATE: The trace, as it is drawn an ellipse each frame, it stacks on the canvas an doesn't update when
                        # changing the size of the window, needs to change so each frame draws ALL the ellipses, don't know how this will affect at the FPS
#                        if(self.inversion == False):
#                            self.plotbox.canvas.add(self.obj2)
#                            self.obj2.add(Color(0.43,0.96,0.16))
#                            self.obj2.add(Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.,(self.s.Y[i,j])*scale*self.R+h/2.),size=(self.Rbig*scale/15,self.Rbig*scale/15)))


            #Here is where speed accelerates animation
            if(self.si_units):
                self.time += interval*self.speed*2.16 #JV: Go to change_units() to understand the "2.16" factor"
            else:
                self.time += interval*self.speed

            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.
            self.acucounter += 1

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

@jit(nopython=True)
def close_particles_list(r2,m,N,L):
    """JV: This functions returns a NxM matrix where each rows saves m indexs corresponding to the m
    closest particles. We will use the r2 matrix calculated in fv() function (go there for more information),
    that contains the distance between our particles"""
    dist = r2.copy()
#        N = self.particles.size
#        L = self.param[2]
    close_list = np.zeros((N,m))
    temp_index = np.zeros((m))

    for i in range(0, N):
        temp_index = np.zeros((m))
        for j in range(0, m):
            min_dist = L**3
            index_min = 0

            for k in range(0, N):
                if(dist[i,k] < min_dist and not(dist[i,k] == 0.)):
                    min_dist = dist[i,k]
                    index_min = k

            temp_index[j] = index_min
            dist[i,index_min] = L**3

        close_list[i,:] = temp_index

    return close_list

@jit(nopython=True)
def fv(X,Y,dx,dy,r2,i,close_list,m,R,L,N,U,Nlist,append):
    """JV: Strongly based on the fv() function in PhySystem (inside physystem.py) but adapted to this scenario, see fv() in physystem.py for a good detailed explanation."""
#    L = self.L/self.R #JV: We are working in reduced units!
#    N = self.n**2

#    MX, MXT = np.meshgrid(X,X,copy=False)
#    MY, MYT = np.meshgrid(Y,Y,copy=False)
#    dx = MXT - MX
#    dx = dx
#    dy = MYT - MY
#    dy = dy
#
#    r2 = np.square(dx)+np.square(dy)

    dUx = 0.
    dUy = 0.
    utot = np.zeros((N))
    f = np.zeros((N,2))

    #JV: Now we do include this block of code outside of this function, so we don't have problems with numba

#    if(np.round(self.time,1)%0.3 == 0): #JV: every certain amount of steps we update the list
#        self.close_list = close_particles_list(r2,self.Nlist,N,L)
##            if(len(self.U) > 1 and len(self.K) > 1 and len(self.T) > 1):
##                print("Total energy: ", self.U[len(self.U)-1]+self.K[len(self.K)-1])
##                print("Temperature: ", self.T[len(self.T)-1])

    for j in range(0,N):
        dUx = 0.
        dUy = 0.
        u = 0.
        u_2 = np.zeros((Nlist*N))

        #JV: we now calculate the force with only the sqrt(N) closest particles
        for k in range(0,Nlist):
            c = int(close_list[j][k])

            if((r2[j,c] < 4*max(R[j],R[c])) and (r2[j,c] > 10**(-2))):
                #JV: We put self.r in the arguments because we want the ratius of both particles in reduced units
                dUx = dUx + dLJverlet(dx[j,c],r2[j,c],R[j],R[c])
                dUy = dUy + dLJverlet(dy[j,c],r2[j,c],R[j],R[c])

            if (append == True):
                if((r2[j,c] < 4*max(R[j],R[c])) and (r2[j,c] > 10**(-2))):
                    u_2[j+k] = LJverlet(r2[j,c],R[c],R[j])
                else:
                    u_2[j+k] = 0

        if(append == True):
            utot[j] = u

        f[j,:] = f[j,:]+np.array([dUx,dUy])

    if (append == True):
        U = np.sum(utot)

    return f

@jit(nopython=True)
def dLJverlet(x,r2,R1,R2):
    """JV: This function too is based on dLJverlet() in PhySystem (the class inside physystem.py) but adapted. Go there to get a fully detailed explanation."""
    rc = (2**(1/6))*((R1+R2)/(2))
    sig_int = (R1+R2)/(2)

    if((r2**(1/2))>rc):
        value = 0
    else:
        value = ((48.*x)/(r2))*(((((sig_int**2)*1.)/r2)**6) - ((((sig_int**2)*0.5)/r2)**3))

    return value

@jit(nopython=True)
def LJverlet(r2,R1,R2):
    """JV: This function too is based on LJverlet() in PhySystem (the class inside physystem.py) but adapted. Go there to get a fully detailed explanation."""
    rc = (2**(1/6))*((R1+R2)/(2))
    sig_int = (R1+R2)/(2) #JV: This is the sigma of the interaction (in the system units)

    if((r2**(1/2))>rc):
        value = 0
    else:
        value = 4*(((((sig_int**2)*1.)/r2)**6) - ((((sig_int**2)*1.)/r2)**3)) + 1

    return value

@jit(nopython=True)
def vel_verlet(t,dt,r0,v0,a0,dx,dy,r2,close_list,m,r,R,L,N,U,Nlist,append,key_wall,key_x_val,key_y_val,wall_h,wall_w,wallpos,wallwidth,holesize,n_holes,bouncing_key,bouncing_static,w,h,b,scale):
    """JV: Like the previous function, this function is strongly based too on the vel_verlet() function in PhySystem (inside physystem.py) but adapted to
     this scenario, see vel_verlet() in physystem.py for a good detailed explanation."""

    """In this function --> r: radius of the particle, R: radius of the Argon gas (in Angstroms)"""
    r1 = r0 + v0*dt + 0.5*a0*dt**2 #JV: We calculate x(t+dt)
    a1 = (1/m)*np.transpose(fv(r1[0,:],r1[1,:],dx,dy,r2,t/dt,close_list,m,r,L,N,U,Nlist,append)) #JV: From x(t+dt) we get a(t+dt)
    v1 = v0 + 0.5*(a0+a1)*dt #JV: From the a(t+dt) and a(t) we get v(t+dt)

    if(key_wall == False):
        #JV: Border conditions, elastic collision. (The "+1" is because 1 is the radius of the ball, in the reduced units that we calculate this part)
        v1[0,:] = np.where((np.abs(r1[0,:])+r/2)**2 > (0.49*L)**2,-v1[0,:],v1[0,:])
        v1[1,:] = np.where((np.abs(r1[1,:])+r/2)**2 > (0.49*L)**2,-v1[1,:],v1[1,:])

    elif(key_wall == True):
        bouncing_limits_x = np.where((np.abs(r1[0,:])+r/2)**2 > (0.49*L)**2,True,False)
        bouncing_limits_y = np.where((np.abs(r1[1,:])+r/2)**2 > (0.49*L)**2,True,False)

        #JV: Check bounce for the keyboard-controlled-walls
        wallpos_x = (key_x_val/(R)-w/(2*scale*R))
        wallpos_y = (key_y_val/(R)-h/(2*scale*R))
        wall_height = wall_h/R
        wall_width = wall_w/R

        bounce_left = np.where(r1[0,:]+r > (wallpos_x-wall_width/2),True,False)
        bounce_right = np.where(r1[0,:]-r < (wallpos_x+wall_width/2),True,False)
        is_leftside = np.where(r1[0,:] < wallpos_x, True, False)
        is_inside = np.where(np.logical_and(r1[1,:]-1*r < wallpos_y+wall_height/2, r1[1,:]+1*r > wallpos_y-wall_height/2), True, False)

        #JV: Check bounce for the static walls
        staticpos_x = (wallpos/(R)-w/(2*scale*R))
        static_width = wallwidth/R
        static_hole_size = holesize/R

        static_bounce_left = np.where(r1[0,:]+r > (staticpos_x-static_width/2),True,False)
        static_bounce_right = np.where(r1[0,:]-r < (staticpos_x+static_width/2),True,False)
        static_is_leftside = np.where(r1[0,:] < staticpos_x, True, False)
        static_is_inhole = np.zeros((N,n_holes))

        for i in range (n_holes):
            #JV: In static_is_inhole, we have an array of N*n_holes size, with 1 when a particles is "inside" the hole (in the y-coordinate)
            bottom = (i*h/(n_holes) - (h/(n_holes) - holesize*scale)/2)/(scale*R) - h/(2*scale*R)
            top = bottom + (h/(n_holes) - holesize*scale)/(scale*R)

            bottom_2 = ((i+1)*h/(n_holes) - (h/(n_holes) - holesize*scale)/2)/(scale*R) - h/(2*scale*R)

            static_is_inhole[:,i] = np.where(np.logical_and(r1[1,:]+0.5*r < bottom_2, r1[1,:]-0.5*r > top), True, False)
#            print(static_is_inhole)

        for i in range (N):
#                for j in range (self.n_holes):
#                    if(static_is_inhole[i,j] == True):
#                        print(i," in hole ",static_is_inhole[i,j])

            if(bouncing_limits_x[i]):
                v1[0,i] = -v1[0,i]
            elif(bouncing_limits_y[i]):
                v1[1,i] = -v1[1,i]
            else:
                #JV: Conditions for the keyboard wall
                if(is_leftside[i] and bounce_left[i] and is_inside[i] and bouncing_key[i] == 0):
                    v1[0,i] = -v1[0,i]
                elif(not(is_leftside[i]) and bounce_right[i] and is_inside[i] and bouncing_key[i] == 0):
                    v1[0,i] = -v1[0,i]
                #JV: This additional condition (bouncing_key[i] == 0) is because we want to avoid particles entering in a loop of conditions when bouncing and
                # making them "get stuck" in the middle of the wall, so now when it bounces it has to wait 2 more time steps to be able to bounce again
                if(is_leftside[i] and bounce_left[i] and is_inside[i]):
                    bouncing_key[i] += 1
                elif(not(is_leftside[i]) and bounce_right[i] and is_inside[i]):
                    bouncing_key[i] += 1
                else:
                    if (bouncing_key[i] != 0):
                        bouncing_key[i] -= 1

                #JV: Conditions for the static wall
                if(static_is_leftside[i] and static_bounce_left[i] and not(static_is_inhole[i,:].any()) and bouncing_static[i] == 0):
                    v1[0,i] = -v1[0,i]
                elif(not(static_is_leftside[i]) and static_bounce_right[i] and not(static_is_inhole[i,:].any()) and bouncing_static[i] == 0):
                    v1[0,i] = -v1[0,i]
                #JV: This additional condition is because we want to avoid particles entering in a loop of conditions when bouncing and
                # making them "get stuck" in the middle of the wall, so now when it bounces it has to wait 2 more time steps to be able to bounce again
                if(static_is_leftside[i] and static_bounce_left[i] and not(static_is_inhole[i,:].any())):
                    bouncing_static[i] += 1
                elif(not(static_is_leftside[i]) and static_bounce_right[i] and not(static_is_inhole[i,:].any())):
                    bouncing_static[i] += 1
                else:
                    if (bouncing_static[i] != 0):
                        bouncing_static[i] -= 1

    return r1[0,:],r1[1,:],v1[0,:],v1[1,:],a1

"""
JV: In this class we will try to make a "live" computation, so we don't need to wait to compute to start playing. Only needs
 physystem.py because we will use the particle() class that is defined there, but all the math is done here.

 UPDATE: As it's not "playable" as we wanted to, this section does not appear at the final program accessible by the user.
 It is still interesting, but needs more development.

 IF YOU WANT TO SEE THIS SECTION IN THE PROGRAM, UNCOMMENT THE BLOCK OF CODE IN "particlesinabox.kv" between the lines 44-57
"""
class GameScreen(Screen):
    charge = 1.

    #JV: Energy (Sub)Plot
    enplot = Figure()
    enplotax = enplot.add_subplot(111, xlabel='t', ylabel = 'Energy')
    enplotax.set_xlim([0,60]) #JV: This initial value should change if we change the total time of computation
    enplotax.set_ylim([0,25])
    enplot.subplots_adjust(0.125,0.19,0.9,0.9)
    enplotax.yaxis.labelpad = 10
    enplotax.xaxis.labelpad = -0.5
    enplotcanvas = FigureCanvasKivyAgg(enplot)

    particles = np.array([])

    def __init__(self, **kwargs):
        super(GameScreen, self).__init__(**kwargs)

    def g_pseudo_init(self):
        #JV: This is the intial positon of the wall controlled by the user until the user brings the wall inside the canvas
        self.key_x_val = 100
        self.key_y_val = 100

        #JV: We call this functions so we can get the keyboard information as input
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        self.time = 0.
        #JV: Here you can modify the time step, no time of computation here because it will keep playing until we stop it
        self.dt = 0.003 #JV: Because the time of computation is a number that ends with 0 or 5 (we can change this in the kivy file), we
        # need a dt that is multiple of 5 (0,01;0,015;...), if we don't do this we get some strange errors (...)

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.previewtimer = Clock.schedule_interval(self.preview,0.04) #JV: Will call the preview() funciton every 0.04 seconds
        self.previewlist = []
        self.progress = 0.
        self.n = 6 #JV: Modify this value if at the start you want to show more than one particle in the simulation (as a default value)
        self.temp = 10 #JV: Initial temperature of the particles

        self.mass = 1 #JV: change this if you want to change the initial mass

        #JV: Here you can modify the units of the simulation
        self.V0 = 0.01 #eV
        self.R = 3.405 #A
        self.L = 200. #A
        self.M = 0.04 #kg/mol

        self.draw_energy = False #JV: This is a switch variable that if True, draws (and also calculates) the energy of the simulation

        self.U = [] #JV: Here we will save the value of the potential energy in every step of time
        self.U_val = 0.
        self.V = 0. #JV: Here we will save the value of the root of the square of the two components of the velocity
        #JV: For the next two we use lists, go to line 620 of "physystem.py" to a brief explanation of why we do this
        self.T = [] #JV: Here we save the value of the temperature of the system in every step of time
        self.K = [] #JV: Here we save the value of the kinetic energy of the system in every step of time
        self.t = []

        if (self.draw_energy == True):
            #JV: Initialization of the plots
            self.enplotbox.add_widget(self.enplotcanvas)

        #JV:This variable defines the number of close particles that will be stored in the list (go to (physystem.py) close_particles_list() for more info)
        self.Nlist = int(1.4*(self.n))

        #JV: Now the variables for the static walls that divide the simulation
        self.walls = True #JV: Switch variable, if True there will be the static walls
        self.n_holes = 4 #JV: Number of holes in these walls
        self.holesize = 20 #JV: Length of the hole
        self.wallwidth = 2 #JV: Width of the wall
        self.wallpos = self.key_x_val #JV: Position of the wall, it goes where the keyboard-controlled-wall goes, if you want to change this you can do it here

        self.key_wall = True #JV: This is a switch-type variable, if True, will draw the Rectangle that will make the particles bounce when touching
        if(self.key_wall == True):
            #JV: We create a list that will be useful for the bouncing wall (both static walls and the keyboard controlled one), that will help us
            # in the border conditions of the wall, see in vel_verlet()
            self.bouncing_key = np.zeros(self.n**2)
            self.bouncing_static = np.zeros(self.n**2)

        #JV: Here we initialize the intial values of the widht and the height of the wall that is controlled by the user
        self.wall_w = 3
        self.wall_h = 12

        self.add_particle_list(False)

    def preview(self,interval):
        """JV: This function is called every frame and draws the particles if we are running the game."""
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L

        if(self.running == False and self.paused == False):
            self.plotbox.canvas.clear()
            with self.plotbox.canvas:
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

            #JV: We draw the static walls
            if(self.walls == True):
                with self.plotbox.canvas:
                    Color(0.37,0.01,0.95)
                    for i in range (self.n_holes+1):
                        Rectangle(pos=(self.wallpos*scale-self.wallwidth*scale/2, i*h/self.n_holes - (h/self.n_holes - self.holesize*scale)/2), size = (self.wallwidth*scale,h/self.n_holes - self.holesize*scale))

            #JV: We draw the keyboard-controlled wall
            if(self.key_wall == True):
                with self.plotbox.canvas:
                    Color(0.0,0.0,1.0)
                    Rectangle(pos=(self.key_x_val*scale-self.wall_w*scale/2,self.key_y_val*scale-self.wall_h*scale/2),size=(self.wall_w*scale,self.wall_h*scale))

    def add_particle_list(self,inversion):
        """JV: Adds the particles calling the particle() class that is stored in physystem.py"""
        self.stop() #I stop the simultion to avoid crashes

        self.reset_particle_list();

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L

        #JV: We now locate the initial positions of these particles
        self.X,self.Y = np.linspace(-self.L/2*0.9,self.wallpos-w/(scale*2)-self.L*0.05,self.n),np.linspace(-self.L/2*0.9,self.L/2*0.9,self.n)

        temp = self.temp
        theta = np.random.ranf(self.n**2)*2*np.pi
        self.Vx,self.Vy = 0.5*np.cos(theta),0.5*np.sin(theta)

        vcm = np.array([np.sum(self.Vx),np.sum(self.Vy )])/self.n**2
        kin = np.sum(self.Vx**2+self.Vy **2)/(self.n**2)

        if(self.n == 1): #JV: To avoid problems, if we only have one particle it will not obey that the velocity of the center of mass is 0
            self.Vx = self.Vx*np.sqrt(2*temp/kin)
            self.Vy = self.Vy*np.sqrt(2*temp/kin)
        else:
            self.Vx = (self.Vx-vcm[0])*np.sqrt(2*temp/kin)
            self.Vy = (self.Vy-vcm[1])*np.sqrt(2*temp/kin)

        k = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                #JV: In "particles" we have the positions and velocities in reduced units (the velocities are already transformed,
                # but for the positions we need to include the scale factor)
                self.particles = np.append(self.particles,particle(self.mass,self.charge,self.R/self.R,np.array([self.X[i],self.Y[j]])/self.R,np.array([self.Vx[k],self.Vy[k]]),2))

                #JV: In this new array we will have the positions and velocities in the physical units (Angstrom,...)
                self.previewlist.append([self.X[i],self.Y[j],self.Vx[k]*self.R,self.Vy[k]*self.R])

                k += 1

        #JV: We create a PhySystem class by passing the array of particles and the physical units of the simulation as arguments
        #JV: self.wallpos, self.holesize, self.wallwidth are in Angtroms, so we need to divide it by self.R to have it in reduced units (the units we work in physystem)

        self.s = PhySystem(self.particles,[self.V0,self.R,self.L/self.R,"In a box","Random Lattice",self.n,0,0,0,0])
        #JV: We redifine the self.X, self.Y so we get an n*n array, so we can iterate through each particle
        self.X = np.vectorize(lambda i: i.r[0])(self.particles)
        self.Y = np.vectorize(lambda i: i.r[1])(self.particles)
        self.r = np.vectorize(lambda i: i.R)(self.particles)

    def stop(self):
        self.pause()
        self.paused = False
        self.time = 0

    def pause(self):
        if(self.running==True):
            self.paused = True
            self.timer.cancel()
            self.timer2.cancel()
            self.running = False
        else:
            pass

    def reset_particle_list(self):
        #Empties particle list
        self.stop()
        self.particles = np.array([])
        self.previewlist = []

    def play_button(self):
        """JV: Function that is called when pressing the play button on the interface, it starts the "game" """
        if(self.running==False):
            self.timer = Clock.schedule_interval(self.step_animate,0.01)
            self.running = True
            self.paused = False
        elif(self.running==True):
            pass

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':
            if(self.key_y_val + self.wall_h/2.0 < self.L):
                self.key_y_val += self.L/100
        if keycode[1] == 's':
            if(self.key_y_val - self.wall_h/2.0 > 0):
                self.key_y_val -= self.L/100
        return True

    def step_animate(self,interval):
        """JV: Calculates a step of vel_verlet and then draws it. Keeps doing this until we stop it from the interface"""

        #JV: TODO: Maybe make this global so we don't need to allocate each frame...
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        self.plotbox.canvas.clear()

        N = self.particles.size

        #JV: Now we calculate the step of vel_verlet and we store the information we get in self.X[j], self.Y[j], self.Vx[j], self.Vy[j]

        #JV: If it is the first step we have to "manually" call some functions in order to obtain the initial acceleration of the particles
        if(self.time == 0):
            #JV: We get the values from particles because we need all the values in reduced units, as they are saved in particles
            self.X = np.vectorize(lambda i: i.r[0])(self.particles)
            self.Y = np.vectorize(lambda i: i.r[1])(self.particles)
            self.Vx = np.vectorize(lambda i: i.v[0])(self.particles)
            self.Vy = np.vectorize(lambda i: i.v[1])(self.particles)
            #JV: Now the values of the mass and radius in reduced units
            self.m = np.vectorize(lambda i: i.m)(self.particles)
            self.r = np.vectorize(lambda i: i.R)(self.particles)

            MX, MXT = np.meshgrid(self.X[:],self.X[:])
            MY, MYT = np.meshgrid(self.Y[:],self.Y[:])

            dx = MXT - MX
            dx = dx

            dy = MYT - MY
            dy = dy

            r2 = np.square(dx)+np.square(dy)

            self.close_list = close_particles_list(r2,self.Nlist,N,self.L) #JV: We calculate the matrix that contains in every row the indexs of the m closest particles

            self.X0 = self.X
            self.Y0 = self.Y
            self.VX0 = self.Vx
            self.VY0 = self.Vy

            self.a0 = (1/self.m)*np.transpose(fv(self.X0[:],self.Y0[:],dx,dy,r2,self.time/self.dt,self.close_list,self.m,self.r,self.L,N,self.U_val,self.Nlist,self.draw_energy))

        #JV: call velocityverlet to compute the next position
        MX, MXT = np.meshgrid(self.X0[:],self.X0[:],copy=False)
        MY, MYT = np.meshgrid(self.Y0[:],self.Y0[:],copy=False)
        dx = MXT - MX
        dx = dx

        dy = MYT - MY
        dy = dy

        r2 = np.square(dx)+np.square(dy)

        if(np.round((self.time/self.dt*self.dt)%0.3,2) == 0): #JV: every certain amount of steps we update the list
            self.close_list = close_particles_list(r2,self.Nlist,N,self.L) #JV: matrix that contains in every row the indexs of the m closest particles

        self.X,self.Y,self.Vx,self.Vy,self.a1 = vel_verlet(self.time,self.dt,np.array([self.X0,self.Y0]),np.array([self.VX0,self.VY0]),self.a0,dx,dy,r2,self.close_list,self.m,self.r,self.R,self.L/self.R,N,self.U_val,self.Nlist,self.draw_energy,self.key_wall,self.key_x_val,self.key_y_val,self.wall_h,self.wall_w,self.wallpos,self.wallwidth,self.holesize,self.n_holes,self.bouncing_key,self.bouncing_static,w,h,b,scale)

        #JV: We keep track of this step in time:
        self.time += self.dt

        self.a0 = self.a1

        #Redefine and repeat
        self.X0,self.Y0 = self.X,self.Y
        self.VX0,self.VY0 = self.Vx,self.Vy

        if (self.draw_energy == True):
            #JV: We calculate now self.V, self.T, self.K
            self.U.append(self.U_val)
            Ki = self.m*(self.Vx**2 + self.Vy**2)/2.
            self.K.append(np.sum(Ki))
            self.V = np.sqrt((self.Vx**2 + self.Vy**2))
            self.T.append(np.sum(self.V[int(self.n/2)]**2)/(self.particles.size*2 - 2))

        #JV: Now we draw the particles
        with self.plotbox.canvas:
            for j in range(0,N):
                Color(1.0,0.0,0.0)
                Ellipse(pos=((self.X[j])*scale*self.R+w/2.-self.r[j]*scale/2.,(self.Y[j])*scale*self.R+h/2.-self.r[j]*scale/2.),size=(self.r[j]*self.R*scale,self.r[j]*self.R*scale))

         #JV: We draw the static walls
        if(self.walls == True):
            with self.plotbox.canvas:
                Color(0.37,0.01,0.95)
                for i in range (self.n_holes+1):
                    Rectangle(pos=(self.wallpos*scale-self.wallwidth*scale/2, i*h/self.n_holes - (h/self.n_holes - self.holesize*scale)/2), size = (self.wallwidth*scale,h/self.n_holes - self.holesize*scale))

        #JV: We draw the keyboard-controlled wall
        if(self.key_wall == True):
            with self.plotbox.canvas:
                Color(0.0,0.0,1.0)
                Rectangle(pos=(self.key_x_val*scale-self.wall_w*scale/2, self.key_y_val*scale-self.wall_h*scale/2), size = (self.wall_w*scale,self.wall_h*scale))

        i = int(self.time/self.dt)

        if (self.draw_energy):
            #JV: We draw the energy:
            self.t.append(self.time)

            self.enplotax.clear()
            self.enplotax.set_xlabel('t')
            self.enplotax.set_ylabel('Energy')

            if(i > 2):
                self.enplotax.set_xlim([0,self.t[i-1]])
                self.enplotax.set_ylim([0,(np.array(self.K[0:i-1]).max()+np.array(self.U[0:i-1]).max())+np.uint(np.array(self.K[0:i-1]).max()+np.array(self.U[0:i-1]).max())/40])

                self.enplotax.plot(self.t[0:i-1],self.K[0:i-1],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.t[0:i-1],self.U[0:i-1],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.t[0:i-1],self.K[0:i-1]+self.U[0:i-1],'g-',label = 'Total Energy')

                self.enplotax.legend(loc=7)

                self.enplotcanvas.draw()

    def transition_GM(self):
        """JV: Screen transition: from 'menu' to 'game' """
        self.stop()
        self.manager.transition = FadeTransition()
        self.manager.current = 'menu'

    def update_pos(self,touch):
        """JV: This function is called when you click the screen"""
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        #JV: We divide by scale so we get an independent-size-of-the-window coordinate
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale
        touch = self.is_touching(x/self.R,y/self.R) #JV: We pass the coordinate arguments in reduced units so it has the same units as the particles
        if(touch != None):
            print("x: ",np.round(x,2),"y: ",np.round(y,2)," -> Bola ",touch," tocada")

    def is_touching(self,x,y):
        """JV: Function that when passing x,y returns the id of the particle if there is one, returns None if there isn't a particle in this location."""
        for i in range (self.n**2):
            if(x < self.X[i]+self.r[i] and x > self.X[i]-self.r[i] and y < self.Y[i]+self.r[i] and y > self.Y[i]-1):
                return i
        return None


class DemoScreen(Screen):
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
    enplotax.set_xlim([0,1])
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

    def transition_DM(self):
        """JV: Screen transition: from 'demo' to 'menu'"""
        self.stop()
        self.manager.transition = FadeTransition()
        self.manager.current = 'menu'


    def __init__(self, **kwargs):
        super(DemoScreen, self).__init__(**kwargs)

    def d_pseudo_init(self):
        """
        JV: Here we initialize some functions and some initial values. Some of this values can be changed inside the simulation
         on the "Advanced Settings" menu. (Go to advanced_settings() for more info)
        """
        self.n_demo = 0 #JV: 0 is the first demo that is loaded, this variable saves the demo that we are in
        self.total_demos = len(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"saves"))) #JV: This stores the number of demos that we have loaded

        if(self.total_demos == 0):
            print("You don't have any demo loaded! You need to have at least one to go to the Demo page.")
        else:
            self.simulationlabel.text = "Demo " + str(self.n_demo + 1) #JV: We count starting from 1 because humans prefer it this way
            self.simulationname.text = os.path.splitext(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"saves"))[self.n_demo])[0] #JV: We show the title of the simulation (without the extension)
        self.time = 0.
        self.T = 0
        self.dt = 0.01 #JV: Because the time of computation is a number that ends with 0 or 5 (we can change this in the kivy file), we
        # need a dt that is multiple of 5 (0,01;0,015;...), if we don't do this we get some strange errors (...)

        #Initialization of the speed button
        self.speedindex = 3
        self.change_speed()

        #JV: Initial values of the temperatures of the particles (temp1) and the second type of particles for the subsystems menu
        self.temp1 = 3
        self.temp2 = 2

        self.compact = 1 #JV: If this variable is increased, the particles will be close to each other, instead of ocupying the whole space they will get into a "corner"

        #JV: This number is the steps it takes in the animation, to draws the plots. If it's 1, it will update each step of time, etc.
        self.update_plots = 5
        #JV: The momentum plot can be slower than the other plots, so we it is possible to only update it every certain steps inside the plotting() function. If 1 it will update like the other plots.
        self.update_momentum = 3
        self.histtype = "stepfilled" #JV: The type of histogram. {'bar', 'barstacked', 'step', 'stepfilled'}. "bar" seems to be the most appealing visually, while "step" runs much faster
        self.counter = 0 #JV: The counter of this "plotting" steps
        #JV: This counters store the last plotting point, so we can print the ones that are missing (one for the energy plot and the other for the entropy)
        self.prev_i_en = 0
        self.prev_i_ex = 0
        self.simulation = False #JV: This bool will be true when we will have loaded a simulation

        self.si_units = False #JV:If false, we start the simulation in reduced units; if true, in SI Units

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)#Always, runs, shows previews. Calls the preview() function every 0.04 seconds
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
        self.wallwidth = 1 #JV: We define the initial value of the width of the wall

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
        self.submenu_list_str = ["Random Lattice","Subsystems","Brownian"] #JV: And a list with the names of each submenu

        #JV: We do the same for the menus. Update this if you add more menus.
        self.menu_list_str = ["In a box","Free!","Walls"] #JV: And a list with the names of each menu

        #JV: We make this so the simulation starts with one particle instead of 0, that could lead to some errors
        self.load_demo()

    def prev_simulation(self):
        """JV: This function is called by the left arrow on the kivy file, so here we update the variable that counts
         the simulation we are in and the call the function self.load_demo(), that will update this changes."""
        if(self.n_demo != 0):
            self.n_demo -= 1
        else:
            self.n_demo = self.total_demos - 1

        #JV: Update the changes
        self.simulationlabel.text = "Demo " + str(self.n_demo + 1)
        self.simulationname.text = os.path.splitext(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"saves"))[self.n_demo])[0]
        self.load_demo()


    def next_simulation(self):
        """JV: Like prev_simulation() but we go to the next demo instead of the previous."""
        if(self.n_demo != (self.total_demos - 1)):
            self.n_demo += 1
        else:
            self.n_demo = 0

        #JV: Update the changes
        self.simulationlabel.text = "Demo " + str(self.n_demo + 1)
        self.simulationname.text = os.path.splitext(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"saves"))[self.n_demo])[0]
        self.load_demo()


    def update_pos(self,touch):
        """This function updates the position parameters when you click the screen"""
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale


    def play(self):
        if(self.running==False):
            time.sleep(0.5)
            self.timer = Clock.schedule_interval(self.animate,0.04)
            self.timer2 = Clock.schedule_interval(self.plotting, 0.04*self.update_plots)
            self.running = True
            self.paused = False
        elif(self.running==True):
            pass


    def pause(self):
        if(self.running==True):
            self.paused = True
            self.timer.cancel()
            self.timer2.cancel()
            self.running = False
        else:
            pass

    def stop(self):
        self.pause()
        self.paused = False
        self.time = 0
        self.obj2.clear()
        self.plotbox.canvas.clear()
        self.progressbar.value = 0 #JV: We set the progress bar to 0
        self.setplotlimits(simulation = self.simulation) #JV: We don't know if we have loaded a simulation yet
        #JV: Restart the counters
        self.acucounter = 0
        self.prev_i_en = 0
        self.prev_i_ex = 0


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


    def load_demo(self):
        """JV: This function loads the simulation that we want."""
        self.stop()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"saves")
        demos = os.listdir(path)

        if(self.total_demos != 0):
            with open(os.path.join(path,demos[self.n_demo]),'rb') as file:
                savedata = pickle.load(file)

            name = demos[self.n_demo]

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
            self.temp1 = savedata[13]
            self.temp2 = savedata[14]
            self.compact = savedata[15]
            self.si_units = savedata[16]
            if(len(savedata) > 17):
                self.demo_info_img = savedata[17] #JV: This variable stores the name (and relative path) to the info image. The simulations created by the user will only have a gray image.
            else:
                self.demo_info_img = "icons/gray_demo.png"

            self.infolabel.source = self.demo_info_img

            #JV: We change the Entropy box if we are in the Brownian submenu
            if(self.our_submenu ==  "Brownian"):
                self.extraplottab.text = "<|x(t)-x(0)|^2>"
            else:
                self.extraplottab.text = "Entropy"

    #        self.timeslider.value = self.T
            self.n = int(np.sqrt(self.s.particles.size))
            print("")
            print("")
            print('Loaded simulation {} with computation'.format(name))

            last = len(self.s.U)-1 #JV: This variable stores the index of the last variable in this arrays, so we can access it easier
            print("Initial energy: ",self.s.K[0]+self.s.U[0],"Final energy: ",self.s.U[last]+self.s.K[last])
            print("Relative increment of energy: ", (abs((self.s.K[0]+self.s.U[0])-(self.s.U[last]+self.s.K[last]))/(self.s.K[0]+self.s.U[0]))*100,"%")

            self.simulation = True #JV: This bool is True if we have a simulation calculated/loaded
            self.setplotlimits(simulation = self.simulation) #JV: Reset and adapt the matplotlib plots


    def loadpopup(self):
        content = loadwindow(load = self.load, cancel = self.dismiss_popup)
        self._popup = Popup(title='Load File', content = content, size_hint=(1,1))
        self._popup.open()


    def dismiss_popup(self):
        self._popup.dismiss()


    def plotting(self,interval):
        """JV: This function is the responsible of updating the plots"""
        i = int(self.time/self.dt)
        pause = 0.001
        n_bins = 15 #JV: The number of bins that we will be plotting in the histogram

        if(self.si_units):
            delta = 5*2.16/self.dt
        else:
            delta = 5./self.dt

        if(self.our_submenu == 'Random Lattice'):
            if(self.plotmenu.current_tab.text == 'Energy'): #JV: instantaneous energy graphic
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.U[self.prev_i_en:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i]+self.s.U[self.prev_i_en:i],'g-',label = 'Total Energy')
                self.enplotcanvas.draw()
                self.prev_i_en = i
                plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.counter % self.update_momentum == 0):
                    self.histax.cla()
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 0.5*len(self.s.V[i,:])
                    self.histax.set_ylim([0,y_lim])

                    if(self.si_units):
                        self.histax.set_xlabel('v (m/s)')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+50)/n_bins
                        hist,bins = np.histogram(self.s.V[i,:], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        center = (bins[:-1] + bins[1:]) / 2
                        self.histax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.histax.set_xlabel('v')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+1)/n_bins
                        hist,bins = np.histogram(self.s.V[i,:], bins=np.arange(0,self.s.V.max()+1,bin_width))
                        center = (bins[:-1] + bins[1:]) / 2
                        self.histax.bar(center, hist, align='center', width=bin_width*0.8)

                    self.s.MB[i,:] = self.s.MB[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.
                    if(self.si_units):
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3))+" K", fontsize=15, color = "red", alpha = 0.85)
                    else:
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(self.vsplot,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                if(self.si_units):
                    wait_time = 30. * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
                else:
                    wait_time = 30.
                if(self.time > wait_time):
                    self.acuhistax.cla()
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 1
                    self.acuhistax.set_ylim([0,y_lim])

                    if(self.si_units):
                        self.acuhistax.set_xlabel('v (m/s)')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)

                    self.acuhistcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Entropy'):
                self.extraplotax.plot(self.tplot[self.prev_i_ex:i],self.s.entropy[self.prev_i_ex:i],'g-')
                self.extraplotcanvas.draw()
                self.prev_i_ex = i
                plt.pause(pause)

        elif(self.our_submenu == 'Subsystems'):
            #JV: Here we do the graphics for this submenu, check the comments from the previous submenu for more info
            #JV: We will draw the two subsystems in two diferent colors
            if(self.plotmenu.current_tab.text == 'Energy'):
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.U[self.prev_i_en:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i]+self.s.U[self.prev_i_en:i],'g-',label = 'Total Energy')
                self.enplotcanvas.draw()
                self.prev_i_en = i
                plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.counter % self.update_momentum == 0):
                    self.histax.cla()
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 0.3*len(self.s.V[i,:])
                    self.histax.set_ylim([0,y_lim])

                    if(self.si_units):
                        self.histax.set_xlabel('v (m/s)')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+50)/int(n_bins*0.7)
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.n1**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.n1**2:self.n1**2+self.n2**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
                        self.histax.bar(center2, hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16])
                    else:
                        self.histax.set_xlabel('v')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+1)/int(n_bins*0.7)
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.n1**2], bins=np.arange(0,self.s.V.max()+1,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.n1**2:self.n1**2+self.n2**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
                        self.histax.bar(center2, hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16])

                    self.s.MB[i,:] = self.s.MB[i,:] * y_lim #JV: The MB curves go from 0 to 1, so we need to adapt them to our plot.
                    self.s.MB1[i,:] = self.s.MB1[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.
                    self.s.MB2[i,:] = self.s.MB2[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.

                    #JV: We now plot the texts that show the temperature of each types of particles
                    if(self.si_units):
                        self.histax.text(self.vsplot[35],y_lim*0.8,"T1 = "+str(np.round(self.s.T1[i],decimals=3))+" K", fontsize = 12, color = "blue", alpha = 0.75)
                        self.histax.text(self.vsplot[70],y_lim*0.8,"T2 = "+str(np.round(self.s.T2[i],decimals=3))+" K", fontsize = 12, color = "green", alpha = 0.75)
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3))+" K", fontsize = 12, color = "red", alpha = 0.85)
                    else:
                        self.histax.text(self.vsplot[45],y_lim*0.8,"T1 = "+str(np.round(self.s.T1[i],decimals=3)), fontsize = 12, color = "blue", alpha = 0.75)
                        self.histax.text(self.vsplot[75],y_lim*0.8,"T2 = "+str(np.round(self.s.T2[i],decimals=3)), fontsize = 12, color = "green", alpha = 0.75)
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max(),"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize = 12, color = "red", alpha = 0.85)
                    self.histax.plot(self.vsplot,self.s.MB1[i,:],'b-', alpha = 0.5)
                    self.histax.plot(self.vsplot,self.s.MB2[i,:],'g-', alpha = 0.5)
                    self.histax.plot(self.vsplot,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                if(self.si_units):
                    wait_time = 30. * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
                else:
                    wait_time = 30.
                if(self.time > wait_time):
                    self.acuhistax.cla()
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 1
                    self.acuhistax.set_ylim([0,y_lim])
                    if(self.si_units):
                        self.acuhistax.set_xlabel('v (m/s)')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)

                    self.acuhistcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Entropy'):
                self.extraplotax.plot(self.tplot[self.prev_i_ex:i],self.s.entropy[self.prev_i_ex:i],'g-')
                self.extraplotcanvas.draw()
                self.prev_i_ex = i
                plt.pause(pause)

        elif(self.our_submenu == 'Brownian'):
            if(self.plotmenu.current_tab.text == 'Energy'):
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.U[self.prev_i_en:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(self.tplot[self.prev_i_en:i],self.s.K[self.prev_i_en:i]+self.s.U[self.prev_i_en:i],'g-',label = 'Total Energy')
                self.enplotcanvas.draw()
                self.prev_i_en = i
                plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                if(self.counter % self.update_momentum == 0):
                    self.histax.cla()
                    self.histax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 0.6*len(self.s.V[i,:])
                    self.histax.set_ylim([0,y_lim])

                    if(self.si_units):
                        self.histax.set_xlabel('v (m/s)')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+50)/n_bins
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.nsmall**2], bins=np.arange(0,self.s.V.max()+50,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.nsmall**2:self.nsmall**2+self.nbig**2], bins=1)
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center2, y_lim*hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16],alpha=0.8)
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])
                    else:
                        self.histax.set_xlabel('v')
                        self.histax.set_ylabel('Number of particles')
                        bin_width = (self.s.V.max()+1)/n_bins
                        hist1,bins1 = np.histogram(self.s.V[i,0:self.nsmall**2], bins=np.arange(0,self.s.V.max()+1,bin_width))
                        hist2,bins2 = np.histogram(self.s.V[i,self.nsmall**2:self.nsmall**2+self.nbig**2])
                        center1 = (bins1[:-1] + bins1[1:]) / 2
                        center2 = (bins2[:-1] + bins2[1:]) / 2
                        self.histax.bar(center2, y_lim*hist2, align='center', width=bin_width*0.8, color=[0.43,0.96,0.16],alpha=0.8)
                        self.histax.bar(center1, hist1, align='center', width=bin_width*0.8, color=[0.32,0.86,0.86])

                    self.s.MB[i,:] = self.s.MB[i,:] * y_lim #JV: The MB curve goes from 0 to 1, so we need to adapt it to our plot.
                    if(self.si_units):
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3))+" K", fontsize=15, color = "red", alpha = 0.85)
                    else:
                        self.histax.text(self.vsplot[np.argmax(self.s.MB[i,:])-int(len(self.vsplot)*0.2)],self.s.MB[i,:].max()*1.1,"T = "+str(np.round(self.s.T[i],decimals=3)), fontsize=15, color = "red", alpha = 0.85)
                    self.histax.plot(self.vsplot,self.s.MB[i,:],'r-')
                    self.histcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                if(self.si_units):
                    wait_time = 30. * 2.16 #JV: Go to change_units() to understand the "2.16" factor"
                else:
                    wait_time = 30.
                if(self.time > wait_time):
                    self.acuhistax.cla()
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    y_lim = 1
                    self.acuhistax.set_ylim([0,y_lim])

                    if(self.si_units):
                        self.acuhistax.set_xlabel('v (m/s)')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)
                    else:
                        self.acuhistax.set_xlabel('v')
                        self.acuhistax.set_ylabel('Number of particles relative')
                        bin_width = (self.s.V.max())/n_bins
                        hist,bins = np.histogram(self.s.Vacu[int((i-int((wait_time/(self.dt))))/delta)], bins=np.arange(0,self.s.V.max(),bin_width),density=True)
                        y_lim = 2.4*hist.max()
                        self.acuhistax.set_ylim([0,y_lim])
                        center = (bins[:-1] + bins[1:]) / 2
                        self.acuhistax.bar(center, hist, align='center', width=bin_width*0.8)

                    self.acuhistcanvas.draw()
                    plt.pause(pause)

            elif(self.plotmenu.current_tab.text == '<|x(t)-x(0)|^2>'):
                self.extraplotax.plot(self.tplot[self.prev_i_ex:i],self.s.X2[self.prev_i_ex:i],'g-')
                self.extraplotcanvas.draw()
                self.prev_i_ex = i
                plt.pause(pause)

        self.counter += 1

    def setplotlimits(self,simulation):
        """JV: This function updates the limits of the matplotlib plots. If there is a simulation loaded, it will adapt the plots limits to it."""

        #JV: Energy plot
        self.enplotax.cla()
        if(self.si_units): #JV: If true, then we will add the units on the labels
            self.enplotax.set_xlabel('t (·$10^{-12}$ s)')
            self.enplotax.set_ylabel('Energy (·$10^{-18}$ J)')
        else:
            self.enplotax.set_xlabel('t')
            self.enplotax.set_ylabel('Energy')
        self.enplot.subplots_adjust(0.125,0.19,0.9,0.9)
        self.enplotax.yaxis.labelpad = 10
        self.enplotax.xaxis.labelpad = -0.5

        #JV: Momentum plot
        self.histax.cla()
        if(self.si_units):
            self.histax.set_xlabel('v (m/s)')
            self.histax.set_ylabel('Number of particles')
        else:
            self.histax.set_xlabel('v')
            self.histax.set_ylabel('Number of particles')
        self.hist.subplots_adjust(0.125,0.19,0.9,0.9)
        self.histax.yaxis.labelpad = 10
        self.histax.xaxis.labelpad = -0.5

        #JV: Accumulated momentum
        self.acuhistax.cla()
        if(self.si_units):
            self.acuhistax.set_xlabel('v (m/s)')
            self.acuhistax.set_ylabel('Number of particles relative')
        else:
            self.acuhistax.set_xlabel('v')
            self.acuhistax.set_ylabel('Number of particles relative')
        self.acuhist.subplots_adjust(0.125,0.19,0.9,0.9)
        self.acuhistax.yaxis.labelpad = 10
        self.acuhistax.xaxis.labelpad = -0.5

        #JV: Entropy/
        self.extraplotax.cla()
        if(self.our_submenu == 'Brownian'):
            if(self.si_units):
                self.extraplotax.set_xlabel('t (·$10^{-12}$ s)')
                self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$ (·$10^{-20}$ $m^2$)')
            else:
                self.extraplotax.set_xlabel('t')
                self.extraplotax.set_ylabel(r'$\langle|x(t)-x(0)|^{2}\rangle$')
        else:
            if(self.si_units):
                self.extraplotax.set_xlabel('t (·$10^{-12}$ s)')
                self.extraplotax.set_ylabel('Entropy (·$10^{-18}$ J/K)')
            else:
                self.extraplotax.set_xlabel('t')
                self.extraplotax.set_ylabel('Entropy')

        self.extraplot.subplots_adjust(0.125,0.19,0.9,0.9)
        self.extraplotax.yaxis.labelpad = 10
        self.extraplotax.xaxis.labelpad = -0.5

        if(simulation):
            n = int(self.T/self.dt)
            self.tplot = np.arange(self.dt,self.T+10+self.dt,self.dt)
            self.vsplot = np.linspace(0,self.s.V.max()+0.5,100)

            #JV: Energy plot
            self.enplotax.set_xlim([0,self.T])
            #JV: We make the red line a little wider so we can see it and doesn't get hidden (in some cases) by the total energy
            self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])
            self.enplotax.plot(0,self.s.K[0],'r-',label = 'Kinetic Energy', linewidth = 2.2)
            self.enplotax.plot(0,self.s.U[0],'b-',label = 'Potential Energy')
            self.enplotax.plot(0,self.s.K[0]+self.s.U[0],'g-',label = 'Total Energy')
            self.enplotax.legend(loc=7)

            #JV: Momentum plot
            self.histax.set_xlim([0,self.s.V.max()+0.5])
            self.histax.set_ylim([0,np.ceil(self.s.MB.max())])

            #JV: Accumulated momentum
            self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])

            #JV: Entropy/
            if(self.our_submenu == 'Brownian'):
                self.extraplotax.set_xlim([0,self.T])
                self.extraplotax.set_ylim([0,self.s.X2.max()+self.s.X2.max()*0.05])
            else:
                self.extraplotax.set_xlim([0,self.T])
                self.extraplotax.set_ylim([self.s.entropy.min(),self.s.entropy.max()+self.s.entropy.max()*0.05])

        else:
            #JV: Energy plot
            self.enplotax.set_xlim([0,self.T])
            self.enplotax.set_ylim([0,25])

            #JV: Momentum plot
            self.histax.set_xlim([0,6])
            self.histax.set_ylim([0,4])

            #JV: Accumulated momentum
            self.acuhistax.set_xlim([0,6])
            self.acuhistax.set_ylim([0,1])

            #JV: Entropy/
            self.extraplotax.set_xlim([0,self.T])
            self.extraplotax.set_ylim([0,5])

        #JV: Now draw this changes in the plot
        self.enplotcanvas.draw()
        self.histcanvas.draw()
        self.acuhistcanvas.draw()
        self.extraplotcanvas.draw()

    def preview(self,interval):
        """Draws the previews of the particles when the animation is not running or before adding
        the preview of the lattice mode before adding is not programmed (mainly because it is a random process)"""

        if(self.running == False and self.paused == False):
            #JV: We can add conditions if we need it for different menus (like we do for the "walls" menu)
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

        elif(self.our_submenu == 'Brownian'):
            with self.plotbox.canvas:
                for j in range(0,(self.nsmall)**2+(self.nbig)**2):
                    if(j < self.nsmall**2):
                        Color(0.32,0.86,0.86)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                    else:
                        Color(0.43,0.96,0.16)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.Rbig*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.Rbig*scale/2.),size=(self.Rbig*scale,self.Rbig*scale))

                        #JV: These lines inside the condition make the trace of the big particle. We will only draw it if we are not in the "timeinversion" mode.
                        #JV: THIS FEATURE IS NOT ACTIVE, NEEDS UPDATE: The trace, as it is drawn an ellipse each frame, it stacks on the canvas an doesn't update when
                        # changing the size of the window, needs to change so each frame draws ALL the ellipses, don't know how this will affect at the FPS
#                        if(self.inversion == False):
#                            self.plotbox.canvas.add(self.obj2)
#                            self.obj2.add(Color(0.43,0.96,0.16))
#                            self.obj2.add(Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.,(self.s.Y[i,j])*scale*self.R+h/2.),size=(self.Rbig*scale/15,self.Rbig*scale/15)))


            self.time += interval*self.speed #Here is where speed accelerates animation
            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.

            self.acucounter += 1

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
            self.stop()

"""
JV: Manages and has access to the screens.
"""
class MyScreenManager(ScreenManager):

    def  __init__(self, **kwargs):
        """JV: Initiates the manager and the screens. Called whenever an instance of MyScreenManager is created."""
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('simulation').s_pseudo_init()
        self.get_screen('game').g_pseudo_init()
        self.get_screen('demo').d_pseudo_init()

"""
JV: Initial screen when the app is opened. This initial screen it's the menu that gives access to the other screens.
"""
class MenuScreen(Screen):

    def __init__ (self, **kwargs):
        """JV: Initiates the screen by calling Screen's __init__. Called from ScreenManager __init__."""
        super(MenuScreen, self).__init__(**kwargs)

    def transition_MG(self):
        """JV: Screen transition: from 'menu' to 'game' """
        GameScreen()
        self.manager.transition = FadeTransition()
        self.manager.current = 'game'

    def transition_MS(self):
        """JV: Screen transition: from 'menu' to 'simulation' """
        SimulationScreen()
        self.manager.transition = FadeTransition()
        self.manager.current = 'simulation'

    def transition_MD(self):
        """JV: Screen transition: from 'menu' to 'demo' """
        self.total_demos = len(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"saves")))
        if(self.total_demos == 0):
            self.warning_window()
        else:
            DemoScreen()
            self.manager.transition = FadeTransition()
            self.manager.current = 'demo'

    def warning_window(self):
        content = warningwindow(cancel = self.dismiss_popup)
        self._popup = Popup(title="Warning", content =content, size_hint = (0.6,0.5), auto_dismiss=False)
        self._popup.open()

    def dismiss_popup(self):
        #JV: Closes the Pop-Up
        self._popup.dismiss()

"""
JV: This is the app class, when it opens calls the ScreenManager
"""
class particlesinaboxApp(App):

    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    particlesinaboxApp().run()