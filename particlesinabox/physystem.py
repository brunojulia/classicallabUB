import numpy as np
from numba import jit
import sys

#JV: Truncated Lennard-Jones potentials (WCA), see LJverlet function for more explanation and documentation for the expressions.
#JV: We include both radius as parameters so we can generalize the function to interaction between diferent radius
@jit(nopython=True)
def dLJverlet(x,r2,R1,R2):
    """The derivative has the same form for x and y so only one is needed,
    this only changes when calling the interaction on the algotyhm,
    for all isotrope interactions this should still hold."""
    rc = (2**(1/6))*((R1+R2)/(2))
    sig_int = (R1+R2)/(2) #JV: This is the sigma of the interaction (in the system units). We don't need to divide by sigma because we are already working with reduced units

    #JV: Because we are working on reduced units (from the values of the Argon gas)
    # we want need to divide our radius by the radius of the Argon gas

    #JV: See LJverlet() for more explanation on the truncation
    if((r2**(1/2))>rc):
        value = 0
    else:
        value = ((48.*x)/(r2))*(((((sig_int**2)*1.)/r2)**6) - ((((sig_int**2)*0.5)/r2)**3))

    return value

@jit(nopython=True)
def LJverlet(r2,R1,R2):
    rc = (2**(1/6))*((R1+R2)/(2))
    sig_int = (R1+R2)/(2) #JV: This is the sigma of the interaction (in the system units)

    """JV: In doing this, we are actually taking the LJ potential truncated at the minimum potentail eneregy, at a distance
    r=(2**(1/6))*sigma, and we shift upward by the amount of minimum energy on the energy scale so the energy and the force are
    0 at or beyond the cutoff distance, so we get a continuous and only repulsive potential. This is actually called a WCA
    potential (Weeks-Chandler-Andersen potential) and it's actually what we need here, we don't actually need the
    atractive part of the LJ potential for this simulation."""
    if((r2**(1/2))>rc):
        value = 0
    else:
        value = 4*(((((sig_int**2)*1.)/r2)**6) - ((((sig_int**2)*1.)/r2)**3)) + 1

    return value

@jit(nopython=True)
def walls(r,param):
    """For saving on lines I have designed the walls function and the derivative in such a way
    that the same line can be used for the right-left and the top-down walls.
    This works thanks to the np.sign.
    The height/width of the wall is scaled to the size of the box so if L is modified
    you don't need to modify this. The parameter a is also escaled to the unit of lenght
    (radius of the particles)"""
    V = param[0]
    sig = param[1]
    L = param[2]

    a = 1/sig


    x0 = L/2.
    y0 = 0.
    V0 = 10000*V
    Rx = 0.01*L
    Ry = 0.6*L

    x = r[0] - x0*np.sign(r[0])
    y = r[1] - y0*np.sign(r[1])
    px = np.sqrt(x**2)
    py = np.sqrt(y**2)

    f1 = V0*(1/(1 + np.exp((px-Rx)/a)))*(1/(1 + np.exp((py-Ry)/a)))

    x0 = 0.
    y0 = L/2.
    V0 = 10000*V
    Rx = 0.6*L
    Ry = 0.01*L

    x = r[0] - x0*np.sign(r[0])
    y = r[1] - y0*np.sign(r[1])
    px = np.sqrt(x**2)
    py = np.sqrt(y**2)

    f2 = V0*(1/(1 + np.exp((px-Rx)/a)))*(1/(1 + np.exp((py-Ry)/a)))

    value = f1+f2
    return value

@jit(nopython=True)
def dwalls(r,param):
    """See walls function for more information, this is just the derivative."""
    V = param[0]
    sig = param[1]
    L = param[2]

    a = 1/sig

    x0 = L/2.
    y0 = 0.
    V0 = 10000*V
    Rx = 0.01*L
    Ry = 0.6*L

    x = r[0] - x0*np.sign(r[0])
    y = r[1] - y0*np.sign(r[1])


    px = np.sqrt(x**2)
    py = np.sqrt(y**2)
    try:
        f1 = -V0*((np.sign(x)*np.exp((px-Rx)/a))/(a*(np.exp((px-Rx)/a)+1)**2))*(1/(1 + np.exp((py-Ry)/a)))

        x0 = 0.
        y0 = L/2.
        V0 = 10000*V
        Rx = 0.6*L
        Ry = 0.01*L

        x = r[0] - x0*np.sign(r[0])
        y = r[1] - y0*np.sign(r[1])
        px = np.sqrt(x**2)
        py = np.sqrt(y**2)

        f2 = -V0*((np.sign(x)*np.exp((Rx+px)/a))/(a*(np.exp(Rx/a)+np.exp(px/a))**2))*(1/(1 + np.exp((py-Ry)/a)))
    except RuntimeWarning:
        f1 = 0.
        f2 = 0.
    except FloatingPointError:
        f1 = 0.
        f2 = 0.
    f = f1 + f2
    return f

@jit(nopython=True)
def close_particles_list(r2,m,N,L):
    """JV: This functions returns a list (a matrix) where each rows saves m indexs corresponding to the m
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
def fv(X,Y,dx,dy,r2,i,append,L,N,U,dt,close_list,Nlist,vel_verlet_on,R,menu,submenu,n1,grid,G,wallcount,X2):
    """fv(X,Y) represents the forces that act on all the particles at a particular time.
    It computes the matrix of forces using the positions given with X and Y which are
    the arrays of size N containing all the positions (coordinates X and Y).
    The resulting matrix, f is of shape (N,2) (it should be (2,N), see the verlet function)."""

    """JV: append is a boolean. If it's true, adds the energy to our list, if it isn't, it doesn't.
     We do that because in some cases we will call the algorithm more times than the actual step number (and
     we only want to sum the value T/dt times), this is needed in the velocity-Verlet algorithm, that we call the fv()
     function one more time than needed just to start the loop."""

#        L = self.param[2]
#
#        N = self.particles.size

    #For computing all the distances I use a trick with the meshgrid function,
    #see the documentation on how this works if you dont see it.

    """JV: X is an array that contains each position, mx is an nxn array that each column is the position of one particle (so it's a matrix
    that has n X rows) and mxt is the same but tranposed (so it's a matrix of n X columns)"""
#    MX, MXT = np.meshgrid(X,X,copy=False)
#    MY, MYT = np.meshgrid(Y,Y,copy=False)

    #JV: So dx is a nxn simetric array with 0 in the diagonal, and each position is the corresponding distance between the particles,
    # so the position [1,2] is the distance between partcle 1 and 2 (x1-x2), and so on
#    dx = MXT - MX
#    dx = dx
#
#    dy = MYT - MY
#    dy = dy
#
#    r2 = np.square(dx)+np.square(dy)

    if(menu == "Free!"):
    #JV: We do this to get the actual distance in the case of the "Free!" simulation
        dx_v2 = (np.abs(dx.copy())-1*L)
        r2_v2 = dx_v2**2+dy**2
        dx = np.where(r2 > r2_v2,dx_v2*np.sign(dx),dx)
        r2 = np.where(r2 > r2_v2,r2_v2,r2)
        dy_v2 = (np.abs(dy.copy())-1*L)
        r2_v2 = dx**2+dy_v2**2
        dy = np.where(r2 > r2_v2,dy_v2*np.sign(dy),dy)
        r2 = np.where(r2 > r2_v2,r2_v2,r2)
        r2_v2 = dx_v2**2+dy_v2**2
        dx = np.where(r2 > r2_v2,dx_v2*np.sign(dx),dx)
        dy = np.where(r2 > r2_v2,dy_v2*np.sign(dy),dy)
        r2 = np.where(r2 > r2_v2,r2_v2,r2)

    dUx = 0.
    dUy = 0.
    utot = np.zeros((N))
    f = np.zeros((N,2))

    #JV: Now we do include this block of code outside of this function, so we don't have problems with numba
#    if((i*dt)%0.5== 0): #JV: every certain amount of steps we update the list
#        close_list = close_particles_list(r2,Nlist,N,L) #JV: matrix that contains in every row the indexs of the m closest particles

    for j in range(0,N):
        dUx = 0.
        dUy = 0.
        u = 0.

        #JV: we now calculate the force with only the Nlist closest particles
        for k in range(0,Nlist):
            c = int(close_list[j][k])

            #In the force computation we include the LJ and the walls (JV: in the verlet case). I truncate the interaction at self.R units of lenght,
            #I also avoid distances close to 0 (which only should affect the diagonal in the matrix of distances)
            #All these conditions are included using the numpy.where function.
            #If you want to include more forces you only need to add terms to these lines.

            if(vel_verlet_on == True):
                if((r2[j,c] < 4*max(R[j],R[c])) and (r2[j,c] > 10**(-2))):
                    dUx = dUx + dLJverlet(dx[j,c],r2[j,c],R[j],R[c])
                    dUy = dUy + dLJverlet(dy[j,c],r2[j,c],R[j],R[c])
#            else:
#                if((r2[j,c] < 4*max(R[j],R[c])) and (r2[j,c] > 10**(-2))):
#                    dUx = dUx + dLJverlet(dx[j,c],r2[j,c],R[j],R[c]) - dwalls([X[j],Y[j]],param)
#                    dUy = dUy + dLJverlet(dy[j,c],r2[j,c],R[j],R[c]) - dwalls([X[j],Y[j]],param)
            #JV: COMMENTED PART BECAUSE NUMBA HAS PROBLEMS WITH THE "TRY: " INSIDE DWALLS (UNSUPORTED UNTIL PYTHON 3.7), TO FIX: ERASE THE "TRY:"

            #JV: We add the energy in the corresponding array in both cases, remember that the verlet algorithm will include the energy from the walls
            # and that will be visible in fluctuations on the energy
            if(vel_verlet_on == True):
                if((r2[j,c] < 2*max(R[j],R[c])) and (r2[j,c] > 10**(-2))):
                    u = u + LJverlet(r2[j,c],R[c],R[j])
#                else:
#                    u = u + walls([X[j],Y[j]])#JV: TO CHANGE; NOW ONLY WORKS WITH VEL_VERLET_ON
#            else:
#                if((r2[j,c] < 2*max(R[j],R[c])) and (r2[j,c] > 10**(-2))):
#                    u = u + LJverlet(r2[j,c],R[c],R[j],param)
#
#                    if((X[j]**2+Y[j]**2) > (0.8*L)**2):
#                        u = u + walls([X[j],Y[j]],param)
                    #JV: COMMENTED FOR NOW, TRYING THINGS WITH NUMBA

        #JV: If the argument it's True, we will append the energy to our corresponding array
        if(append == True):
            utot[j] = u

        f[j,:] = f[j,:]+np.array([dUx,dUy])

#        #JV: Now we check where this particle is in a RxR grid, that will help us to calcule the entropy.
#        if(submenu == "Subsystems"):
#            if(j < n1**2): #JV: n1 stores the number of n1xn1 type 1 particles
#                grid[int((X[j]+0.495*L) / (L/G)), int((Y[j]+0.495*L) / (L/G)),0] += 1
#            else:
#                grid[int((X[j]+0.495*L) / (L/G)), int((Y[j]+0.495*L) / (L/G)),1] += 1
#        else:
#            grid[int((X[j]+0.495*L) / (L/G)), int((Y[j]+0.495*L) / (L/G)),0] += 1

    if(append == True):
        U[int(i)] = np.sum(utot)
#        if(submenu == "Brownian"):
#            if(wallcount[0] == 0):
#                X2[int(i)] = (abs(X[N-1]))**2
#            else:
#                X2[int(i)] = (L*wallcount[0]+(X[N-1]))**2
#        entropy[int(i)] = entropy_val

    return f

@jit(nopython=True)
def vel_verlet(t,dt,r0,v0,a0,dx,dy,r2,close_list,m,R,L,N,menu,submenu,wallpos,holesize,wallwidth,U,Nlist,vel_verlet_on,n1,grid,G,wallcount,X2,bouncing):
    """JV: vel_verlet(t,dt,r0,r1,v0,v1) performs one step of the velocity verlet algorithm at
    time t with a step of dt with the previous position r0 and the previous velocity v0, returns
    the next postion r1 and the next velocity v1. If we are on the "In a box" menu, the particles will
    elastically collide with the walls and in the "Free!" menu, they will go through the walls and appear
    on the other side."""

    r1 = r0 + v0*dt + 0.5*a0*dt**2 #JV: We calculate x(t+dt)
    a1 = (1/m)*np.transpose(fv(r1[0,:],r1[1,:],dx,dy,r2,t/dt,True,L,N,U,dt,close_list,Nlist,vel_verlet_on,R,menu,submenu,n1,grid,G,wallcount,X2)) #JV: From x(t+dt) we get a(t+dt)
    v1 = v0 + 0.5*(a0+a1)*dt #JV: From the a(t+dt) and a(t) we get v(t+dt)

    if(menu == "In a box"):
        #JV: Border conditions, elastic collision. (The "+1" is because 1 is the radius of the ball, in the reduced units that we calculate this part)
        v1[0,:] = np.where((np.abs(r1[0,:])+R/2)**2 > (0.49*L)**2,-v1[0,:],v1[0,:])
        v1[1,:] = np.where((np.abs(r1[1,:])+R/2)**2 > (0.49*L)**2,-v1[1,:],v1[1,:])
    elif(menu == "Free!"):
        if(submenu == "Brownian"):
            #JV: We want to track the position of the brownian ball. If it goes through a wall, we will acknowledge it and save this into a variable that
            # counts the amount of times it has gone through (imagine this as a grid of simulations in which we start at the center one)
            if(r1[0,N-1] > 0.5*L):
#                    print("passa x")
                wallcount[0] += 1 #JV: At position 0 in self.wallcount we track the x axis, we sum 1 if it goes through the right wall
            elif(r1[0,N-1] < -0.5*L):
#                    print("passa -x")
                wallcount[0] -= 1 #JV: If it goes through the left wall, we subtract 1
            elif(r1[1,N-1] > 0.5*L):
#                    print("passa y")
                wallcount[1] += 1 #JV: Now the same for the y axis
            elif(r1[1,N-1] < -0.5*L):
#                    print("passa -y")
                wallcount[1] -= 1
        r1[0,:] = np.where(r1[0,:] > 0.5*L,r1[0,:]-1*L,r1[0,:])
        r1[0,:] = np.where(r1[0,:] < -0.5*L,r1[0,:]+1*L,r1[0,:])
        r1[1,:] = np.where(r1[1,:] > 0.5*L,r1[1,:]-1*L,r1[1,:])
        r1[1,:] = np.where(r1[1,:] < -0.5*L,r1[1,:]+1*L,r1[1,:])
    elif(menu == "Walls"):
        #JV: Border conditions, elastic collision with the limits of the simulations aswell as the elastic wall
        #JV: First the limits of the simulation
        bouncing_limits_x = np.where((np.abs(r1[0,:])+R/2)**2 > (0.49*L)**2,True,False)
        bouncing_limits_y = np.where((np.abs(r1[1,:])+R/2)**2 > (0.49*L)**2,True,False)
        #JV: Now the elastic wall
#            wallpos = self.param[7]
#            holesize = self.param[8]
#            wallwidth = self.param[9]
        bounce_left = np.where(np.logical_and(r1[0,:]+R/2 > (wallpos-wallwidth/2),np.abs(r1[1,:])+R/2 > holesize/2),True,False)
        bounce_right = np.where(np.logical_and(r1[0,:]-R/2 < (wallpos+wallwidth/2),np.abs(r1[1,:])+R/2 > holesize/2),True,False)
        is_leftside = np.where(r1[0,:] < wallpos, True, False)
        is_inhole = np.where(np.abs(r1[1,:])+R/2 > holesize/2, True, False)

        for i in range (N):
            if(bouncing_limits_x[i]):
                v1[0,i] = -v1[0,i]
            elif(bouncing_limits_y[i]):
                v1[1,i] = -v1[1,i]
            else:
                if(is_leftside[i] and bounce_left[i] and is_inhole[i] and bouncing[i] == 0):
                    v1[0,i] = -v1[0,i]
#                    print("bounce_left ",i)
                elif(not(is_leftside[i]) and bounce_right[i] and is_inhole[i] and bouncing[i] == 0):
                    v1[0,i] = -v1[0,i]
#                    print("bounce_right ",i)

                #JV: This additional condition is because we want to avoid particles entering in a loop of conditions when bouncing and
                # making them "get stuck" in the middle of the wall, so now when it bounces it has to wait 2 more time steps to be able to bounce again
                if(is_leftside[i] and bounce_left[i] and is_inhole[i]):
                    bouncing[i] += 1
                elif(not(is_leftside[i]) and bounce_right[i] and is_inhole[i]):
                    bouncing[i] += 1
                else:
                    if (bouncing[i] != 0):
                        bouncing[i] -= 1

#            print(r1[1,:], holesize/2)
#            v1[0,:] = np.where(r1[0,:] < wallpos, np.where(np.logical_and(r1[0,:] + 1 > (wallpos-wallwidth/2),abs(r1[1,:]) > holesize/2),-v1[0,:],v1[0,:]),np.where(np.logical_and(r1[0,:] - 1 < (wallpos+wallwidth/2),abs(r1[1,:]) > holesize/2),-v1[0,:],v1[0,:]))

#            v1[0,:] = np.where(np.logical_and(np.logical_and(r1[0,:] + 1 > (wallpos-wallwidth/2),r1[0,:] - 1 < (wallpos+wallwidth/2)),abs(r1[1,:]) > holesize/2),-v1[0,:],v1[0,:])
#            v1[1,:] = np.where(np.logical_and(abs(r1[1,:]) + 1 > holesize/2,np.logical_and(r1[0,:] + 1 > wallpos+wallwidth/2, r1[0,:] - 1 < wallpos+wallwidth/2)),-v1[1,:],v1[1,:])


    return r1[0,:],r1[1,:],v1[0,:],v1[1,:],a1

class particle:
    """particle(m,q,r0,v0,D) class stores the intrinsic properties of a particle (mass, charge)
    and its initial and current position (r0,v0) as well as the dimension of the space (D).
    The dimension of the space is not used but it could be useful for some applications.
    r0 and v0 can be numpy.arrays or lists"""

    def __init__(self,m,q,R,r0,v0,D):
        self.m = m
        self.q = q
        self.R = R
        self.r0 = r0
        self.v0 = v0
        self.r = r0
        self.v = v0
    def reset(self):
        self.r = self.r0
        self.v = self.v0

class PhySystem:
    """PhySystem(particles,param) class stores all the particles and contains functions for
    computing the trajectory of the system. Particles is a list or numpy.array full of
    particle objects. param is a numpy array or list with any parameters of the system.
    PhySystem has the verlet algorythm incorporated and param are the units of the system
    (potential depth for the LJ for energy,particle radius for lenght and size of the box, see
    intsim.py and documentation for more details on reduced units)."""
    def __init__(self,particles,param):
        self.particles = particles
        self.param = param
        #Usage of the vectorize function is very useful throughout this class,
        #in this case is not necessary since all particles have the same mass
        #but it is useful for other aplications (Gravitational problems, for example)
        self.m = np.vectorize(lambda i: i.m)(particles)
        self.R = np.vectorize(lambda i: i.R)(particles)
#        self.q = np.vectorize(lambda i: i.q)(particles)
        self.U = np.array([])
        self.X2 = np.array([])
        self.entropy = np.array([])


    def verlet(self,t,dt,r0,r1):
        """verlet(t,dt,r0,r1) performs one step of the verlet algorythm at time t
        with a step of dt with the previous position r0 and the current position r1, returns
        the next position r2.
        All of the r have shape (2,N) where N is the number of particles. The first
        index acceses either the x or y coordinate and the second the particle. The function
        returns the coordinates by separate."""
        r2 = np.zeros([2,self.particles.size])

        MX, MXT = np.meshgrid(r1[0,:],r1[0,:],copy=False)
        MY, MYT = np.meshgrid(r1[1,:],r1[1,:],copy=False)
        dx = MXT - MX
        dx = dx

        dy = MYT - MY
        dy = dy

        r2 = np.square(dx)+np.square(dy)

        if(np.round((t/self.dt*dt)%0.5,1) == 0): #JV: every certain amount of steps we update the list
            self.close_list = close_particles_list(r2,self.Nlist,self.particles.size,self.param[2]) #JV: matrix that contains in every row the indexs of the m closest particles

        r2 = (2*r1 - r0 + np.transpose(fv(r1[0,:],r1[1,:],dx,dy,r2,t/self.dt,True,self.param[2],self.particles.size,self.U,self.dt,self.close_list,self.Nlist,self.vel_verlet_on,self.R,self.param[3],self.param[4],self.param[5],self.grid,self.G,self.wallcount,self.X2)) * (dt**2))
        #The transpose is necessary because I messed up the shapes when I did the fv function.

        #JV: this needs to change if we want to include particles with mass diferent than 1 (in reduced units),
        # in other words, diferent particles than the Argon gas

        return r2[0,:],r2[1,:]

    def solveverlet(self,T,dt):
        """solververlet(T,dt) solves the equation of movement from t=0 to t=T
        at a step of dt. It also computes the potential and kinetic energy as well
        as the temperature of the system both at each instant and acumulated
        every delta (see below)."""
        t = 0.
        self.dt = dt
        self.n = int(T/dt)
        L = self.param[2]
        N = self.particles.size

        self.U = np.zeros([self.n])

        progress = t/T*100

        #JV: Here we define the number of the GxG grid that we will need to calcule the entropy, change in order to change the precision of this grid
        self.G = 5

        #JV: We create a list that will be useful for the walls submenu, that will help us in the border conditions of the wall, see in vel_verlet()
        self.bouncing = np.zeros(self.particles.size)

        if(self.param[4] == "Subsystems"): #JV: If we are on "Subsystems", we will count different the types of particles
            self.grid = np.zeros([self.G,self.G,2])
        else:
            self.grid = np.zeros([self.G,self.G,2]) #JV: When we are not in "Subsystems", we will have the same type of variable, but will only use the [:,:,0] (this is because numba has problems otherwise)

        self.entropy_val = 0

        #JV: If we are simulating the brownian simulation, we initialize the array that will keep track if the brownian particle goes through a wall
        if(self.param[4] == "Brownian"):
            self.wallcount = np.zeros([2])
        else:
            self.wallcount = np.zeros([2]) #JV: We have to keep both in the same type of variables, otherwise numba will have problems. So now this conditional block is quite poinless. TO-ERASE

        np.vectorize(lambda i: i.reset())(self.particles) #This line resets the particles to their initial position

        self.vel_verlet_on = True #JV: If it's true, it will compute with the velocity verlet algorithm, if it's not, it will compute with normal verlet

        self.Nlist = int(1*(self.particles.size)**(1/2)) #JV:This variable defines the number of close particles that will be stored in the list (go to close_particles_list() for more info)
#        self.Nlist = int(np.arctan(((self.particles.size)**(1/2))/8)*60/np.pi)
        print(self.Nlist)
        #X,Y,VX,VY has the trajectories of the particles with two indexes that
        #access time and particles, respectively
        self.X = np.vectorize(lambda i: i.r[0])(self.particles)
        self.Y = np.vectorize(lambda i: i.r[1])(self.particles)
        self.VX = np.vectorize(lambda i: i.v[0])(self.particles)
        self.VY = np.vectorize(lambda i: i.v[1])(self.particles)

        MX, MXT = np.meshgrid(self.X[:],self.X[:])
        MY, MYT = np.meshgrid(self.Y[:],self.Y[:])

        #JV: So dx is a nxn simetric array with 0 in the diagonal, and each position is the corresponding distance between the particles,
        # so the position [1,2] is the distance between partcle 1 and 2 (x1-x2), and so on
        dx = MXT - MX
        dx = dx

        dy = MYT - MY
        dy = dy

        r2 = np.square(dx)+np.square(dy)

        self.close_list = close_particles_list(r2,self.Nlist,self.particles.size,self.param[2]) #JV: we first calculate the matrix that contains in every row the indexs of the m closest particles

        #JV: DELETE -> close_particles_list CALLED 2 TIMES, ONLY NEEDED 1


        if(self.vel_verlet_on == True):
            #JV: We define the variables that we will need in the velocity verlet algorithm
            print("Computing with the Velocity-Verlet algorithm")
            X0 = self.X
            Y0 = self.Y
            VX0 = self.VX
            VY0 = self.VY

            X1 = self.X
            Y1 = self.Y
            VX1 = self.VX
            VY1 = self.VY

            MX, MXT = np.meshgrid(X0[:],X0[:],copy=False)
            MY, MYT = np.meshgrid(Y0[:],Y0[:],copy=False)
            dx = MXT - MX
            dx = dx

            dy = MYT - MY
            dy = dy

            r2 = np.square(dx)+np.square(dy)

            if(np.round((t/self.dt*dt)%0.5,1) == 0): #JV: every certain amount of steps we update the list
                self.close_list = close_particles_list(r2,self.Nlist,self.particles.size,self.param[2]) #JV: matrix that contains in every row the indexs of the m closest particles

            a0 = (1/self.m)*np.transpose(fv(X0[:],Y0[:],dx,dy,r2,t/self.dt,False,self.param[2],self.particles.size,self.U,self.dt,self.close_list,self.Nlist,self.vel_verlet_on,self.R,self.param[3],self.param[4],self.param[5],self.grid,self.G,self.wallcount,self.X2))

            for i in range(0, self.n):
                r1 = np.array([X0,Y0]) + np.array([VX0,VY0])*dt + 0.5*a0*dt**2

                MX, MXT = np.meshgrid(r1[0,:],r1[0,:],copy=False)
                MY, MYT = np.meshgrid(r1[1,:],r1[1,:],copy=False)
                dx = MXT - MX
                dx = dx

                dy = MYT - MY
                dy = dy

                r2 = np.square(dx)+np.square(dy)

                #JV: call velocityverlet to compute the next position
                if(np.round((t/self.dt*dt)%0.5,1) == 0): #JV: every certain amount of steps we update the list
                    self.close_list = close_particles_list(r2,self.Nlist,self.particles.size,self.param[2]) #JV: matrix that contains in every row the indexs of the m closest particles

                X1,Y1,VX1,VY1,a1 = vel_verlet(t,dt,np.array([X0,Y0]),np.array([VX0,VY0]),a0,dx,dy,r2,self.close_list,self.m,self.R,L,N,self.param[3],self.param[4],self.param[7],self.param[8],self.param[9],self.U,self.Nlist,self.vel_verlet_on,self.param[5],self.grid,self.G,self.wallcount,self.X2,self.bouncing)

                #JV: Now we check where this particle is in a RxR grid, that will help us to calcule the entropy.
                for h in range(0, N):
                    if(self.param[4] == "Subsystems"):
                        if(h < self.param[5]**2): #JV: self.param[5] stores the number of n1xn1 type 1 particles
                            self.grid[int((X1[h]+0.495*L) / (L/self.G)), int((Y1[h]+0.495*L) / (L/self.G)),0] += 1
                        else:
                            self.grid[int((X1[h]+0.495*L) / (L/self.G)), int((Y1[h]+0.495*L) / (L/self.G)),1] += 1
                    else:
                        self.grid[int((X1[h]+0.495*L) / (L/self.G)), int((Y1[h]+0.495*L) / (L/self.G))] += 1

                if(self.param[4] == "Brownian"):
                    if(self.wallcount[0] == 0):
                        self.X2 = np.append(self.X2,(abs(X1[N-1]))**2)
                    else:
                        self.X2 = np.append(self.X2,(L*self.wallcount[0]+(X1[N-1]))**2)
                self.entropy = np.append(self.entropy,self.entropy_val)

                t += dt

                self.X = np.vstack((self.X,X1))
                self.Y = np.vstack((self.Y,Y1))
                self.VX = np.vstack((self.VX, VX1))
                self.VY = np.vstack((self.VY, VY1))
                a0 = a1

                #Redefine and repeat
                X0,Y0 = X1,Y1
                VX0,VY0 = VX1,VY1

                #JV: Every amount of steps of time we calculate the entropy
                update_entropy = 1
                if(i % update_entropy == 0):

                    self.entropy_val = 0
                    sumagrid = np.sum(self.grid)

                    if(self.param[4] == "Subsystems"):
                        sumagrid_subs = np.zeros([2])
                        sumagrid_subs[0] = np.sum(self.grid[:,:,0]) #JV: Number of type-0 particles
                        sumagrid_subs[1] = sumagrid - sumagrid_subs[0] #JV: Number of type-1 particles

                        for j in range(self.G):
                            for k in range(self.G):
                                for l in range(2):
                                    if ((self.grid[j,k,0]+self.grid[j,k,1]) != 0):
                                        pji = float(self.grid[j,k,l])/(update_entropy*(self.grid[j,k,0]+self.grid[j,k,1]))
                                    else:
                                        pji = 0
                                    if(pji != 0):
                                        self.entropy_val += -pji*np.log(pji) #JV: We will only calculate the value when pji != 0

#                        pjc = p0/sum(p0)
#                        pjc = p0
#
#                        for j in range(counter):
#                            if(pjc[j] != 0):
#                                self.entropy_val += -pjc[j]*np.log(pjc[j])

                        self.entropy_val = self.entropy_val /(self.G**2)

                    else:
                        for j in range(self.G):
                            for k in range(self.G):
                                pji = float(self.grid[j,k,0])/(update_entropy*sumagrid)
                                if(pji != 0):
                                    self.entropy_val  += -pji*np.log(pji)

                        self.entropy_val = self.entropy_val /(self.G**2)

                    if(self.param[4] == "Subsystems"):
                        self.grid = np.zeros([self.G,self.G,2])
                    else:
                        self.grid = np.zeros([self.G,self.G,2])

                #Update and show progress through console
                progress = t/T*100
                if(i%1000 == 0):
                    print(int(progress),'% done')

        else:
            print("Computing with the Verlet algorithm")

            #Generation of the precious position (backwards euler step)
            X1 = self.X
            Y1 = self.Y
            X0 = X1 - self.VX*dt
            Y0 = Y1 - self.VY*dt

            for self.i in range(0,self.n):
                #Call verlet to compute the next position
                X2,Y2 = self.verlet(t,dt,np.array([X0,Y0]),np.array([X1,Y1]))
                t = t + dt

                #Add the new positions to X,Y,VX,VY
                self.X = np.vstack((self.X,X2))
                self.Y = np.vstack((self.Y,Y2))
                self.VX = np.vstack((self.VX,(X2-X0)/(2*dt)))
                self.VY = np.vstack((self.VY,(Y2-Y0)/(2*dt)))

                #Redefine and repeat
                X0,Y0 = X1,Y1
                X1,Y1 = X2,Y2

                #Update and show progress through console
                progress = t/T*100
                if(self.i%1000 == 0):
                    print(int(progress),'% done')

        #Once the computation has ended, I compute the kinetic energy,
        #the magnitude of the velocity V and the temperature
        #(see doc for temperature definition)
        self.KE()
        self.V = np.sqrt((self.VX**2 + self.VY**2))
        self.T = (np.sum(self.V**2,axis=1)/(self.particles.size*2 - 2))

        #Generation of the MB functions, you can modify the definition by
        #changing the linspace points
        vs,a = np.meshgrid(np.linspace(0,self.V.max(),100),self.T)
        a,ts = np.meshgrid(np.linspace(0,self.V.max(),100),self.T)
        self.MB = (vs/(ts)*np.exp(-vs**2/(2*ts)))

        #JV: If we are on the Subsystems submenu, we will calculate the temperature and the MB distribution of both types of particles
        if(self.param[4] == "Subsystems"):

            #JV: 1st group of particles
            self.V1 = np.sqrt((self.VX[:,0:(self.param[5]**2)]**2 + self.VY[:,0:(self.param[5]**2)]**2))
            self.T1 = (np.sum(self.V1**2,axis=1)/((self.param[5]**2)*2 - 2))

            vs1,a1 = np.meshgrid(np.linspace(0,self.V1.max(),100),self.T1)
            a1,ts1 = np.meshgrid(np.linspace(0,self.V1.max(),100),self.T1)
            self.MB1 = (vs1/(ts1)*np.exp(-vs1**2/(2*ts1)))

            #JV: 2nd group
            self.V2 = np.sqrt((self.VX[:,(self.param[5]**2):self.particles.size]**2 + self.VY[:,(self.param[5]**2):self.particles.size]**2))
            self.T2 = (np.sum(self.V2**2,axis=1)/((self.particles.size-self.param[5]**2)*2 - 2))

            vs2,a2 = np.meshgrid(np.linspace(0,self.V2.max(),100),self.T2)
            a2,ts2 = np.meshgrid(np.linspace(0,self.V2.max(),100),self.T2)
            self.MB2 = (vs2/(ts2)*np.exp(-vs2**2/(2*ts2)))

        """Here I generate the accumulated V,T and MB using lists, the reason I use lists is because if you append two numpy arrays
         to an empty numpy array, they merge instead of remaining separate. You could technically use splicing to save on memory
         but sacrificing cpu."""

        self.Vacu = []
        self.Tacu = []
        self.MBacu = []
        self.Vacu.append(self.V[int(self.n/2),:])
        self.Tacu.append(np.sum(self.V[int(self.n/2),:]**2)/(self.particles.size*2 - 2))

        vs = np.linspace(0,self.V.max(),100)
        self.MBacu.append((vs/(self.Tacu[0])*np.exp(-vs**2/(2*self.Tacu[0]))))

        #This delta controls the time interval for accumulation, right now its every 5 units
        delta = 5./dt

        #This 40 that appers in these lines is the time from which I start accumulating
        #to ensure the system has reached equilibrium.
        for i in range(1,int((self.n-(40./dt))/delta)):
            self.Vacu.append(np.hstack((self.Vacu[i-1],self.V[int(40./dt)+int(i*delta),:])))
            self.Tacu.append(np.sum(self.Vacu[i]**2)/(self.Vacu[i].size*2 - 2))
            self.MBacu.append((vs/(self.Tacu[i])*np.exp(-vs**2/(2*self.Tacu[i]))))
        return


    def KE(self):
        #Function for computing the kinetic energy, it also computes the mean kinetic energy.
        Ki = self.m*(self.VX**2 + self.VY**2)/2.
        self.K = np.sum(Ki,axis=1)[1:]
        self.Kmean = (np.sum(Ki,axis=1)/self.particles.size)[1:]
        return