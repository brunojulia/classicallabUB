import numpy as np
import sys

#JV: Truncated Lennard-Jones potentials (WCA), see LJverlet function for more explanation and documentation for the expressions.
#JV: We include both radius as parameters so we can generalize the function to interaction between diferent radius
def dLJverlet(x,r2,R1,R2,param):
    """The derivative has the same form for x and y so only one is needed,
    this only changes when calling the interaction on the algotyhm,
    for all isotrope interactions this should still hold."""
    V = param[0]
    sig = param[1] #JV: This is the system units, so we will calculate the new sigma relative to this
    L = param[2]
    rc = (2**(1/6))*((R1+R2)/(2*sig))
    sig_int = (R1+R2)/(2*sig) #JV: This is the sigma of the interaction (in the system units)

    #JV: Because we are working on reduced units (from the values of the Argon gas)
    # we want need to divide our radius by the radius of the Argon gas

    #JV: See LJverlet() for more explanation on the truncation
    if((r2**(1/2))>rc):
        value = 0
    else:
        value = ((48.*x)/(r2))*(((((sig_int**2)*1.)/r2)**6) - ((((sig_int**2)*0.5)/r2)**3))

    return value

def LJverlet(r2,R1,R2,param):
    V = param[0]
    sig = param[1] #JV: This is the system units, so we will calculate the new sigma relative to this
    L = param[2]
    rc = (2**(1/6))*((R1+R2)/(2*sig))
    sig_int = (R1+R2)/(2*sig) #JV: This is the sigma of the interaction (in the system units)

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

    def verlet(self,t,dt,r0,r1):
        """verlet(t,dt,r0,r1) performs one step of the verlet algorythm at time t
        with a step of dt with the previous position r0 and the current position r1, returns
        the next position r2.
        All of the r have shape (2,N) where N is the number of particles. The first
        index acceses either the x or y coordinate and the second the particle. The function
        returns the coordinates by separate."""
        r2 = np.zeros([2,self.particles.size])
        r2 = (2*r1 - r0 + np.transpose(self.fv(r1[0,:],r1[1,:],True)) * (dt**2))
        #The transpose is necessary because I messed up the shapes when I did the fv function.

        #JV: this needs to change in order to include particles with mass diferent than 1 (in reduced units),
        # in other words, diferent particles than the Argon gas

        return r2[0,:],r2[1,:]

    def vel_verlet(self,t,dt,r0,v0,a0):
        """JV: vel_verlet(t,dt,r0,r1,v0,v1) performs one step of the velocity verlet algorithm at
        time t with a step of dt with the previous position r0 and the previous velocity v0, returns
        the next postion r1 and the next velocity v1. If we are on the "In a box" menu, the particles will
        elastically collide with the walls and in the "Free!" menu, they will go through the walls and appear
        on the other side."""
        r1 = r0 + v0*dt + 0.5*a0*dt**2 #JV: We calculate x(t+dt)
        a1 = (1/self.m)*np.transpose(self.fv(r1[0,:],r1[1,:],True)) #JV: From x(t+dt) we get a(t+dt)
        v1 = v0 + 0.5*(a0+a1)*dt #JV: From the a(t+dt) and a(t) we get v(t+dt)

        L = self.param[2]

#        print(r1)
#        print("")
#        print(0.5*L)
#        sys.exit()
        if(self.param[3] == "In a box"):
            #JV: Border conditions, elastic collision
            v1[0,:] = np.where((r1[0,:]**2)+(self.R)**2 > (0.495*L)**2,-v1[0,:],v1[0,:])
            v1[1,:] = np.where((r1[1,:]**2)+(self.R)**2 > (0.495*L)**2,-v1[1,:],v1[1,:])
        elif(self.param[3] == "Free!"):
            r1[0,:] = np.where(r1[0,:] > 0.5*L,r1[0,:]-1*L,r1[0,:])
            r1[0,:] = np.where(r1[0,:] < -0.5*L,r1[0,:]+1*L,r1[0,:])
            r1[1,:] = np.where(r1[1,:] > 0.5*L,r1[1,:]-1*L,r1[1,:])
            r1[1,:] = np.where(r1[1,:] < -0.5*L,r1[1,:]+1*L,r1[1,:])

        return r1[0,:],r1[1,:],v1[0,:],v1[1,:],a1

    def close_particles_list(self,r2,m):
        """JV: This functions returns a list (a matrix) where each rows saves m indexs corresponding to the m
        closest particles. We will use the r2 matrix calculated in fv() function (go there for more information),
        that contains the distance between our particles"""
        dist = r2.copy()
        N = self.particles.size
        L = self.param[2]


        close_list = []

        for i in range(0, N):
            temp_index = []
            for j in range(0, m):
                min_dist = L**3
                index_min = 0

                for k in range(0, N):
                    if(dist[i,k] < min_dist and not(dist[i,k] == 0.)):
                        min_dist = dist[i,k]
                        index_min = k

                temp_index.append(index_min)
                dist[i,index_min] = L**3

            close_list.append(temp_index)

        return close_list



    def fv(self,X,Y,append):
        """fv(X,Y) represents the forces that act on all the particles at a particular time.
        It computes the matrix of forces using the positions given with X and Y which are
        the arrays of size N containing all the positions (coordinates X and Y).
        The resulting matrix, f is of shape (N,2) (it should be (2,N), see the verlet function)."""

        """JV: append is a boolean. If it's true, adds the energy to our list, if it isn't, it doesn't.
         We do that because in some cases we will call the algorithm more times than the actual step number (and
         we only want to sum the value T/dt times), this is needed in the velocity-Verlet algorithm, that we call the fv()
         function one more time than needed just to start the loop."""

        L = self.param[2]

        N = self.particles.size

        #For computing all the distances I use a trick with the meshgrid function,
        #see the documentation on how this works if you dont see it.

        """JV: X is an array that contains each position, mx is an nxn array that each column is the position of one particle (so it's a matrix
        that has n X rows) and mxt is the same but tranposed (so it's a matrix of n X columns)"""
        MX, MXT = np.meshgrid(X,X)
        MY, MYT = np.meshgrid(Y,Y)

        #JV: So dx is a nxn simetric array with 0 in the diagonal, and each position is the corresponding distance between the particles,
        # so the position [1,2] is the distance between partcle 1 and 2 (x1-x2), and so on
        dx = MXT - MX
        dx = dx

        dy = MYT - MY
        dy = dy

        r2 = np.square(dx)+np.square(dy)

        if(self.param[3] == "Free!"):
        #JV: We do this to get the actual distance in the case of the "Free!" simulation
#            print(dx)
#            print("")
#            print(dy)
#            print("")
#            print(r2)
#            print("")
            dx_v2 = (abs(dx.copy())-1*L)
            r2_v2 = dx_v2**2+dy**2
            dx = np.where(r2 > r2_v2,dx_v2*np.sign(dx),dx)
            r2 = np.where(r2 > r2_v2,r2_v2,r2)
            dy_v2 = (abs(dy.copy())-1*L)
            r2_v2 = dx**2+dy_v2**2
            dy = np.where(r2 > r2_v2,dy_v2*np.sign(dy),dy)
            r2 = np.where(r2 > r2_v2,r2_v2,r2)
            r2_v2 = dx_v2**2+dy_v2**2
            dx = np.where(r2 > r2_v2,dx_v2*np.sign(dx),dx)
            dy = np.where(r2 > r2_v2,dy_v2*np.sign(dy),dy)
            r2 = np.where(r2 > r2_v2,r2_v2,r2)
#            print(dx)
#            print("")
#            print(dy)
#            print("")
#            print(r2)
#            sys.exit()

        dUx = 0.
        dUy = 0.
        utot = np.array([])
        f = np.zeros([N,2])

        i = len(self.U)

        if((i*self.dt)%0.5 == 0): #JV: every certain amount of steps we update the list
            self.close_list = self.close_particles_list(r2,self.Nlist) #JV: matrix that contains in every row the indexs of the m closest particles
#            print(self.close_list)
#            print("")
#            print(r2)
#            sys.exit()

        for j in range(0,N):
            dUx = 0
            dUy = 0
            u = 0

            #JV: we now calculate the force with only the sqrt(N) closest particles
            for k in range(0,self.Nlist):
                c = int(self.close_list[j][k])

                #In the force computation we include the LJ and the walls (JV: in the verlet case). I truncate the interaction at self.R units of lenght,
                #I also avoid distances close to 0 (which only should affect the diagonal in the matrix of distances)
                #All these conditions are included using the numpy.where function.
                #If you want to include more forces you only need to add terms to these lines.

                if(self.vel_verlet_on == True):
                    if((r2[j,c] < 1.2*max(self.R[j],self.R[c])) and (r2[j,c] > 10**(-2))):
                        dUx = dUx + dLJverlet(dx[j,c],r2[j,c],self.R[j],self.R[c],self.param)
                        dUy = dUy + dLJverlet(dy[j,c],r2[j,c],self.R[j],self.R[c],self.param)
                else:
                    if((r2[j,c] < 1.2*max(self.R[j],self.R[c])) and (r2[j,c] > 10**(-2))):
                        dUx = dUx + dLJverlet(dx[j,c],r2[j,c],self.R[j],self.R[c],self.param) - dwalls([X[j],Y[j]],self.param)
                        dUy = dUy + dLJverlet(dy[j,c],r2[j,c],self.R[j],self.R[c],self.param) - dwalls([X[j],Y[j]],self.param)

                #JV: We add the energy in the corresponding array in both cases, remember that the verlet algorithm will include the energy from the walls
                # and that will be visible in fluctuations on the energy
                if(self.vel_verlet_on == True):
                    if((r2[j,c] < 1.2*max(self.R[j],self.R[c])) and (r2[j,c] > 10**(-2))):
                        u = np.append(u, LJverlet(r2[j,c],self.R[c],self.R[j],self.param))
                    else:
                        u = np.append(u, 0)
                else:
                    if((r2[j,c] < max(self.R[j],self.R[c])) and (r2[j,c] > 10**(-2))):
                        u = u + LJverlet(r2[j,c],self.R[c],self.R[j],self.param)

                        if((X[j]**2+Y[j]**2) > (0.8*L)**2):
                            u = u + walls([X[j],Y[j]],self.param)

            #JV: If the argument it's True, we will append the energy to our corresponding aray
            if(append == True):
                utot = np.append(utot,u)

            f[j,:] = f[j,:]+np.array([dUx,dUy])

        if(append == True):
            self.U = np.append(self.U,np.sum(utot))

        return f

    def solveverlet(self,T,dt):
        """solververlet(T,dt) solves the equation of movement from t=0 to t=T
        at a step of dt. It also computes the potential and kinetic energy as well
        as the temperature of the system both at each instant and acumulated
        every delta (see below)."""
        t = 0.
        self.dt = dt
        self.n = int(T/dt)

        progress = t/T*100

        np.vectorize(lambda i: i.reset())(self.particles) #This line resets the particles to their initial position

        self.vel_verlet_on = True #JV: If it's true, it will compute with the velocity verlet algorithm, if it's not, it will compute with normal verlet

        self.Nlist = int(2*(self.particles.size)**(1/2)) #JV:This variable defines the number of close particles that will be stored in the list (go to close_particles_list() for more info)

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

        self.close_list = self.close_particles_list(r2,self.Nlist) #JV: we first calculate the matrix that contains in every row the indexs of the m closest particles


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

            a0 = (1/self.m)*np.transpose(self.fv(X0[:],Y0[:],False))

            for i in range(0, self.n):
                #JV: call velocityverlet to compute the next position
                X1,Y1,VX1,VY1,a1 = self.vel_verlet(t,dt,np.array([X0,Y0]),np.array([VX0,VY0]),a0)

                t = t + dt

                self.X = np.vstack((self.X,X1))
                self.Y = np.vstack((self.Y,Y1))
                self.VX = np.vstack((self.VX, VX1))
                self.VY = np.vstack((self.VY, VY1))
                a0 = a1

                #Redefine and repeat
                X0,Y0 = X1,Y1
                VX0,VY0 = VX1,VY1

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

        #Here I generate the accumulated V,T and MB using lists
        #The reason I use lists is because if you append
        #two numpy arrays to an empty numpy array
        #they merge instead of remaining separate
        #You could technically use splicing to save on memory
        #but sacrificing cpu
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
        #Function for computing the kinetic energy, it also computes the mean kinetic energy.p
        Ki = self.m*(self.VX**2 + self.VY**2)/2.
        self.K = np.sum(Ki,axis=1)[1:]
        self.Kmean = (np.sum(Ki,axis=1)/self.particles.size)[1:]
        return