
import numpy as np

def LJ(x,y,param):
    V = param[0]
    sig = param[1]
    
    d = np.sqrt(x**2+y**2)
    mask = np.where(d < 0.000001,1,0)
    d = d + mask
    
    value = 4*V*((sig/d)**12-(sig/d)**6)
    value = value*(1-mask)
    return value

def dLJx(x,y,param):
    V = param[0]
    sig = param[1]
    
    d = x**2+y**2
    mask = np.where(d < 0.000001,1,0)
    d = d + mask
    x = x + mask
    
    value = 24*(sig**6)*V*x*((d**3 - 2*(sig**6))/(d**7))
    value = value*(1-mask)
    return value

def dLJy(x,y,param):
    V = param[0]
    sig = param[1]
    
    d = x**2+y**2
    mask = np.where(d < 0.000001,1,0)
    d = d + mask
    y = y + mask
    
    value = 24*(sig**6)*V*y*((d**3 - 2*(sig**6))/(d**7))
    value = value*(1-mask)
#    print(value)
    return value

def dLJverlet(x,r2,param):
    V = param[0]
    sig = param[1]
    L = param[2]
    rc = L/2.
    
    
    value = (48.*x)/(r2)*(1./(r2**6) - 0.5/(r2**3))

    return value

def LJverlet(r2,param):
    V = param[0]
    sig = param[1]
    L = param[2]
    rc = L/2.
    
    
    
    value = 4*(1./(r2**6) - 1./(r2**3)) - 4*(1./(rc**6) - 1./(rc**3))

    return value


class particle:
    
    def __init__(self,m,q,r0,v0,D):
        self.m = m
        self.q = q
        self.r0 = r0
        self.v0 = v0
        self.r = r0
        self.v = v0
    def reset(self):
        self.r = self.r0
        self.v = self.v0

class PhySystem:
    
    def __init__(self,particles,param):
        self.particles = particles
        self.param = param
        self.m = np.vectorize(lambda i: i.m)(particles)
#        self.q = np.vectorize(lambda i: i.q)(particles)
        self.U = np.array([])
    def RK4(self,t,dt):
        
        k1 = self.f(t,np.zeros([self.particles.size,4]))
        k2 = self.f(t+dt/2,k1*dt/2)
        k3 = self.f(t+dt/2,k2*dt/2)
        k4 = self.f(t+dt,k3*dt)
    
        for j in range(0,self.particles.size):
            self.particles[j].r = self.particles[j].r + dt/6 * (k1[j,:2] + 2*k2[j,:2] + 2*k3[j,:2] + k4[j,:2])
            self.particles[j].v = self.particles[j].v + dt/6 * (k1[j,2:] + 2*k2[j,2:] + 2*k3[j,2:] + k4[j,2:])
            if((np.abs(self.particles[j].r[0])) >= 100):
                self.particles[j].v[0] = -self.particles[j].v[0]
            if((np.abs(self.particles[j].r[1])) >= 100):
                self.particles[j].v[1] = -self.particles[j].v[1]
        
        return
        
    def f(self,t,delta):
        N = self.particles.size
        X = np.vectorize(lambda i: i.r[0])(self.particles) + delta[:,0]
        Y = np.vectorize(lambda i: i.r[1])(self.particles) + delta[:,1]
        MX, MXT = np.meshgrid(X,X)
        MY, MYT = np.meshgrid(Y,Y)
        
        dx = MXT - MX
        dy = MYT - MY
#        print(dx,dy)
        dUx = 0.
        dUy = 0.
        
        f = np.zeros([N,4])
        for j in range(0,N):
            dUx = np.sum(dLJx(dx[j,:],dy[j,:],self.param))
            dUy = np.sum(dLJy(dx[j,:],dy[j,:],self.param))
            
            f[j,:] = np.array([self.particles[j].v[0] + delta[j,2],self.particles[j].v[1] + delta[j,3],-(1/self.particles[j].m)*dUx,-(1/self.particles[j].m)*dUy])
        return f
    
    def solve(self,T,dt):
        t = 0.
        self.n = int(T/dt)
        progress = t/T*100
        
        np.vectorize(lambda i: i.reset())(self.particles)
        
        self.X = np.vectorize(lambda i: i.r[0])(self.particles)
        self.Y = np.vectorize(lambda i: i.r[1])(self.particles)
        self.VX = np.vectorize(lambda i: i.v[0])(self.particles)
        self.VY = np.vectorize(lambda i: i.v[1])(self.particles)
        self.U = np.array([])
        
        for i in range(0,self.n):
            self.RK4(t,dt)
            t = t + dt
            self.X = np.vstack((self.X,np.vectorize(lambda i: i.r[0])(self.particles)))
            self.Y = np.vstack((self.Y,np.vectorize(lambda i: i.r[1])(self.particles)))
            self.VX = np.vstack((self.VX,np.vectorize(lambda i: i.v[0])(self.particles)))
            self.VY = np.vstack((self.VY,np.vectorize(lambda i: i.v[1])(self.particles)))
            
            progress = t/T*100
            if(progress%10 < 1/30.):
                print(int(progress),'% done')
            
            
        return
    
    def verlet(self,t,dt,r0,r1):
        r2 = np.zeros([2,self.particles.size])
        r2 = (2*r1 - r0 + np.transpose(self.fv(r1[0,:],r1[1,:])) * (dt**2))

        
        return r2[0,:],r2[1,:]
    
    def fv(self,X,Y):
        
        L = self.param[2]
        
        rc = L/2.
        
        N = self.particles.size
        MX, MXT = np.meshgrid(X,X)
        MY, MYT = np.meshgrid(Y,Y)
    
        dx = MXT - MX
        dx = dx - L*np.rint(dx/L)
        
        dy = MYT - MY
        dy = dy - L*np.rint(dy/L)
        
        r2 = np.square(dx)+np.square(dy) 
        
        dUx = 0.
        dUy = 0.
        
        utot = np.array([])        
        f = np.zeros([N,2])
        for j in range(0,N):
            dUx = np.sum(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-5)), dLJverlet(dx[j,:],r2[j,:],self.param),0.))
            dUy = np.sum(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-5)), dLJverlet(dy[j,:],r2[j,:],self.param),0.))
            u =  np.sum(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-5)), LJverlet(r2[j,:],self.param),0.))
            f[j,:] = np.array([dUx,dUy])
            utot = np.append(utot,u)
        self.U = np.append(self.U,np.sum(utot))
        return f
    
    def solveverlet(self,T,dt):
        t = 0.
        self.n = int(T/dt)
        progress = t/T*100
        
        np.vectorize(lambda i: i.reset())(self.particles)
        
        self.X = np.vectorize(lambda i: i.r[0])(self.particles)
        self.Y = np.vectorize(lambda i: i.r[1])(self.particles)
        self.VX = np.vectorize(lambda i: i.v[0])(self.particles)
        self.VY = np.vectorize(lambda i: i.v[1])(self.particles)
        
        
        X1 = self.X
        Y1 = self.Y
        X0 = X1 - self.VX*dt
        Y0 = Y1 - self.VY*dt
        
        for i in range(0,self.n):
            X2,Y2 = self.verlet(t,dt,np.array([X0,Y0]),np.array([X1,Y1]))
            t = t + dt
            
#            X2 = X2 + np.where(X2 > self.param[2]/2.,-self.param[2],0.)
#            X2 = X2 + np.where(X2 < self.param[2]/2., self.param[2],0.)
#            Y2 = Y2 + np.where(Y2 > self.param[2]/2.,-self.param[2],0.)
#            Y2 = Y2 + np.where(Y2 < self.param[2]/2., self.param[2],0.)
            
            self.X = np.vstack((self.X,X2))
            self.Y = np.vstack((self.Y,Y2))
            self.VX = np.vstack((self.VX,(X2-X0)/(2*dt)))
            self.VY = np.vstack((self.VY,(Y2-Y0)/(2*dt)))
            
            X0,Y0 = X1,Y1
            X1,Y1 = X2,Y2
            
            progress = t/T*100
            if(progress%10 == 0.):
                print(int(progress),'% done')   
        return
        
    
    def KE(self):
        Ki = self.m*(self.VX**2 + self.VY**2)/2.
        self.K = np.sum(Ki,axis=1)[1:]
        self.Kmean = (np.sum(Ki,axis=1)/self.particles.size)[1:]
        return
    def PE(self):
        self.U = np.zeros([self.n+1])
        for i in range(0,self.n+1):
            MX, MXT = np.meshgrid(self.X[i,:],self.X[i,:])
            MY, MYT = np.meshgrid(self.Y[i,:],self.Y[i,:])
            
            dx = MXT - MX
            dy = MYT - MY
            u = np.array([])
            
            for j in range(0,self.particles.size):
                u1 = np.sum(LJ(dx[j,:],dy[j,:],self.param))
                u = np.append(u,np.sum(u1))
            u = np.sum(u) #Energia potencial total en un instante
            self.U[i] = u
        return
        

#a = particle(1,1,np.array([-3,0]),np.array([0,0]),2)
#b = particle(1,1,np.array([3,0]),np.array([0,0]),2)
#c = particle(1,1,np.array([0,1]),np.array([0,1]),2)
#d = particle(1,1,np.array([3,0]),np.array([1,1]),2)
#e = particle(1,1,np.array([7,4]),np.array([-1,-1]),2)
##
#s = PhySystem(np.array([a,b,c,d,e]),[self.V0,self.R,self.L])
#s = PhySystem(np.array([a,b]))
#s = PhySystem(np.array([a]),[0.01,3.405,200.])
#print(s.particles[0].r)
#print(s.particles[1].r)
#s.solveverlet(10,0.1)


#a = np.array([a,b,c,d,e])
#msf = lambda i: i.m
#msfv = np.vectorize(msf)
#
#ms = np.vectorize(lambda i: i.m)(a) 
#R = np.vectorize(lambda i: i.r[0])(a)
#MR, MRT = np.meshgrid(R,R)