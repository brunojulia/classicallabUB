
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



class particle:
    
    def __init__(self,m,q,r0,v0,D):
        self.m = m
        self.q = q
        self.r0 = r0
        self.v0 = v0
        self.r = r0
        self.v = v0


class PhySystem:
    
    def __init__(self,particles,param):
        self.particles = particles
        self.param = param
        self.m = np.vectorize(lambda i: i.m)(particles)
#        self.q = np.vectorize(lambda i: i.q)(particles)
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
    
    def KE(self):
        self.K = np.sum((self.m*(self.VX**2 + self.VY**2)/2.),axis=1)
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
#s = PhySystem(np.array([a,b,c,d,e]))
#s = PhySystem(np.array([a,b]))
#print(s.particles[0].r)
#print(s.particles[1].r)
#X,Y = s.solve(10,0.1)


#a = np.array([a,b,c,d,e])
#msf = lambda i: i.m
#msfv = np.vectorize(msf)
#
#ms = np.vectorize(lambda i: i.m)(a) 
#R = np.vectorize(lambda i: i.r[0])(a)
#MR, MRT = np.meshgrid(R,R)