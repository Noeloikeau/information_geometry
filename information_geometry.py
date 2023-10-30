import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.collections import LineCollection

def force_aspect(ax,aspect=1):
    '''
    Helper function to force the aspect of the matplotlib 'ax' axes object.
    '''
    try:
        im = ax.get_images()
        extent =  im[0].get_extent()
    except:
        x,y=ax.get_xlim(),ax.get_ylim()
        extent = [x[0],x[1],y[0],y[1]]
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

#Define various bounds

inf = np.inf
nonzero = np.nextafter(0., 1.)
real_line = np.array([-inf,inf])
geq_zero = np.array([0,inf])
leq_zero = np.array([-inf,0])
positives = np.array([nonzero,inf])
negatives = -np.flip(positives)
unit_interval = np.array([0,1])
unit_interval_noninclusive = np.array([nonzero,1-nonzero])
perturbation = 1e-4

#Define probability distributions

@njit
def normal_distribution(x,a):
    """
    a[0] = mean
    a[1] = s.d.
    """
    return np.exp(-0.5*((x-a[0])/a[1])**2)

normal_distribution.f_shape = (1,)
normal_distribution.x = np.linspace(-5,5,10000)
normal_distribution.a = np.array([0,1])
normal_distribution.x_bounds = real_line
normal_distribution.a_bounds = np.array([real_line,positives])
normal_distribution.f_bounds = geq_zero


@njit
def normalized_distribution(x,a):
    """
    a[0] = mean
    a[1] = s.d.
    """
    return np.exp(-0.5*((x-a[0])/a[1])**2)/(a[1]*np.sqrt(2.0*np.pi))

normalized_distribution.f_shape = (1,)
normalized_distribution.x = np.linspace(-5,5,10000)
normalized_distribution.a = np.array([0,1])
normalized_distribution.x_bounds = real_line
normalized_distribution.a_bounds = np.array([real_line,positives])
normalized_distribution.f_bounds = geq_zero


@njit
def beta_distribution(x,a):
    return (x**(a[0]-1.))*(1.-x)**(a[1]-1.)

beta_distribution.f_shape = (1,)
beta_distribution.a = np.array([0.5,0.5])
beta_distribution.x = np.linspace(5*perturbation,1-5*perturbation,10000)
beta_distribution.x_bounds = unit_interval
beta_distribution.a_bounds = np.array([positives,positives])
beta_distribution.f_bounds = geq_zero


@njit
def pareto_distribution(x,a):
     sigma,zeta=a[0],a[1]
     sigmainv=1.0/sigma
     return np.exp(x)*sigmainv*(1.0+zeta*np.exp(x)*sigmainv)**(-1.0/zeta-1.0)
    
pareto_distribution.f_shape = (1,)
pareto_distribution.a = np.array([1.0,5.0]) #default point
pareto_distribution.x = np.linspace(-10,25,10000) #default x axis
pareto_distribution.x_bounds = real_line
pareto_distribution.a_bounds = np.array([positives,positives])
pareto_distribution.f_bounds = positives


#Write bounding function
@njit
def clip(f,x,a,x_bounds,a_bounds,f_bounds,f_shape=(1,),replaces_inf=0):
    """
    Restricts the domain and range of f(x,a).
    
    f (CPUDispatcher): jitted function with signature f(x,a).

    x (np.ndarray): N-dim vector of variables.

    a (np.ndarray): M-dim vector of parameters.

    x_bounds (np.ndarray) : 2-dim vector of inclusive bounds.

    a_bounds (np.ndarray) : (M*2)-dim vector of inclusive bounds.

    f_shape (tuple) : shape of the output function.

    Returns array of f_shape containing clipped function value. 
    """
    #store function evals
    F = np.zeros((a.size,)+f_shape)

    #store new variables and parameters
    new_x = np.zeros(x.shape).astype(x.dtype)
    new_a = np.zeros(a.shape).astype(a.dtype)

    #clip parameters
    for i in range(a.size):
        if a[i] < a_bounds[i,0]:
            try:
                new_a[i] = a_bounds[i,0]
            except:
                print(a,a_bounds)
        elif a[i] > a_bounds[i,1]:
            new_a[i] = a_bounds[i,1]
        else:
            new_a[i] = a[i]

    #clip variables
    for i in range(x.size):
        if x[i] < x_bounds[0]:
            new_x[i] = x_bounds[0]
        elif x[i] > x_bounds[1]:
            new_x[i] = x_bounds[1]
        else:
            new_x[i] = x[i]
    
    #evaluate & clip function
    F = f(x=new_x,a=new_a)
    for s in np.ndindex(F.shape):
        if F[s] == np.inf or F[s] == -np.inf:
            F[s] = replaces_inf
        elif F[s] < f_bounds[0]:
            F[s] = f_bounds[0]
        elif F[s] > f_bounds[1]:
            F[s] = f_bounds[1]
        else:
            pass
    
    return F

#Compute derivatives

@njit
def gradient_x(f,x,a,dx,f_shape=(1,),normalize=True):
    """
    2-nd order finite difference estimation of D_dx[f(x,a)],

    i.e. the gradient of f at (x,a) along the dx direction.

    https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

    f (CPUDispatcher): jitted function with signature f(x,a).

    x (np.ndarray): NxT matrix of vector variables.

    a (np.ndarray): M-dim vector of parameters.

    dx (np.ndarray) : N-dim vector of perturbations.

    order (int = 0, 1, or 2) : computes order'th deriv. 
    
    f_shape (tuple) : shape of the output function.

    Returns one of (f, df/dx, d2f/dx2). 

    """
    finite_differences = np.array([0,1,-1])
    F = np.zeros((finite_differences.size,x.shape[0],)+f_shape)

    d = np.sqrt(np.square(dx).sum())
    for h in finite_differences:
        for xp in range(x.shape[0]):
            new_x = x[xp] + dx * h
            F[h,xp] = f(x=new_x,a=a)
    if normalize:
        dX = (x.max()-x.min())/x.shape[0]
        F=F/(dX*F[0].sum())

    f1 = (F[1]-F[-1])/(2.0*d)
    G = np.zeros((2,x.shape[0],)+f_shape)
    G[0] = F[0]
    G[1] = f1
    return G

@njit
def gradient_a(f,x,a,da,f_shape=(1,),normalize=True):
    """
    2nd order finite difference estimation of D_dx[f(x,a)],

    i.e. the gradient of f at (x,a) along the dx direction.

    https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

    f (CPUDispatcher): jitted function with signature f(x,a).

    x (np.ndarray): N-dim matrix of vector variables.

    a (np.ndarray): M-dim vector of parameters.

    dx (np.ndarray) : N-dim vector of perturbations.

    order (int = 0, 1, or 2) : computes order'th deriv. 
    
    f_shape (tuple) : shape of the output function.

    Returns one of (f, df/dx, d2f/dx2). 

    """
    finite_differences = np.array([0,1,-1])
    F = np.zeros((finite_differences.size,x.shape[0],)+f_shape)
    d = np.sqrt(np.square(da).sum())
    for h in finite_differences:
        new_a = a + da * h
        for xp in range(x.shape[0]):
            F[h,xp] = f(x=x[xp],a=new_a)
    if normalize:
        dX = (x.max()-x.min())/x.shape[0]
        for h in finite_differences:
            F[h] = F[h]/(dX*F[h].sum())
    f1 = (F[1]-F[-1])/(2.0*d)
    G = np.empty((2,x.shape[0],)+f_shape)
    G[0] = F[0]
    G[1] = f1
    return G

@njit
def gradient_xy(f,x,a,dx1,dx2,f_shape=(1,),normalize=True):
    """
    2nd order finite difference estimation of D_dx2(D_dx1[f(x,a)]),

    i.e. the cross-partial of f at (x,a) along the dx2dx1 direction.

    https://math.stackexchange.com/questions/2931510/cross-derivatives-using-finite-differences

    f (CPUDispatcher): jitted function with signature f(x,a).

    x (np.ndarray): N-dim vector of variables.

    a (np.ndarray): M-dim vector of parameters.

    dx_ (np.ndarray) : N-dim vector of perturbations.
    
    f_shape (tuple) : shape of the output function.

    Returns array of f_shape containing cross partial. 

    """
    F = np.zeros((5,x.shape[0],)+f_shape)
    d = 4.0*np.sqrt(np.square(dx1).sum())*np.sqrt(np.square(dx2).sum())
        
    for xpt in range(x.shape[0]):
        F[4,xpt] = f(x=x[xpt],a=a)
        F[0,xpt] = f(x=x[xpt]-dx1-dx2,a=a)
        F[1,xpt] = f(x=x[xpt]-dx1+dx2,a=a)
        F[2,xpt] = f(x=x[xpt]+dx1-dx2,a=a)
        F[3,xpt] = f(x=x[xpt]+dx1+dx2,a=a)
    
    if normalize: #normalization integral
        dX = (x.max()-x.min())/x.shape[0]
        #for i in range(4):
            #F[i]=F[i]/(dX*F[i].sum())
        F=F/(dX*F[4].sum())
    
    d2f = (F[0]+F[3]-F[1]-F[2])/d
        
    return d2f


@njit
def gradient_ab(f,x,a,da1,da2,f_shape=(1,),normalize=True):
    """
    2nd order finite difference estimation of D_dx2(D_dx1[f(x,a)]),

    i.e. the cross-partial of f at (x,a) along the dx2dx1 direction.

    https://math.stackexchange.com/questions/2931510/cross-derivatives-using-finite-differences

    f (CPUDispatcher): jitted function with signature f(x,a).

    x (np.ndarray): N-dim vector of variables.

    a (np.ndarray): M-dim vector of parameters.

    dx_ (np.ndarray) : N-dim vector of perturbations.
    
    f_shape (tuple) : shape of the output function.

    Returns array of f_shape containing cross partial. 

    """
    F = np.zeros((4,x.shape[0],)+f_shape)
    d = 4.0*np.sqrt(np.square(da1).sum())*np.sqrt(np.square(da2).sum())
        
    for xpt in range(x.shape[0]):
        F[0,xpt] = f(x=x[xpt],a=a-da1-da2)
        F[1,xpt] = f(x=x[xpt],a=a-da1+da2)
        F[2,xpt] = f(x=x[xpt],a=a+da1-da2)
        F[3,xpt] = f(x=x[xpt],a=a+da1+da2)
    
    if normalize: #normalization integral
        dX = (x.max()-x.min())/x.shape[0]
        for i in range(4):
            F[i]=F[i]/(dX*F[i].sum())
    
    d2f = (F[0]+F[3]-F[1]-F[2])/d
        
    return d2f     

#Iterate over multiple parameter points and observation points

@njit
def submanifold(f,x,a,x_bounds,a_bounds,f_bounds,f_shape=(1,),replaces_inf=0,normalize=True):
    x_pts, x_dim = x.shape
    a_pts, a_dim = a.shape
    F_shape = (x_pts,)+f_shape
    F = np.zeros((a_pts,)+F_shape)
    for apt in range(a_pts):
        for xpt in range(x_pts):
            F[apt,xpt] = clip(f=f,x=x[xpt],a=a[apt],replaces_inf=replaces_inf,
                x_bounds=x_bounds,a_bounds=a_bounds,f_bounds=f_bounds,f_shape=f_shape)
    if normalize:
        dX = (x.max()-x.min())/x.shape[0]
        for apt in range(a_pts):
            F[apt] = F[apt]/(dX*F[apt].sum())
    return F

@njit
def jacobian_x(f,x,a,dx,f_shape=(1,),normalize=True):
    x_pts, x_dim = x.shape
    a_pts, a_dim = a.shape
    F_shape = (x_pts,)+f_shape
    G = np.zeros((a_pts,x_dim,)+F_shape)
    for apt in range(a_pts):
        for xdir in range(x_dim):
            g = gradient_x(f=f,x=x,a=a[apt],dx=dx[xdir],f_shape=f_shape,normalize=normalize)
            G[apt,xdir] = g[1]
    return G

@njit
def jacobian_a(f,x,a,da,f_shape=(1,),normalize=True):
    x_pts, x_dim = x.shape
    a_pts, a_dim = a.shape
    F_shape = (x_pts,)+f_shape
    G = np.zeros((a_pts,a_dim,)+F_shape)
    for apt in range(a_pts):
        for adir in range(a_dim):
            g = gradient_a(f=f,x=x,a=a[apt],da=da[adir],f_shape=f_shape,normalize=normalize)
            G[apt,adir] = g[1]
    return G

@njit
def hessian_x(f,x,a,dx,f_shape=(1,),normalize=True):
    x_pts, x_dim = x.shape
    a_pts, a_dim = a.shape
    F_shape = (x_pts,)+f_shape
    H = np.zeros((a_pts,x_dim,x_dim,)+F_shape)
    for apt in range(a_pts):
        hx = np.zeros((x_dim,x_dim,)+F_shape)
        for xdir in range(x_dim):
            for ydir in range(x_dim):
                hx[xdir,ydir] = gradient_xy(f=f,x=x,
                    a=a[apt],dx1=dx[xdir],dx2=dx[ydir],f_shape=f_shape,normalize=normalize)
        H[apt] = hx
    return H

@njit
def hessian_a(f,x,a,da,f_shape=(1,),normalize=True):
    x_pts, x_dim = x.shape
    a_pts, a_dim = a.shape
    F_shape = (x_pts,)+f_shape
    H = np.zeros((a_pts,a_dim,a_dim,)+F_shape)
    for apt in range(a_pts):
        ha = np.zeros((a_dim,a_dim,)+F_shape)
        for adir in range(a_dim):
            for bdir in range(a_dim):
                ha[adir,bdir] = gradient_ab(f=f,x=x,
                    a=a[apt],da1=da[adir],da2=da[bdir],f_shape=f_shape,normalize=normalize)
        H[apt] = ha
    return H

#Compute differential geometry terms

@njit
def christoffel(x,y,jacobian,hessian,bias=nonzero):
    L,H=log_deriv(y=y,jacobian=jacobian,hessian=hessian,bias=bias)
    dx=(x.max()-x.min())/x.size
    g=(H*y).sum(axis=-1)*dx
    dg=np.zeros(g.shape+(g.shape[0],))
    for (i,j,k) in np.ndindex(dg.shape):
        dg[k,i,j] = (y*(-L[i]*L[j]*L[k]+L[i]*H[k,j]+L[j]*H[k,i])).sum(axis=-1)*dx
    ginv = np.linalg.inv(g)
    gamma = np.zeros(dg.shape)
    for (l,i,j) in np.ndindex(dg.shape):
        for k in range(g.shape[0]):
            gamma[l,i,j] += ginv[l,k]*(y*(-0.5*L[i]*L[j]*L[k]+H[i,j]*L[k])).sum(axis=-1)*dx
    return g,ginv,dg,gamma

@njit
def christoffel_numeric(x,y,g,dg):
    dx=(x.max()-x.min())/x.size
    ginv=np.linalg.inv(g)
    gamma = np.zeros(dg.shape)
    for (l,i,j) in np.ndindex(dg.shape):
        for k in range(g.shape[0]):
            gamma[l,i,j] += 0.5*ginv[l,k]*(y*(dg[j,k,i]+dg[i,k,j]-dg[k,i,j])).sum(axis=-1)*dx
    return gamma

@njit
def log_deriv(y,jacobian,hessian,bias=nonzero):
    log_jac = -jacobian/(y+bias)
    log_hess = -hessian/(y+bias)
    for s in np.ndindex(log_hess.shape[:2]):
        log_hess[s] += log_jac[s[0]]*log_jac[s[1]]
    return log_jac,log_hess

@njit
def geodesic(dx,gamma):
    d2x=np.zeros(dx.shape)
    for k in range(dx.shape[0]):
        for i,j in np.ndindex((dx.size,dx.size)):
            d2x[k] -= gamma[k,i,j]*dx[i]*dx[j]
    return d2x
    
#Package into class

class StatisticalManifold:
    def __init__(self,
        f,
        x = None,
        a = None,
        x_bounds = None,
        a_bounds = None,
        f_bounds = None,
        f_shape = None,
        dx = perturbation,
        da = perturbation,
        normalize = True,
        replaces_inf = 0
        ):

        #parse and store keywords
        self.__dict__.update({k:v for k,v in locals().items() if k!='self'})

        #if the input function has any attributes, take them
        for k,v in list(self.__dict__.items()):
            if v is None:
                assert hasattr(f,k), f"Require f.{k}!=None if {k}==None"
                self.__dict__[k] = getattr(f,k)      
          
        #mandate lists of vectors
        while len(self.x.shape)<2:
            self.x=np.expand_dims(self.x,axis=-1)
        while len(self.a.shape)<2:
            self.a=np.expand_dims(self.a,axis=0)

        self.x_pts, self.x_dim = self.x.shape
        self.a_pts, self.a_dim = self.a.shape

        #set perturbations
        self.da=np.array(self.da)
        while len(self.da.shape)<1:
            self.da=np.expand_dims(self.da,-1)
        while self.da.size<self.a_dim:
            self.da=np.append(da,perturbation)

        self.dx_matrix = np.eye(self.x_dim)*self.dx
        self.da_matrix = np.zeros((self.a_dim,self.a_dim))
        for i in range(self.a_dim):
            self.da_matrix[i,i] = self.da[i]
        
        #create jitted function respecting bounds
        x_bounds=self.x_bounds
        a_bounds=self.a_bounds
        f_bounds=self.f_bounds
        f_shape=self.f_shape
        replaces_inf=self.replaces_inf
        @njit
        def _f(x,a):
            return clip(f=f,
                x=x,
                a=a,
                x_bounds=x_bounds,
                a_bounds=a_bounds,
                f_bounds=f_bounds,
                replaces_inf=replaces_inf,
                f_shape=f_shape)

        #store result of computations on given inputs
        self._f = _f
        self.y = self()
        self.jx = self.jacobian('x')
        self.ja = self.jacobian('a')
        self.hx = self.hessian('x')
        self.ha = self.hessian('a')
        self.x = np.squeeze(self.x)
        self.g,self.ginv,self.dg,self.gamma = self.christoffel(y=self.y,jacobian=self.ja,hessian=self.ha)

    def unravel_args(self,x=None,a=None):
        if x is None:
            x = self.x
        if a is None:
            a = self.a
        while len(x.shape)<2:
            x=np.expand_dims(x,axis=-1)
        while len(a.shape)<2:
            a=np.expand_dims(a,axis=0)
        return x,a

    def __call__(self,x=None,a=None):
        x,a=self.unravel_args(x=x,a=a)
        res=submanifold(f=self.f,x=x,a=a,normalize=self.normalize,
            x_bounds=self.x_bounds,a_bounds=self.a_bounds,f_bounds=self.f_bounds,f_shape=self.f_shape)
        res = np.squeeze(res)
        return res
        
    def grad_x(self,axis=0,x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=gradient_x(f=self._f,
            x=x,
            a=a,
            dx=self.dx_matrix[axis],
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)

    def grad_a(self,axis=0,x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=gradient_a(f=self._f,
            x=x,
            a=a,
            da=self.da_matrix[axis],
            f_shape=self.f_shape,
            normalize=self.normalize)
        res=np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)

    def grad_xy(self,axis=(0,0),x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=gradient_xy(f=self._f,
            x=x,
            a=a,
            dx1=self.dx_matrix[axis[0]],
            dx2=self.dx_matrix[axis[1]],
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)

    def grad_ab(self,axis=(0,0),x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=gradient_ab(f=self._f,
            x=x,
            a=a,
            da1=self.da_matrix[axis[0]],
            da2=self.da_matrix[axis[1]],
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)
    
    def jac_x(self,x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=jacobian_x(f=self._f,
            x=x,
            a=a,
            dx=self.dx_matrix,
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)

    def jac_a(self,x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=jacobian_a(f=self._f,
            x=x,
            a=a,
            da=self.da_matrix,
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)

    def hess_x(self,x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=hessian_x(f=self._f,
            x=x,
            a=a,
            dx=self.dx_matrix,
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)
    
    def hess_a(self,x=None,a=None,bounds=None):
        x,a=self.unravel_args(x=x,a=a)
        res=hessian_a(f=self._f,
            x=x,
            a=a,
            da=self.da_matrix,
            f_shape=self.f_shape,
            normalize=self.normalize)
        res = np.squeeze(res)
        if bounds is None:
            return res
        else:
            return np.clip(res,*bounds)

    def gradient(self,x_or_a='x',axis=0,x=None,a=None,bounds=None):
        assert (x_or_a == 'x') or (x_or_a == 'a'), "Require x_or_a == 'x' or x_or_a == 'a'"
        if x_or_a=='x':
            if isinstance(axis,int):
                return self.grad_x(axis=axis,x=x,a=a,bounds=bounds)
            else:
                return self.grad_xy(axis=axis,x=x,a=a,bounds=bounds)
        else:
            if isinstance(axis,int):
                return self.grad_a(axis=axis,x=x,a=a,bounds=bounds)
            else:
                return self.grad_ab(axis=axis,x=x,a=a,bounds=bounds)

    def jacobian(self,x_or_a='a',x=None,a=None,bounds=None):
        assert (x_or_a == 'x') or (x_or_a == 'a'), "Require x_or_a == 'x' or x_or_a == 'a'"
        if x_or_a=='x':
            return self.jac_x(x=x,a=a,bounds=bounds)
        else:
            return self.jac_a(x=x,a=a,bounds=bounds)

    def hessian(self,x_or_a='a',x=None,a=None,bounds=None):
        assert (x_or_a == 'x') or (x_or_a == 'a'), "Require x_or_a == 'x' or x_or_a == 'a'"
        if x_or_a=='x':
            return self.hess_x(x=x,a=a,bounds=bounds)
        else:
            return self.hess_a(x=x,a=a,bounds=bounds)

    def plot(self,a=None,x=None,y=None,ja=None,jx=None,ha=None,hx=None,xlim=None,ylim=None):
        if a is None:
            a=self.a[0]
        if x is None:
            x=self.x
        if y is None:
            y=self.y
        if ja is None:
            ja=self.ja
        if jx is None:
            jx=self.jx
        if ha is None:
            ha=self.ha
        if hx is None:
            hx=self.hx
        plt.plot(x,y,label=r'$f(x'+f',{a[0]},{a[1]}'+r')$')
        for j in range(ja.shape[0]):
            s=r'$\partial_{a_{'+f'{j}'+r'}}f$'
            plt.plot(x,ja[j],label=s,linestyle='dashed')
        for h in np.ndindex(ha.shape[:2]):
            if h[0]!=h[1]:
                s=r'$\partial_{a_{'+f'{h[0]}'+r'}}\partial_{a_{'+f'{h[1]}'+r'}}f$'
            else:
                s=r'$\partial^{2}_{a_{'+f'{h[0]}'+r'}}f$'
            plt.plot(x,ha[h],label=s,linestyle='dashdot')
        plt.plot(x,jx,label=r'$\partial_{x}f$',linestyle='dotted')
        plt.plot(x,hx,label=r'$\partial^{2}_{x}f$',linestyle='dotted')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.42),
                ncol=3, fancybox=True, shadow=True)
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title(str(self.f.__name__)+' and its derivatives',y=1.4)
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.show()

    def metric(self,a=None,x=None):
        x,a=self.unravel_args(x=x,a=a)
        dx=(x.max()-x.min())/x.size
        h=self.hess_a(x=x,a=a)
        j=self.jac_a(x=x,a=a)
        y=self(x=x,a=a)
        _,lh=log_deriv(y=y,jacobian=j,hessian=h)
        g=(lh*y).sum(axis=-1)*dx
        return g

    def positive_definite(self,g):
        try:
            np.linalg.cholesky(g)
            return True
        except:
            print("Warning: Non-Riemannian metric; g is not positive-definite.")
            return False
    
    def inverse_metric(self,g=None,a=None,x=None):
        if g is None:
            x,a=self.unravel_args(x=x,a=a)
            g=self.metric(self,a=a,x=x)
        ginv = np.linalg.inv(g)
        return ginv

    def metric_jacobian_numeric(self,a=None,da=2.0*perturbation,x=None):
        x,a=self.unravel_args(x=x,a=a)
        a_dim = a.size
        d=da
        da=np.eye(a_dim)*(d)
        dg=np.zeros((a_dim,a_dim,a_dim))
        for adir in range(a_dim):
            gup=self.metric(a=a+da[adir],x=x)
            gdown=self.metric(a=a-da[adir],x=x)
            dg[adir]=(gup-gdown)/(2.0*d)
        return dg

    def christoffel(self,a=None,x=None,y=None,jacobian=None,hessian=None,bias=nonzero):
        x,a=self.unravel_args(x=x,a=a)
        if y is None:
            y=self(x=x,a=a)
        if jacobian is None:
            jacobian=self.jac_a(x=x,a=a)
        if hessian is None:
            hessian=self.hess_a(x=x,a=a)
        return christoffel(x=x,y=y,jacobian=jacobian,hessian=hessian,bias=bias)

    def geodesic_equation(self,a=None,da=None,T=100):
        if a is None:
            a=self.a[0]
        if da is None:
            da=self.da_matrix[0]
        dt=float(1/T)
        position=np.zeros((T,)+a.shape)
        position[0]=a
        velocity=np.zeros((T,)+a.shape)
        velocity[0]=da

        def F(pos,vel):
            g,ginv,dg,gamma = self.christoffel(a=pos)
            acceleration = geodesic(dx=vel,gamma=gamma)
            return acceleration

        for i in range(T-1):
            position[i+1] = position[i] + velocity[i]*dt
            velocity[i+1] = velocity[i] + F(position[i],velocity[i])*dt

        return position

    def geodesics(self,a=None,N=16,T=25,R=1):
        if a is None:
            a=self.a[0]
        dA=np.array([[np.cos(2*np.pi*i/N),np.sin(2*np.pi*i/N)] for i in range(N)])
        pos=np.zeros((N,T,)+a.shape)
        for i in range(N):
            da=dA[i]
            S=(da.size,da.size)
            #normalize using g_ij*dx^i*dx^j; inner product on curved manifold
            norm=np.array([self.g[i,j]*da[i]*da[j] for (i,j) in np.ndindex(S)]).sum()
            norm=np.sqrt(norm)
            da=R*da/norm
            pos[i]=self.geodesic_equation(a=a,da=da,T=T)
        return pos


    def plot_geodesics(self,
        pos=None,
        title=r'Geodesics of the Normal Distribution $N(\mu,\sigma)$',
        xlabel=r'$\mu$',
        ylabel=r'$\sigma$',
        fontsize=16,
        ):

        if pos is None:
            pos=self.geodesics()

        fig, ax = plt.subplots()
        N,T,D = pos.shape #number of tangents, steps, dimensions
        #get as stacked D-dim points
        y = pos.reshape(N*T,D).T
        #label colors according to angle
        colors = np.zeros((N,T)) #for sequential steps along path
        qolors = np.zeros((T,N)) #for instantaneous hypersurfaces
        for i in range(N):
            colors[i]+=i/N
        for j in range(T):
            qolors[j]=np.array([i/N for i in range(N)])
        colors = colors.reshape(N*T)
        ax.scatter(*y,c=colors,cmap='hsv')
        qolors=qolors*2*np.pi
        for t in range(T): #plot level sets; connect instantaneous times
            #append origin for complete rotation
            u=np.append(pos[:,t,0],pos[0,t,0])
            v=np.append(pos[:,t,1],pos[0,t,1])
            points=np.array([u, v]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(qolors[t].min(), qolors[t].max())
            lc = LineCollection(segments, cmap='hsv', norm=norm)
        # Set the values used for colormapping
            lc.set_array(qolors[t])
            lc.set_linewidth(0.5)
            line = ax.add_collection(lc)
        #break into octants defined by the orthogonal axes
        #and the lines y=x and y=-x
        for i in range(N):
            if i%4==0: #original axes
                linewidth=1.
                ls='-'
                color='black'
            elif i%4==2: #y=+-x
                linewidth=1.
                ls='--'
                color='white'
            else: #inbetween quadrants
                linewidth=1
                ls='-.'
                color='gray'
            ax.plot(pos[i,:,0],pos[i,:,1],color=color,linewidth=linewidth,ls=ls)

        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        ax.set_title(title)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        fig.colorbar(line, ax=ax, label=f'Angle of initial unit tangent vector in radians',
            orientation='vertical',location='left',shrink=1)
        ax.grid(True)
        force_aspect(ax,1)