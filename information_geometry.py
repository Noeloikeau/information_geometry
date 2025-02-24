"""
author: Noeloikeau Charlot
date: 2/24/2025
version: 2.0

Implements methods used to calculate information geometry quantities.
See LICENSE, README, and statistical_manifolds.ipynb for context.
Full re-write using njit parallelization of tangent vectors.
"""

import numpy as np
from numba import njit, prange
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
from matplotlib.collections import LineCollection

# Constants #
delta = 1e-5
x_pts = 1000
nonzero = np.nextafter(0., 1.)
nearinf = 1e100
isnan = 0.
isinf = nearinf

# Intervals #
reals = np.array([-np.inf,np.inf])
unit_interval = np.array([0.,1.])
open_interval = np.array([delta,1.-delta])
five_sigma = np.array([-5.,5.])

# Functions #
@njit
def normal_distribution(x,a):
    """
    Normalized Gaussian distribution over input array `x` 
    with a[0] = mean  &  a[1] = standard deviation.
    """
    y = np.exp(-0.5*((x-a[0])/a[1])**2)/(a[1]*np.sqrt(2.*np.pi))
    y = y / np.sum(y)
    return y

normal_distribution.x = np.linspace(*five_sigma,x_pts)
normal_distribution.a = np.array([0.,1.])

@njit
def beta_distribution(x,a):
    y = (x**(a[0]-1.))*(1.-x)**(a[1]-1.)
    y = y / (np.sum(y)*(x.max()-x.min()/x.shape[0])) 
    return y

beta_distribution.x = np.linspace(*open_interval,x_pts)
beta_distribution.a = np.array([2.,2.])

@njit
def clip(f,x,a,
         f_bounds = reals,
         x_bounds = reals,
         a_bounds = reals,
         normalize = False):
    """
    Restricts the domain and range of y = f(x,a).

    f (CPUDispatcher): jitted function with signature f(x,a).
    x (np.ndarray): matrix or vector of variables.
    a (np.ndarray): matrix or vector of parameters.
    f_/x_/a_bounds (np.ndarray) : 2-dim vector(s) of inclusive bounds.
    
    returns y/np.sum(y) if normalize is True; else y
    """

    for i in range(x.shape[0]):
        if x[i] < x_bounds[0]:
            x[i] = x_bounds[0]
        elif x[i] > x_bounds[1]:
            x[i] = x_bounds[1]
    
    for j in range(a.shape[0]):
        if a[j] < a_bounds[0]:
            a[j] = a_bounds[0]
        elif a[j] > a_bounds[1]:
            a[j] = a_bounds[1]

    y = f(x,a)
    
    for s in range(y.shape[0]):
        if y[s] < f_bounds[0]:
            y[s] = f_bounds[0]
        elif y[s] > f_bounds[1]:
            y[s] = f_bounds[1]

    if normalize:
        y /= np.sum(y)

    return y

def clipped(f,
            f_bounds = None,
            x_bounds = None,
            a_bounds = None,
            normalize = False):
            """
            Partially evaluates `clip` over `f`
            if any bounds are provided.
            Returns a new callable `f_clipped`.
            """
            if (f_bounds is None) and (x_bounds is None) and (a_bounds is None) \
                and (normalize is False):
            
                return f

            else:
                f_bounds = f_bounds if (f_bounds is not None) else reals
                x_bounds = x_bounds if (x_bounds is not None) else reals
                a_bounds = a_bounds if (a_bounds is not None) else reals

                @njit
                def f_clipped(x,a):
                    return clip(f,x,a,
                                f_bounds=f_bounds,x_bounds=x_bounds,a_bounds=a_bounds,
                                normalize=normalize)
                f_clipped.__name__ = f.__name__

                return f_clipped

@njit
def gradient(f,x,a,h,f_shape=()):
    """
    Returns the function y = f(x,a) and its Jacobian and Hessian
    with respect to the parameter `a`. Uses symmetric 2nd-order 
    finite differences that call the function the minimum number 
    of times required, equal to  1 + 4 * a_dim**2  calls, 
    where a_dim is the size of `a`. 

    Args:
        f (CPUDispatcher): jitted function with signature f(x,a).
        x (array): coordinate matrix of shape (x_pts, x_dim).
        a (array): parameter vector of shape (a_dim,).
        h (array): differential matrix of shape (a_dim, a_dim).
        f_shape (tuple): output shape, if function is not scalar.

    The parameter vector is perturbed differentially by the vectors of `h`,
    and the calculation proceeds as differences of f(x,a+h[i]+h[j]) over (i,j).
    Thus `h` acts as a differential operator specifying the coordinate system
    in which the derivative is being calculated, indexed by the basis of `a`.
    
    Returns:
        y (array): `f` evaluated over `x` at `a`. Shape is (x_pts)+f_shape.
        jacobian (array): partial derivatives of `f` w.r.t `h` in `a`'s basis.
                Shape is: (a_dim, x_pts)+f_shape. 
        hessian (array): cross partials of `f` w.r.t `h` in `a`'s basis. 
                Shape is: (a_dim, a_dim, x_pts)+f_shape. 
    
    Example use:
        f = normal_distribution     #a[0]=mean, a[1]=std
        x = np.linspace(-5,5,1000)
        a = np.array([0.,1.])
        h = np.array([[1e-5,0],[0,1e-5]])
        y,J,H = gradient(f,x,a,h,f_shape=())

    See also:
    https://math.stackexchange.com/questions/2931510/cross-derivatives-using-finite-differences
    https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
    """
    a_dim = a.shape[0]     # parameter vector
    x_pts = x.shape[0]     # coordinate matrix
    jacobian = np.zeros((a_dim,x_pts)+f_shape)
    hessian = np.zeros((a_dim,a_dim,x_pts)+f_shape)
    # first function evaluation [out of 1 + 4 * a_dim**2]
    y = f(x=x,a=a)
    # calculate (-2,-1,1,2) perturbations along each axis
    # begin with jacobian and hessian diagonal 
    for i in range(a_dim): 
        da1 = h[i]                      # differential vector
        d1 = np.sqrt(np.dot(da1,da1))   # differential norm
        fp1 = f(x=x, a=a+da1)  
        fm1 = f(x=x, a=a-da1)
        jacobian[i] = (fp1-fm1)/(2.*d1)
        fp2 = f(x=x, a=a+2.*da1)         
        fm2 = f(x=x, a=a-2.*da1)
        # equality of cross terms on the diagonal f(x,a+da-da) = f(x,a) 
        # hence diagonal requires only 2nd differences       
        hessian[i,i] = (fp2+fm2-2.*y)/(4.*(d1**2))
        # hessian off-diagonal 
        for j in range(a_dim):
            if j!=i:
                da2 = h[j]
                d2 = np.sqrt(np.dot(da2,da2)) 
                f00 = f(x=x, a=a-da1-da2)
                f01 = f(x=x, a=a-da1+da2)
                f10 = f(x=x, a=a+da1-da2)
                f11 = f(x=x, a=a+da1+da2)
                hessian[i,j] = (f00+f11-f01-f10)/(4.*d1*d2)
    return y,jacobian,hessian

@njit
def grxdient(f,x,a,h,f_shape=()):
    """
    Identical to the `gradient` function except
    `x` is replaced with `a` in the sense that
    `h` differentiates with respect to the 
    coordinate matrix `x` instead of the
    parameter vector `a`.
    Example use: # note x.shape = (1000,1)
        f = normal_distribution     #a[0]=mean, a[1]=std
        x = np.atleast_2d(np.linspace(-5,5,1000)).T
        a = np.array([0.,1.])
        h = np.array([[1e-5]])
        y,J,H = grxdient(f,x,a,h,f_shape=(1,)) # note (1,) output shape
    """
    x_pts, x_dim = x.shape     
    jxcobixn = np.zeros((x_dim,x_pts)+f_shape)
    hessixn = np.zeros((x_dim,x_dim,x_pts)+f_shape)
    y = f(x=x,a=a)
    for i in range(x_dim): 
        dx1 = h[i]
        d1 = np.sqrt(np.dot(dx1,dx1)) 
        fp1 = f(x=x+dx1,a=a)  
        fm1 = f(x=x-dx1,a=a)
        jxcobixn[i] = (fp1-fm1)/(2.*d1)
        fp2 = f(x=x+2.*dx1,a=a)  
        fm2 = f(x=x-2.*dx1,a=a)
        hessixn[i,i] = (fp2+fm2-2.*y)/(4.*(d1**2))
        for j in range(x_dim):
            if j!=i:
                dx2 = h[j]
                d2 = np.sqrt(np.dot(dx2,dx2)) 
                f00 = f(x=x-dx1-dx2,a=a)
                f01 = f(x=x-dx1+dx2,a=a)
                f10 = f(x=x+dx1-dx2,a=a)
                f11 = f(x=x+dx1+dx2,a=a)
                hessixn[i,j] = (f00+f11-f01-f10)/(4.*d1*d2)
    return y,jxcobixn,hessixn

@njit
def log_deriv(y,jacobian,hessian,bias=nonzero):
    """
    Transforms the Jacobian and Hessian of a function y
    into the corresponding derivatives of -log(y).
    """
    log_jac = -jacobian/(y+bias)
    log_hess = -hessian/(y+bias)
    for s in np.ndindex(log_hess.shape[:2]):
        log_hess[s] += log_jac[s[0]]*log_jac[s[1]]
    return log_jac,log_hess

@njit
def remove_singularities(g,isinf=isinf,isnan=isnan):
    """
    Return a copy of `g` where inf and NaN 
    are replaced by `isinf` `isnan`.
    """
    z = np.zeros(g.shape,dtype=g.dtype)
    for s in np.ndindex(z.shape):
        if np.isinf(g[s]):
            if g[s]>0:
                z[s] = isinf
            else:
                z[s] = -isinf
        elif np.isnan(g[s]):
            z[s] = isnan
        else:
            z[s] = g[s]
    return z

@njit
def inverse_matrix(g,isinf=isinf,isnan=isnan):
    """
    Calculate the inverse matrix of `g`
    even if it is singular.
    """
    try: # get inverse matrix
        ginv = np.linalg.inv(g)
    except: # try pseudoinverse
        try: 
            ginv = np.linalg.pinv(g)
        except: #singular to machine precision; remove singularities
            ginv = np.linalg.pinv(remove_singularities(g,isinf=isinf,isnan=isnan))
    return ginv

@njit
def alpha_connection(y,jacobian,hessian,alpha=0.,bias=nonzero,isinf=isinf,isnan=isnan):
    """
    Calculate the Fisher information metric and 
    its derivatives, inverse, and affine connection
    given by the Christoffel symbols of the 2nd kind with 
    a control parameter `alpha` tuning the first-order derivatives.
    """
    #log jacobian and hessian
    L,H = log_deriv(y=y,jacobian=jacobian,hessian=hessian,bias=bias)
    # fisher information matrix analytic results
    g = (H*y).sum(axis=-1)
    dg = np.zeros(g.shape+(g.shape[0],))
    for (i,j,k) in np.ndindex(dg.shape):
        dg[k,i,j] = (y*(-L[i]*L[j]*L[k]+L[i]*H[k,j]+L[j]*H[k,i])).sum(axis=-1)
    # inverse metric
    ginv = inverse_matrix(g,isinf=isinf,isnan=isnan)
    # connection / christoffel symbols
    gamma = np.zeros(dg.shape)
    for (l,i,j) in np.ndindex(dg.shape):
        for k in range(g.shape[0]):
            gamma[l,i,j] += ginv[l,k]*(
                y*(-0.5*(1.+alpha)*L[i]*L[j]*L[k]+H[i,j]*L[k])
                ).sum(axis=-1)
    # note quantities have not been normalized by dx in sums over axis=-1
    # this is optionally done in the `StatisticalManifold` class by selecting PDF=True
    return g,dg,ginv,gamma

@njit
def inner_product(g,u,v):
    """
    Inner product of vectors `u` and `v`
    given the metric matrix `g`.
    """
    S = (u.size,v.size)
    dots = np.zeros(S,dtype=np.float_)
    for (i,j) in np.ndindex(S):
        dots[i,j] = g[i,j]*u[i]*v[j]
    return dots.sum()

@njit
def geodesic_equation(gamma,dx):
    """
    Differentiates the tangent vector `dx`
    given the metric connection /
    Christoffel symbols of the 2nd kind `gamma`.
    """
    d2x = np.zeros(dx.shape)
    for k in range(dx.shape[0]):
        for i,j in np.ndindex((dx.size,dx.size)):
            d2x[k] -= gamma[k,i,j]*dx[i]*dx[j]
    return d2x

@njit(parallel=True)
def tangent_space(f,x,a,h,tangents,
                  T = 1, f_shape = (),
                  function_and_derivatives = gradient,
                  metric_and_connection = alpha_connection
                  ):
    """
    Computes the first `T` points of the geodesic equations
    for the function `f` at point `a` on the manifold, given 
    matrices of initial unit vectors `tangents` and 
    coordinate differentials `h`. Returns all results of 
    `function_and_derivatives` & `metric_and_connection` 
    as well as new arrays `position` and `velocity`
    containing solutions to geodesic paths in the basis of `a`.   

    Args:
        f (CPUDispatcher): jitted function with signature f(x,a) = y.
        x (array): coordinate matrix of shape (x_pts, x_dim).
        a (array): parameter vector of shape (a_dim,).
        h (array): differential matrix of shape (a_dim, a_dim).
        tangents (array): matrix of unit tangent vectors with shape (N,a_dim).
        f_shape (tuple): output shape, if function is not scalar.
        T (int): number of geodesic steps to take; defaults to none.
        function_and_derivatives (CPUDispatcher):   signature (f,x,a,h,f_shape) 
                                                    returns (y,jacobian,hessian).
        metric_and_connection (CPUDispatcher):      signature (y,jacobian,hessian)
                                                    returns (metric,dmetric,invmetric,connection).
    
    Returns tuple of arrays of shape (...,x_pts)+f_shape:
        (y,jacobian,hessian,metric,dmetric,invmetric,connection,position,velocity)
    
    Example use:
        f = normal_distribution     #a[0]=mean, a[1]=std
        x = np.linspace(-5,5,10000)
        a = np.array([0.,1.])
        h = np.array([[1e-4,0],[0,1e-4]])
        N, T = 16, 100
        # disk of N input tangent vectors, unnormalized
        tangents = np.array([[np.cos(2*np.pi*i/N),np.sin(2*np.pi*i/N)] for i in range(N)])
        y,J,H,g,dg,ginv,gamma,position,v = tangent_space(f=f,x=x,a=a,h=h,tangents=tangents,T=T)
        for i in range(position.shape[0]):                  #initial tangents
            plt.scatter(position[i,:,0],position[i,:,1])    #geodesic paths
    """
    x_pts = x.shape[0]  # evaluated function points (not on manifold)
    N,a_dim = tangents.shape  # number of manifold tangent vectors & their dimension
    dt = float(1./T)    # geodesic affine parameter step size
    # function and its derivatives
    function = np.zeros((N,T,x_pts))        
    jacobian = np.zeros((N,T,a_dim,x_pts))        
    hessian = np.zeros((N,T,a_dim,a_dim,x_pts)) 
    # inner product and christoffel symbols
    metric = np.zeros((N,T,a_dim,a_dim))
    invmetric = np.zeros((N,T,a_dim,a_dim))
    dmetric = np.zeros((N,T,a_dim,a_dim,a_dim))
    connection = np.zeros((N,T,a_dim,a_dim,a_dim))
    # manifold coordinates of geodesics
    position = np.zeros((N,T,a_dim))
    velocity = np.zeros((N,T,a_dim))
    # initialize metric information at point a
    y,J,H = function_and_derivatives(f,x,a,h,f_shape)
    g,dg,ginv,gamma = metric_and_connection(y,J,H)
    for i in range(N):
        function[i,0] = y
        jacobian[i,0] = J
        hessian[i,0] = H
        metric[i,0] = g
        dmetric[i,0] = dg
        invmetric[i,0] = ginv
        connection[i,0] = gamma
    # loop over tangent vectors radiating from point a
    for i in prange(N): # `prange` parallelizes loop
        position[i,0] = a
        # normalize tangent vector by inner product on curved space
        velocity[i,0] = tangents[i] / np.sqrt(inner_product(metric[i,0],tangents[i],tangents[i]))
        # propagate using Euler method
        for t in prange(1,T):
            position[i,t] = position[i,t-1] + velocity[i,t-1]*dt
            # recompute metric information
            function[i,t],jacobian[i,t],hessian[i,t] = function_and_derivatives(f,x,position[i,t-1],h,f_shape)
            metric[i,t],dmetric[i,t],invmetric[i,t],connection[i,t] = metric_and_connection(
                function[i,t],jacobian[i,t],hessian[i,t])
            # solve dynamical equations
            acceleration = geodesic_equation(connection[i,t],velocity[i,t-1]) 
            velocity[i,t] = velocity[i,t-1] + acceleration*dt

    return function,jacobian,hessian,metric,dmetric,invmetric,connection,position,velocity

# Plotting helpers
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

def color_line(x,y,colors,ax,lw=1):
    """
    Breaks curve into segments and colors each segment.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)
    return line

# Classes #
class StatisticalManifold:
    """
    Wrapper for the `tangent_space` function.
    Saves the function outputs as attributes.
    """
    blacklist = ['self']
    a_gradients = [gradient]
    x_gradients = [grxdient]
    def __init__(self,f,x,a,
                 h=delta,tangents=1,T=1,f_shape = (),
                 function_and_derivatives = gradient,
                 metric_and_connection = alpha_connection,
                 squeeze = False,
                 density = True,
                 f_bounds = None,
                 x_bounds = None,
                 a_bounds = None,
                 normalize = False
                 ):
        
        self.__dict__.update({k:v for k,v in locals().items() if k not in StatisticalManifold.blacklist})
        self.f = clipped(f,f_bounds,x_bounds,a_bounds,normalize)
        self.get_dim()
        self.get_h()
        self.get_tangents()
        self.dx = (self.x.max()-self.x.min())/self.x.shape[0]
        self.__call__()
    
    def get_dim(self):
        if self.function_and_derivatives in StatisticalManifold.a_gradients:
            self.dim = self.a.shape[-1]
        elif self.function_and_derivatives in StatisticalManifold.x_gradients:
            while len(self.x.shape)<2:
                self.x=np.expand_dims(self.x,axis=-1)
            self.dim = self.x.shape[-1]
        if self.dim<1:
            self.dim = 1
    
    def get_h(self):
        #interpret number as scaled identity matrix
        if isinstance(self.h,(float,int)):
            self.h = np.eye(self.dim,dtype=type(self.h))*self.h

    def get_tangents(self):
        #interpret number as # of tangent rays in a plane
        if isinstance(self.tangents,(int,float)):
            N = round(self.tangents)
            n = self.dim
            if n==1:
                self.tangents = np.array([[np.cos(2*np.pi*i/N)] for i in range(N)])
            else: #default to hyperplane that is nonzero only along 0 and -1 axes
                self.tangents = np.array([[np.cos(2*np.pi*i/N)]+[0]*(n-2)+[np.sin(2*np.pi*i/N)] 
                    for i in range(N)])
        self.N = self.tangents.shape[0]

    def __call__(self):
        res = tangent_space(
            f=self.f,x=self.x,a=self.a,
            h=self.h,
            tangents=self.tangents,
            T = self.T, 
            f_shape = self.f_shape,
            function_and_derivatives = self.function_and_derivatives,
            metric_and_connection = self.metric_and_connection)
            
        if self.squeeze:
            res = map(np.squeeze, res)         
        
        self.function,self.jacobian,self.hessian,self.metric,self.dmetric,self.invmetric, \
            self.connection,self.position,self.velocity = res
        
        if self.density:
            self.function /= self.dx
            self.jacobian /= self.dx
            self.hessian /= self.dx
            #self.invmetric /= self.dx
            #self.metric *= self.dx
            #self.dmetric *= self.dx
            #self.connection *= self.dx

    def plot_function(self,i=0,t=0,xlim=None,ylim=None):
        title = self.f.__name__+' and its derivatives'
        x = self.x
        y = self.function[i,t]
        ja = self.jacobian[i,t]
        ha = self.hessian[i,t]
        a = self.position[i,t]
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
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.42),
                ncol=3, fancybox=True, shadow=True)
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title(title, y=1.4)
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.show()
    
    def plot_geodesic(self,
             i=0,
             cmap=plt.cm.jet,
             style='dark_background',
             show=True,
             ax=None,
             fig=None,
             figsize=(8,4),
             fontsize=15,
             x2lim=[0,8],
             y2lim=[0,8]
             ):
        if (fig is None) or (ax is None):
            fig,ax = plt.subplots(1,2,figsize=figsize)
        #ax1: x,f(x,a) ; ax2: a0,a1
        ax1,ax2 = ax
        ax1.clear()
        plt.style.use(style)
        colors = cmap(np.linspace(0,1,self.T))
        artists = []
        clabel = r'$t$'
        x1label = r'$x$'
        y1label = 'f'+r'$(x,a(t))$'
        for t in range(self.T):
            artists+=[ax1.plot(self.x,self.function[i,t],color=colors[t])]
        ax1.set_xlabel(x1label,fontsize=fontsize)
        ax1.set_ylabel(y1label,fontsize=fontsize)
        x2label = r'$a_{0}$'
        y2label = r'$a_{1}$'
        ax2.set_xlim(*x2lim)
        ax2.set_ylim(*y2lim)
        ax2.set_xlabel(x2label,fontsize=fontsize)
        ax2.set_ylabel(y2label,fontsize=fontsize)
        cmap2=cmap(np.linspace(0,1,self.T))
        cl=color_line(self.position[i,:,0],self.position[i,:,1],cmap2,ax2,1.)
        artists+=[list(np.ravel(cl))]
        if show:
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap),ax=ax2,label=clabel)
            plt.show()
        else:
            return np.ravel(list(artists))
        
    def animate(self,fname='geodesic.mp4',
                frames=360,fps=20,dpi=100,
                figsize=(13,6),
                fontsize = 18,
                x2lim = [0,8],
                y2lim = [0,8],
                cmap=plt.cm.jet
                ):
        animation_kwargs = dict(
            interval=1,
            blit=False,
            frames=frames,
            repeat=False,
            cache_frame_data = False,
            save_count=frames
        )
        save_kwargs = dict(
            filename=fname,
            dpi=dpi,
            writer=FFMpegWriter(fps=fps) if fname.split('.')[-1] == '.mp4' else (
                PillowWriter(fps=fps) if fname.split('.')[-1] == '.gif' else None)
        )
        matplotlib.rcParams.update({'font.size': fontsize})
        fig,ax = plt.subplots(1,2,figsize=figsize)
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap),ax=ax[1],label=r'$t$')
        plt.gcf().set_size_inches(figsize)    
        cmap=cmap(np.linspace(0,1,self.T)) 
        def _animate(i):
            title = f'Geodesic path at {i}'+r'$\degree$'
            artists = self.plot_geodesic(i=i,ax=ax,fig=fig,show=False,
                                        figsize=figsize,fontsize=fontsize,
                                        x2lim=x2lim,y2lim=y2lim)
            fig.suptitle(title,fontsize=fontsize)
            return artists
            
        ani = FuncAnimation(fig, _animate, **animation_kwargs)
        ani.save(**save_kwargs)
