'''
Created on 31.7.2012

Contains functions used for plotting data. I use the matplotlib (pylab) package, which has extensive documentation
http://matplotlib.org/

@author: kfinn
'''
import pylab as p
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d #used for 3d axese
p.rcParams.update({'font.size': 20}) # makes the dfault text size larger
p.rc('text', usetex=True) #allows the use of latex expressions for labels

def get3dax(): #returns a pylab 3d axis instance labelled with x, y and z
    f=p.figure()
    out=f.gca(projection='3d')
    out.set_xlabel('x')
    out.set_ylabel('y')
    out.set_zlabel('z')
    return out

def makegrid(x,y,z,log=True, method='nearest',res=False, **kwargs):
    '''turns a list of coordinates into a format that can be plotted by, e.g. wireframe or imshow
    options are:
    log: treats the data as on a log scale
    method: the method for interpolation, options are nearest, bilinear or None
    res, xres, yres: the resolution of the grid
    form required is:
    x=[[x1,x1,x1...]
        [x2,x2,x2...]
        ..]
    y=[[y1,y2,y3...]
        [y1,y2,y3...]
        ..]
    z=[[z(x1,y1),z(x1,y2),z(x1,y3)...]
        [z(x2,y1),z(x2,y2),z(x2,y3)...]
        ..]'''
    if not res:#default resolution
        if log:
            res=2
        else:
            res=100j
    if 'xres' in kwargs:
        xres=kwargs['xres']
    else:
        xres=res
    if 'yres' in kwargs:
        yres=kwargs['yres']
    else:
        yres=res
    xmax=max(x)
    xmin=min(x)
    ymax=max(y)
    ymin=min(y)
    points=[]
    for i in range(max([len(x),len(y)])):#makes sure x, y and z have the same number of values
        try:
            points.append([x[i],y[i]])
        except IndexError:
            print 'ERROR: x and y do not have the same dimensions.'
            return False
    if len(z)!=len(y):
        print 'ERROR: z does not have the same dimensions as x and y.'
        return False
    points=np.array(points)
    values=np.array(z)
    if log:
        i=xmin
        x=[]
        y=[]
        while i<=xmax:
            j=ymin
            tempx=[]
            tempy=[]
            while j<=ymax:
                tempx.append(i)
                tempy.append(j)
                j*=yres
            x.append(tempx)
            y.append(tempy)
            i*=xres
        x=np.array(x)
        y=np.array(y)
    else:
        x,y=np.mgrid[xmin:xmax:xres,ymin:ymax:yres]
    z=griddata(points,values,(x,y),method=method)
    if log: #3d axese cannot handle log scaled plots so this is done manually
        x=np.log10(x)
        y=np.log10(y)
    return (x,y,z)

def colourmap(x,y,z,log=True, method='nearest',res=False):#produces a colour map image from a random selection of points in greyscale
    methods={'nearest':'nearest','linear':'bilinear','cubic':None}
    x,y,image=makegrid(x,y,z,log=log,method=method,res=False)
    xmax=max(x.flatten())
    xmin=min(x.flatten())
    ymax=max(y.flatten())
    ymin=min(y.flatten())
    if method not in methods.keys():
        method=None
    else:
        method=methods[method]
    p.imshow(image.T,extent=(xmin,xmax,ymin,ymax),cmap=p.cm.gray,origin='lower',interpolation=method) #imshow inverts some axese so we need the .T for it to be the right way round
    return True

def wireframe(x,y,z,log=True,method='linear'): #creates a wireframe plot from a set of points using makegrid
    x,y,z=makegrid(x,y,z,log=log,method=method)
    ax=get3dax()
    if log:
        x=np.log10(x)
        y=np.log10(y)
        z=np.log10(z)
        ax.set_xlabel('log x')
        ax.set_ylabel('log y')
        ax.set_zlabel('log z')
    ax.plot_wireframe(x,y,z)
    return ax #returns ax so that it is easy to update the plot

def fwireframe(f,x_range,y_range):
    '''creates a wireframe 3d plot from a function f(x,y).
    This is probably more useful than wireframe and is much more reliable.
    It is currently set up for log scaled plots but could easily be changed to accomodate linear.
    x_range and y_range are the ranges IN LOG SPACE i.e. 10**x_range[0]<x<10**x_range[1]'''
    X=[]
    Y=[]
    Z=[]
    for i in np.linspace(x_range[0],x_range[1],100): #resolution is 100 by 100 but this could be changed by editing this line and the next for loop
        tempx=[]
        tempy=[]
        tempz=[]
        for j in np.linspace(y_range[0],y_range[1],100):
            x=10.0**i
            y=10.0**j
            z=f(x,y)
            tempx.append(i)
            tempy.append(j)
            tempz.append(np.log10(z))
        X.append(tempx)
        Y.append(tempy)
        Z.append(tempz)
    ax=get3dax()
    ax.set_xlabel('log x')
    ax.set_ylabel('log y')
    ax.set_zlabel('log z')
    ax.plot_wireframe(X,Y,Z)
    return ax #returns ax so that it is easy to update the plot

def fcolourmap(f,x_range,y_range):
    '''creates a colourmap plot from a function f(x,y) with a colourbar to show the values.
    This is probably more useful than colourmap and is much more reliable.
    It is currently set up for log scaled plots but could easily be changed to accomodate linear.
    x_range and y_range are the ranges IN LOG SPACE i.e. 10**x_range[0]<x<10**x_range[1]'''
    X=[]
    Y=[]
    Z=[]
    for i in np.linspace(x_range[0],x_range[1],100): #resolution is 100 by 100 but this could be changed by editing this line and the next for loop
        tempx=[]
        tempy=[]
        tempz=[]
        for j in np.linspace(y_range[0],y_range[1],100):
            x=10.0**i
            y=10.0**j
            z=f(x,y)
            tempx.append(i)
            tempy.append(j)
            tempz.append(np.log10(z))
        X.append(tempx)
        Y.append(tempy)
        Z.append(tempz)
    x=np.array(X).flatten()
    y=np.array(Y).flatten()
    extent=[min(x),max(x),min(y),max(y)]
    z=np.array(Z).T
    im=np.zeros_like(z)
    for j in range(len(z)): #p.imshow inverts some of the axese so this section is to make sure the plot is the right way round
        for i in range(len(z[j])):
            im[len(z)-j-1][i]=z[j][i]
    p.imshow(im,extent=extent,aspect='equal')
    p.colorbar()
    return im

def fplot(f,x_range, step): #plots a 1d function in log space. 10**x_range[0]<x<10**x_range[1]
    x=[]
    y=[]
    i=10.0**x_range[0]
    while i<10.0**x_range[1]:
        x.append(i)
        y.append(f(i))
        i*=step
    p.plot(x,y)
    p.loglog()
    return (x,y)

def label(x,y): #labels the x and y axese, treating the arguments as latex code (in math mode)
    p.xlabel(r'\huge{$%s$}' %x)
    p.ylabel(r'\huge{$%s$}' %y)
    return True

def log_labels(ax=False):
    '''this function attempts to convert the labels on a 3d plot from the value of i=log(x) to the more traditional 10^i.
    It is currently not working but was abandoned as I no longer needed to produce 3d plots'''
    locs=[]
    if ax:
        locs.append(ax.get_xticks()[0])
        locs.append(ax.get_yticks()[0])
        try:
            locs.append(ax.get_zticks()[0])
        except:
            pass
    else:
        locs.append(p.xticks()[0])
        locs.append(p.yticks()[0])
    labels=[]
    print locs
    for loc in locs:
        lab=[]
        for l in loc:
            if l==int(l):
                lab.append('\huge{$10^{%d}$}' %int(l))
            else:
                lab.append('')
        labels.append(lab)
    if ax:
        ax.set_xticklabels(labels[0])
        ax.set_yticklabels(labels[1])
        try:
            ax.set_zticklabels(labels[2])
        except:
            pass
    else:
        p.xticks(locs[0],labels[0])
        p.yticks(locs[1],labels[1][::-1])
    return(True)

def piechart(dictionary):
    '''creates a piechart from a dictionary dictionary={label:frac}'''
    labels=[]
    fracs=[]
    for key in dictionary:
        labels.append(key)
        fracs.append(dictionary[key])
    p.pie(fracs,labels=labels)
    return True

def barchart(dictionary):
    '''creates a barchart from a dictionary dictionary={label:frac}'''
    labels=[]
    fracs=[]
    indices=[]
    i=0
    width=0.35
    for key in dictionary:
        indices.append(i)
        labels.append(key)
        fracs.append(dictionary[key])
        i+=1
    indices=np.array(indices)
    p.bar(indices,fracs,width)
    p.xticks(indices+width/2., labels )
    return True
        
def line(f,start, stop, step=False, log=True):
    '''similar to fplot but with more options'''
    x=[]
    y=[]
    if not step:
        if log:
            step=2
        else:
            step=1
    i=start
    while i<=stop:
        x.append(i)
        y.append(f(i))
        if log:
            i*=step
        else:
            i+=step
    p.plot(x,y)
    if log:
        p.loglog()
    return True