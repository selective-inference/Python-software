import itertools
import numpy as np
from scipy.spatial import (voronoi_plot_2d, Voronoi, 
                           ConvexHull, convex_hull_plot_2d)
from scipy.spatial._plotutils import _adjust_bounds
from matplotlib import pyplot
from selection.constraints import constraints, simulate_from_constraints

def hull_without_points(hull, ax=None):
    """
    modified from scipy.spatial.convex_hull_plot_2d
    """
    if ax is None:
        ax = pyplot.gcf().gca()
        
    if hull.points.shape[1] != 2:
        raise ValueError("Convex hull is not 2-D")

    for simplex in hull.simplices:
        ax.plot(hull.points[simplex,0], hull.points[simplex,1], 'k-')

    _adjust_bounds(ax, hull.points)

    return ax.figure

def angle(x):
    """
    recover angle from a 2 vector
    """
    theta = np.arccos(x[0] / np.linalg.norm(x))
    if x[1] < 0:
        theta = 2 * np.pi - theta
    return theta

def just_hull(W, fill=True, fill_args={}, label=None, ax=None):
    
    """
    Draw the hull without points
    """
    hull = ConvexHull(W)
    f = hull_without_points(hull, ax=ax)
    a = f.gca()
    a.set_xticks([])
    a.set_yticks([])

    if fill:
        A, b, pairs, angles, perimeter = extract_constraints(hull)
        perimeter_vertices = np.array([v for _, v in perimeter])

        pyplot.scatter(perimeter_vertices[:,0],
                       perimeter_vertices[:,1], c='gray', s=100)
        pyplot.fill(perimeter_vertices[:,0], perimeter_vertices[:,1],
                    label=label, **fill_args)
    a.scatter(0,0, marker='+', c='k', s=50)
    return f, A, b, pairs, angles, perimeter 
    
def extract_constraints(hull):
    """
    given a convex hull, extract

    (A,b) such that

    $$hull = \{x: Ax+b \geq 0 \}$$

    also, return rays of the normal cone associated to each vertex as `pairs`

    """
    A = []
    b = []
    pairs = []
    angles = []

    perimeter = []

    for simplex1, simplex2 in itertools.combinations(hull.simplices, 2):
        intersect = set(simplex1).intersection(simplex2)

        for p in simplex1:
            perimeter.append((angle(hull.points[p]), list(hull.points[p])))

        for p in simplex2:
            perimeter.append((angle(hull.points[p]), list(hull.points[p])))

        if intersect:
            v1, v2 = hull.points[[simplex1[0], simplex1[1]]]
            diff = v1-v2
            normal1 = np.array([diff[1],-diff[0]])

            # find a point not in the simplex
            i = 0
            while True:
                s = hull.points[i]
                if i not in simplex1:
                    break
                i += 1    
            if np.dot(normal1, s-hull.points[simplex1[0]]) > 0:
                normal1 = -normal1
                
            v1, v2 = hull.points[[simplex2[0], simplex2[1]]]
            diff = v1-v2
            normal2 = np.array([diff[1],-diff[0]])
            
            # find a point not in the simplex
            i = 0
            while True:
                s = hull.points[i]
                if i not in simplex2:
                    break
                i += 1
                
            if np.dot(normal2, s-hull.points[simplex2[0]]) > 0:
                normal2 = -normal2
                
            dual_basis = np.vstack([normal1, normal2])
            angles.extend([angle(normal1), angle(normal2)])
            pairs.append((hull.points[list(intersect)[0]], dual_basis))

    for simplex in hull.simplices:
        v1, v2 = hull.points[[simplex[0], simplex[1]]]
        diff = v1-v2
        normal = np.array([diff[1],-diff[0]])
        offset = -np.dot(normal, v1)
        scale = np.linalg.norm(normal)
        if offset < 0:
            scale *= -1
            normal /= scale
            offset /= scale
        A.append(normal)
        b.append(offset)

    # crude rounding
    angles = np.array(angles)
    angles *= 50000
    angles = np.unique(angles.astype(np.int))
    angles = angles / 50000.

    return np.array(A), np.array(b), pairs, angles, sorted(perimeter)

def hull_with_point(W, fill=True, fill_args={}, label=None, ax=None,
                    Y=None):
    f, A, b, pairs, angles, perimeter = just_hull(W,
                                                  fill=fill,
                                                  label=label,
                                                  ax=ax,
                                                  fill_args=fill_args)
    
    representation = constraints((A,b), None)
    if Y is None:
        Y = simulate_from_constraints(representation)
    a = f.gca()
    a.scatter(Y[0], Y[1], label=r'$y$', s=100)

    return f, A, b, pairs, angles, perimeter

def hull_with_slice(W, fill=True, fill_args={}, label=None, ax=None,
                    Y=None,
                    eta=None):
    f, A, b, pairs, angles, perimeter = just_hull(W,
                                                  fill=fill,
                                                  label=label,
                                                  ax=ax,
                                                  fill_args=fill_args)
    
    representation = constraints((A,b), None)
    if Y is None:
        Y = simulate_from_constraints(representation)

    ax.scatter(Y[0], Y[1], label=r'$y$', s=100)

    if eta is None:
        eta = np.random.standard_normal(2)

    ax.arrow(0,0,eta[0],eta[1], linewidth=5, head_width=0.05, fc='k')

    Vp, _, Vm = representation.pivots(eta, Y)[:3]

    Yperp = Y - (np.dot(eta, Y) / 
                 np.linalg.norm(eta)**2 * eta)

    if Vm == np.inf:
        Vm = 10000

    width_points = np.array([(Yperp + Vp*eta /  
                              np.linalg.norm(eta)**2),
                             (Yperp + Vm*eta /  
                              np.linalg.norm(eta)**2)])

    ax.plot(width_points[:,0], width_points[:,1], 'r--', linewidth=2)
    ax.scatter(width_points[:,0], width_points[:,1], label=r'${\cal V}$', s=150, marker='x', linewidth=2, c='k')


    return f, A, b, pairs, angles, perimeter

def hull_with_point_and_rays(W, fill=True, fill_args={}, label=None, ax=None,
                             Y=None):
    
    f, A, b, pairs, angles, perimeter = hull_with_point(W,
                                                        fill=fill,
                                                        label=label,
                                                        ax=ax,
                                                        fill_args=fill_args,
                                                        Y=Y)

    a = f.gca()
    for i in range(len(pairs)):
        v, D = pairs[i]
        a.plot([v[0],v[0]+10000*D[0,0]],[v[1],v[1]+10000*D[0,1]], 'k--')
        a.plot([v[0],v[0]+10000*D[1,0]],[v[1],v[1]+10000*D[1,1]], 'k--')
    
    a.set_xlim(1.5*np.array(a.get_xlim()))
    a.set_ylim(1.5*np.array(a.get_ylim()))
    
    return f, A, b, pairs, angles, perimeter 

def hull_zoom(W, fill=True, fill_args={}, label=None, ax=None,
              Y=None):
    
    f, A, b, pairs, angles, perimeter = hull_with_point(W,
                                                        fill=fill,
                                                        label=label,
                                                        ax=ax,
                                                        fill_args=fill_args,
                                                        Y=Y)

    a = f.gca()
    for i in range(len(pairs)):
        v, D = pairs[i]
        a.plot([v[0],v[0]+100000*D[0,0]],[v[1],v[1]+100000*D[0,1]], 'k--')
        a.plot([v[0],v[0]+100000*D[1,0]],[v[1],v[1]+100000*D[1,1]], 'k--')
    
    a.set_xlim(5000*np.array(a.get_xlim()))
    a.set_ylim(5000*np.array(a.get_ylim()))
    
    return f, A, b, pairs, angles, perimeter 
    
def cone_rays(angles, which=None, ax=None, fill_args={}):
    """

    Plot the given Voronoi diagram in 2-D based on a set of directions

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib.

    """
    angles = np.sort(angles)
    points = np.array([np.cos(angles), np.sin(angles)]).T
    
    vor = Voronoi(points)
    
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if ax is None:
        ax = pyplot.gca()

    rays = np.array([(np.cos(angle), np.sin(angle)) for angle in angles])
    for i in range(rays.shape[0]):
        rays[i] /= np.linalg.norm(rays[i])
    rays *= 100
    

    for ray in rays:
        ax.plot([0,ray[0]],[0,ray[1]], 'k--')
    
    if which is not None:
        if which < rays.shape[0]-1:
            active_rays = [rays[which], rays[which+1]]    
        else:
            active_rays = [rays[0], rays[-1]]
        poly = np.vstack([active_rays[0], np.zeros(2), active_rays[1], 100*(active_rays[0]+active_rays[1])])
        dual_rays = np.linalg.pinv(np.array(active_rays))
        
    else:
        poly = None
        active_rays = None
        dual_rays = None

    _adjust_bounds(ax, vor.points)
    
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, poly, dual_rays, np.array(active_rays)

def cone_with_point(angles, which, fill_args={}, ax=None, label=None,
                    Y=None):

    ax, poly, constraint, rays = cone_rays(angles, which, ax=ax, fill_args=fill_args)
    eta = rays[0]
    representation = constraints((constraint, np.zeros(2)), None)
    if Y is None:
        Y = simulate_from_constraints(representation)
    pyplot.scatter(Y[0], Y[1], s=100, label=label)
    return ax, poly, constraint, rays

def cone_with_region(angles, which, fill_args={}, ax=None, label=None,
                    Y=None):

    ax, poly, constraint, rays = cone_rays(angles, which, ax=ax, fill_args=fill_args)
    eta = rays[0]
    representation = constraints((constraint, np.zeros(2)), None)
    if Y is None:
        Y = simulate_from_constraints(representation)
    pyplot.scatter(Y[0], Y[1], s=100, label=label)
    ax.fill(poly[:,0], poly[:,1], label=r'$C(y)$', **fill_args)
    return ax, poly, constraint, rays

def cone_with_arrow(angles, which, fill_args={}, ax=None, label=None,
                    Y=None):

    ax, poly, constraint, rays = cone_rays(angles, which, ax=ax, fill_args=fill_args)
    eta = rays[0]
    representation = constraints((constraint, np.zeros(2)), None)
    if Y is None:
        Y = simulate_from_constraints(representation)
    pyplot.scatter(Y[0], Y[1], s=100, label=label)
    ax.fill(poly[:,0], poly[:,1], label=r'$C(y)$', **fill_args)
    ax.arrow(0,0,eta[0]/200.,eta[1]/200., label=r'$\eta$', linewidth=5, head_width=0.05, fc='k')
    return ax, poly, constraint, rays

def cone_with_slice(angles, which, fill_args={}, ax=None, label=None,
                    Y=None):

    ax, poly, constraint, rays = cone_rays(angles, which, ax=ax, fill_args=fill_args)
    eta = rays[0]
    representation = constraints((constraint, np.zeros(2)), None)

    if Y is None:
        Y = simulate_from_constraints(representation)

    pyplot.scatter(Y[0], Y[1], s=100, label=label)
    ax.fill(poly[:,0], poly[:,1], label=r'$C(y)$', **fill_args)
    ax.arrow(0,0,eta[0]/200.,eta[1]/200., label=r'$\eta$', linewidth=5, head_width=0.05, fc='k')


    Vp, _, Vm = representation.pivots(eta, Y)[:3]

    Yperp = Y - (np.dot(eta, Y) / 
                 np.linalg.norm(eta)**2 * eta)

    if Vm == np.inf:
        Vm = 10000

    width_points = np.array([(Yperp + Vp*eta /  
                              np.linalg.norm(eta)**2),
                             (Yperp + Vm*eta /  
                              np.linalg.norm(eta)**2)])

    ax.plot(width_points[:,0], width_points[:,1], 'r--', linewidth=2)
    ax.scatter(width_points[:,0], width_points[:,1], label=r'${\cal V}$', s=150, marker='x', linewidth=2, c='k')

    return ax, poly, constraint, rays

if __name__ == "__main__":

    Y = np.array([0.5,0.5])
    import os

    if os.path.exists('points.npy'):
        W = np.load('points.npy')
    else:
        np.random.seed(10)
        W = np.random.standard_normal((30,2))

    hull = ConvexHull(W)
    f = pyplot.figure(figsize=(12,12))
    f = just_hull(W, fill_args={'facecolor':'gray', 'alpha':0.2}, ax=f.gca(),
                  label=r'$K$')[0]
    legend_args = {'scatterpoints':1, 'fontsize':30, 'loc':'lower left'}
    f.gca().legend(**legend_args)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('hull%s' % ext)
    pyplot.clf()

    f = hull_with_point(W, 
                        fill_args={'facecolor':'gray', 'alpha':0.2}, 
                        ax=f.gca(),
                        label=r'$K$',
                        Y=Y)[0]
    f.gca().legend(**{'scatterpoints':1, 'fontsize':30, 'loc':'lower right'})
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('hull_after_sampling%s' % ext)
    pyplot.clf()

    f = hull_with_point_and_rays(W, 
                                 fill_args={'facecolor':'gray', 'alpha':0.2}, 
                                 ax=f.gca(),
                                 label=r'$K$',
                                 Y=Y)[0]

    f.gca().legend(**legend_args)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('hull_with_rays%s' % ext)
    pyplot.clf()

    f = hull_zoom(W, 
                  fill_args={'facecolor':'gray', 'alpha':0.2}, 
                  ax=f.gca(),
                  label=r'$K$',
                  Y=Y)[0]

    for ext in ['.svg', '.pdf']:
        pyplot.savefig('hull_zoom%s' % ext)
    pyplot.clf()

    f = hull_with_slice(W, 
                        fill_args={'facecolor':'gray', 'alpha':0.2}, 
                        ax=f.gca(),
                        label=r'$K$',
                        Y=Y)[0]
    f.gca().legend(**{'scatterpoints':1, 'fontsize':30, 'loc':'lower right'})

    for ext in ['.svg', '.pdf']:
        pyplot.savefig('hull_slice%s' % ext)
    pyplot.clf()

    f, A, b, pairs, angles, perimeter = \
        just_hull(W, 
                  fill_args={'facecolor':'gray', 'alpha':0.2}, 
                  ax=f.gca(),
                  label=r'$K$')
    pyplot.clf()

    ax, poly, dual_rays, active_rays = cone_rays(angles)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('cone_rays%s' % ext)
    pyplot.clf()

    region = 5
    ax, poly, dual_rays, active_rays = cone_with_point(angles,
                                                       region,
                                                       label=r'$y$',
                                                       Y=Y)
    ax.legend(**legend_args)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('cone_point%s' % ext)
    pyplot.clf()

    ax, poly, dual_rays, active_rays = cone_with_region(angles,
                                                        region,
                                                        label=r'$y$',
                                                        Y=Y,
                                                        fill_args=\
                  {'facecolor':'gray', 'alpha':0.2})

    ax.legend(**legend_args)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('cone_region%s' % ext)
    pyplot.clf()

    ax, poly, dual_rays, active_rays = cone_with_arrow(angles,
                                                       region,
                                                       label=r'$y$',
                                                       Y=Y,
                                                       fill_args=\
                  {'facecolor':'gray', 'alpha':0.2})

    ax.legend(**legend_args)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('cone_arrow%s' % ext)
    pyplot.clf()

    ax, poly, dual_rays, active_rays = cone_with_slice(angles,
                                                       region,
                                                       label=r'$y$',
                                                       Y=Y,
                                                       fill_args=\
                  {'facecolor':'gray', 'alpha':0.2})

    ax.legend(**legend_args)
    for ext in ['.svg', '.pdf']:
        pyplot.savefig('cone_slice%s' % ext)
    pyplot.clf()
    
