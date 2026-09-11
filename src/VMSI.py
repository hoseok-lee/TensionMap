import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generic_filter
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, least_squares, LinearConstraint, lsq_linear
from sklearn.cluster import KMeans
import pandas as pd
from skimage import measure, color
from skimage.segmentation import expand_labels, find_boundaries
from matplotlib import cm, patches, colors
import matplotlib
from src.segment import Segmenter, set_array_at
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from concurrent.futures import ProcessPoolExecutor


def sx_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
    """

    Computes part of the jacobian of the energy function (the partial derivative with respect to x) analytically
    to improve the speed and accuracy of initial optimisation.

    """
    t3 = p1-p2
    t5 = p1*q1x
    t6 = p2*q2x
    t7 = rx*t3
    t2 = -t5+t6+t7
    t9 = p1*q1y
    t10 = p2*q2y
    t11 = ry*t3
    t4 = -t9+t10+t11
    t8 = np.power(t2,2)
    t12 = np.power(t4,2)
    t13 = t8+t12
    t14 = np.divide(1,np.power(t13, 1.5))
    t15 = np.divide(1,np.sqrt(t13))
    t16 = q1x-rx
    t17 = q2x-rx

    Dx = np.array([p1*t15-p1*t8*t14,-p1*t2*t4*t14,
                   t15*t16-t2*t14*(t2*t16*2+t4*(q1y-ry)*2)*(0.5),
                   -p2*t15+p2*t8*t14,p2*t2*t4*t14,
                   -t15*t17+t2*t14*(t2*t17*2+t4*(q2y-ry)*2)*(0.5)]).T
    return Dx

def sy_grad(p1, p2, q1x, q1y, q2x, q2y, rx, ry):
    """

    Computes part of the jacobian of the energy function (the partial derivative with respect to y) analytically
    to improve the speed and accuracy of initial optimisation.

    """
    t3 = p1-p2
    t5 = p1*q1x
    t6 = p2*q2x
    t7 = rx*t3
    t2 = -t5+t6+t7
    t8 = p1*q1y
    t9 = p2*q2y
    t10 = ry*t3
    t4 = -t8+t9+t10
    t11 = np.power(t2,2)
    t12 = np.power(t4,2)
    t13 = t11+t12
    t14 = np.divide(1,np.power(t13,1.5))
    t15 = np.divide(1,np.sqrt(t13))
    t16 = q1y-ry
    t17 = q2y-ry
    Dy = np.array([-p1*t2*t4*t14,p1*t15-p1*t12*t14,
                   t15*t16-t4*t14*(t4*t16*2+t2*(q1x-rx)*2)*(0.5),
                   p2*t2*t4*t14,-p2*t15+p2*t12*t14,
                   -t15*t17+t4*t14*(t4*t17*2+t2*(q2x-rx)*2)*(0.5)]).T
    return Dy

def radius_grad_theta(p1,p2,q1x,q1y,q2x,q2y,t1,t2):
    """

    Computes the jacobian of the energy function (the partial derivative with respect to x) analytically to improve the
    speed and accuracy of theta optimisation.

    """
    t4 = p1-p2
    # For a relatively homogeneous tissue, neighbouring cells can genuinely
    # have near-identical pressure (dP -> 0 is the physical straight-edge /
    # infinite-radius limit, not an error). Dividing by t4/t4^2 below is
    # singular exactly there, so clamp its magnitude away from zero - kept
    # in sync with the same floor used in initial_minimization/fit.
    _min_dp = 1e-2
    t4 = np.where(t4 >= 0, np.maximum(t4, _min_dp), np.minimum(t4, -_min_dp))
    t5 = q1x-q2x
    t6 = q1y-q2y
    t7 = np.divide(1,np.power(t4,2))
    t8 = t1-t2
    t9 = t4*t8
    t10 = np.power(t5,2)
    t11 = np.power(t6,2)
    t12 = t10+t11
    t13 = t9-p1*p2*t12
    t14 = np.divide(1,np.sqrt(-t7*t13))
    t15 = np.divide(1,t4)
    Dr = np.array([t14*t15*(-0.5),t14*t15*(0.5)]).T
    return Dr

def rx_grad(p1, p2, q1x, q2x, rx):
    """

    Computes part of the jacobian of the nonlinear constraint function (the partial derivative with respect to x)
    analytically to improve the speed and accuracy of initial optimisation.

    """
    Dx = np.array([p1, np.zeros(p1.shape), q1x-rx, -p2, np.zeros(p1.shape), -q2x+rx]).T
    return Dx

def ry_grad(p1, p2, q1y, q2y, ry):
    """

    Computes part of the jacobian of the nonlinear constraint function (the partial derivative with respect to y)
    analytically to improve the speed and accuracy of initial optimisation.

    """
    Dy = np.array([np.zeros(p1.shape),p1,q1y-ry,np.zeros(p1.shape),-p2,-q2y+ry]).T
    return Dy

def rho_x_grad(p1,p2,q1x,q2x):
    """

    Computes part of the jacobian of the energy function analytically to improve the speed and accuracy of
    the main optimisation step.

    """
    t2 = p1-p2
    t3 = np.divide(1,t2)
    t4 = np.divide(1,np.power(t2,2))
    t5 = p1*q1x
    t6 = t5-p2*q2x
    z = np.zeros(p1.shape)
    dRhoX = np.array([p1*t3,z,z,q1x*t3-t4*t6,-p2*t3,z,z,-q2x*t3+t4*t6]).T
    return dRhoX

def rho_y_grad(p1,p2,q1y,q2y):
    """

    Computes part of the jacobian of the energy function analytically to improve the speed and accuracy of
    the main optimisation step.

    """
    t2 = p1-p2
    t3 = np.divide(1,t2)
    t4 = np.divide(1,np.power(t2,2))
    t5 = p1*q1y
    t6 = t5-p2*q2y
    z = np.zeros(p1.shape)
    dRhoY = np.array([z,p1*t3,z,q1y*t3-t4*t6,z,-p2*t3,z,-q2y*t3+t4*t6]).T
    return dRhoY

def radius_grad(p1,p2,q1x,q1y,q2x,q2y,t1,t2):
    """

    Computes part of the jacobian of the energy function analytically to improve the speed and accuracy of
    the main optimisation step.

    """
    t4 = p1-p2
    # See the analogous clamp in radius_grad_theta: dP -> 0 is a genuine
    # physical limit (near-uniform pressure / straight edge) for a
    # relatively homogeneous tissue, not an error - clamp away from the
    # singularity at t4=0 instead of dividing by it directly.
    _min_dp = 1e-2
    t4 = np.where(t4 >= 0, np.maximum(t4, _min_dp), np.minimum(t4, -_min_dp))
    t5 = q1x-q2x
    t6 = q1y-q2y
    t7 = np.divide(1,np.power(t4,2))
    t8 = t1-t2
    t9 = t4*t8
    t10 = np.power(t5,2)
    t11 = np.power(t6,2)
    t12 = t10+t11
    t15 = p1*p2*t12
    t13 = t9-t15
    t16 = t7*t13
    t14 = np.divide(1,np.sqrt(-t16))
    t17 = q1x*2
    t18 = q2x*2
    t19 = t17-t18
    t20 = p1*p2*t7*t14*t19*(0.5)
    t21 = q1y*2
    t22 = q2y*2
    t23 = t21-t22
    t24 = p1*p2*t7*t14*t23*(0.5)
    t25 = np.divide(1,t4)
    t26 = np.divide(1,np.power(t4,3))
    t27 = t13*t26*2
    dR = np.array([t20,t24,t14*t25*(-0.5),
                   t14*(t27+t7*(-t1+t2+p2*t12))*(0.5),
                   -t20,-t24,t14*t25*(0.5),
                   t14*(t27-t7*(t1-t2+p1*t12))*(-0.5)]).T
    return dR

def gather_pairs(X, pairs):
    """
    Equivalent to np.matmul(dC, X), where dC is the (n_edges, n_cells) difference
    operator built in build_diff_operators - dC has exactly one +1 (at pairs[e,0])
    and one -1 (at pairs[e,1]) per row e, so dC @ X == X[pairs[:,0]] - X[pairs[:,1]]
    exactly, for any X indexed by cell (1-D or 2-D, e.g. per-cell scalars or q's
    (n_cells, 2) coordinates). This is O(n_edges) instead of O(n_edges * n_cells),
    since dC's density (2 nonzeros/row against a potentially large cell count) makes
    the dense matmul do far more arithmetic than the difference it's actually computing.
    """
    return X[pairs[:,0]] - X[pairs[:,1]]

def scatter_pairs(w, pairs, n):
    """
    Equivalent to np.matmul(dC.T, w) (dC as in gather_pairs above), i.e. the
    reverse/reduction direction: accumulate each edge's per-edge value(s) w back
    onto its two cells (+w at pairs[:,0], -w at pairs[:,1]), summing contributions
    when multiple edges share a cell. w may be 1-D (n_edges,) or 2-D (n_edges, k).
    Uses np.bincount per column rather than a dense (n_edges, n_cells) matmul, for
    the same O(n_edges) vs O(n_edges * n_cells) reason as gather_pairs.
    """
    if w.ndim == 1:
        return np.bincount(pairs[:,0], weights=w, minlength=n) - np.bincount(pairs[:,1], weights=w, minlength=n)
    out = np.zeros((n, w.shape[1]))
    for k in range(w.shape[1]):
        out[:,k] = (np.bincount(pairs[:,0], weights=w[:,k], minlength=n)
                    - np.bincount(pairs[:,1], weights=w[:,k], minlength=n))
    return out

def symmetric_scatter_pairs(w, pairs, n):
    """
    Equivalent to np.matmul(np.abs(dC).T, w) (dC as in gather_pairs above): like
    scatter_pairs, but adds w to *both* of an edge's cells (abs(dC) has +1, not
    -1, at pairs[:,1]) rather than subtracting at the second. w may be 1-D or 2-D.
    """
    if w.ndim == 1:
        return np.bincount(pairs[:,0], weights=w, minlength=n) + np.bincount(pairs[:,1], weights=w, minlength=n)
    out = np.zeros((n, w.shape[1]))
    for k in range(w.shape[1]):
        out[:,k] = (np.bincount(pairs[:,0], weights=w[:,k], minlength=n)
                    + np.bincount(pairs[:,1], weights=w[:,k], minlength=n))
    return out

def dense_jacobian_pairs(w, pairs, n_cols):
    """
    Equivalent to np.multiply(dC.T, w).T (dC as in gather_pairs above): builds the
    dense (n_edges, n_cols) matrix nlopt's vector-constraint API requires as an
    actual dense Jacobian block (so, unlike gather_pairs/scatter_pairs, this can't
    avoid the O(n_edges * n_cols) output allocation) - but constructs it via direct
    index assignment into a zeros array instead of an elementwise multiply over the
    already-dense (mostly-zero) dC.T, avoiding that redundant full pass.
    """
    out = np.zeros((w.shape[0], n_cols))
    rows = np.arange(w.shape[0])
    out[rows, pairs[:,0]] = w
    out[rows, pairs[:,1]] = -w
    return out

def _make_stage_pbar(desc, total, active):
    """
    Progress bar for one nlopt minimisation stage, ticked once per objective
    evaluation. Returns a tqdm instance, or None if tqdm isn't installed or the
    stage isn't being tracked (active=False) - callers must handle None. nlopt
    has no iteration callback, so the objective function itself does the ticking;
    `total` is the stage's maxeval (evaluations may finish early, or slightly
    exceed it via AUGLAG subproblems).
    """
    if not active:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, leave=True)

class VMSI():

    def __init__(self, vertices, cells, edges, width, height, verbose, optimiser='nlopt', mask=None,
                 isolated_tension=1.0, isolated_background_pressure=0.0):
        self.vertices = vertices
        self.cells = cells
        self.edges = edges
        self.width = width
        self.height = height
        self.verbose = verbose
        self.optimiser = optimiser
        # Relabelled mask (as returned by Segmenter.process_segmented_image),
        # where a cell's C_df row index equals its label in this mask - same
        # convention already used by inject_isolated_cell_topology. Optional
        # and only used by infer_isolated_cells() to fit a circle to an
        # isolated cell's true contour; kept None by default (e.g. tiling)
        # since callers that don't have this mask handy still work, just
        # falling back to the older synthetic-edge-based estimate.
        self.mask = mask
        self.isolated_tension = isolated_tension
        self.isolated_background_pressure = isolated_background_pressure
        # Set by run_VMSI's attach_adata() when an adata is passed in - an AnnData
        # object whose .obs has been synced with this model's self.cells (see
        # attach_adata's docstring). None if no adata was provided.
        self.adata = None

        # Mark fourfold vertices
        self.vertices['fourfold'] = [(np.shape(nverts)[0] != 3) for nverts in self.vertices['nverts']]

        # Initialize new columns
        self.edges['radius'] = np.zeros(len(self.edges))
        self.edges['rho'] = tuple((0,0) for _ in range(len(self.edges)))
        self.edges['fitenergy'] = np.zeros(len(self.edges))
        self.edges['tension'] = np.zeros(len(self.edges))
        self.cells['pressure'] = np.zeros(len(self.cells))
        self.cells['qx'] = np.zeros(len(self.cells))
        self.cells['qy'] = np.zeros(len(self.cells))
        self.cells['theta'] = np.zeros(len(self.cells))
        self.cells['stress'] = [np.array([0,0,0]) for _ in range(self.cells.shape[0])]
        # Only meaningful for isolated cells (see infer_isolated_cells) - NaN
        # for normal cells that go through the main vertex-network fit.
        self.cells['circularity'] = np.full(len(self.cells), np.nan)

        # Initialize attributes
        self.dV = None
        self.dC = None
        self.involved_cells = None
        self.involved_vertices = None
        self.involved_edges = None
        self.isolated_cells = np.array([], dtype=int)
        self.cluster_cells = np.array([], dtype=int)
        self.bulk_cells = None
        self.bulk_vertices = None
        self.ext_cells = None
        self.ext_vertices = None
        self.cell_pairs = None
        self.edgearc_x = None
        self.edgearc_y = None
        self.avg_edge_length = None

        if self.optimiser == 'matlab':
            import matlab
            import matlab.engine
            import pathlib
            src_path = str(pathlib.Path(__file__).parent.resolve())

            # Initialise Matlab engine
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(src_path)
        elif self.optimiser == 'nlopt':
            import nlopt


    def fit_circle(self):
        """

        Fit circle to each edge
        If edge is too flat, fit line instead

        """
        for i in range(len(self.edges)):
            r1 = np.array(self.vertices['coords'][self.edges['verts'][i][0]])
            r2 = np.array(self.vertices['coords'][self.edges['verts'][i][1]])

            edge_pixels = [np.unravel_index(pixel, (self.width, self.height)) for pixel in self.edges['pixels'][i]]

            nB = np.matmul(np.array([[0, 1], [-1, 0]]), r1 - r2)
            D = np.sqrt(np.sum(np.power(nB, 2)))
            nB = np.divide(nB, D)
            x0 = 0.5*(r1 + r2)

            delta = np.subtract(edge_pixels, x0)
            IP = (delta[:,0] * nB[0]) + (delta[:,1] * nB[1])
            L0 = D/2

            A = 2*np.sum(np.power(IP, 2))
            B = np.sum((np.sum(np.power(delta, 2), axis=1) - np.power(L0, 2)) * IP)
            y0 = np.divide(B, A)

            # Minimise the MSQ between edge pixels and fitted arc
            def energyfunc(x):
                return np.mean(np.power(np.sqrt(np.sum(np.power(delta-(x*nB), 2), axis=1)) - np.sqrt(np.power(x, 2) + np.power(L0, 2)), 2))

            if not np.isnan(y0):
                res = minimize(energyfunc, y0, tol=1e-8)
            else:
                res = minimize(energyfunc, 0, tol=1e-8)
            y = res.x
            E = res.fun

            linedistance = np.mean(np.power(IP, 2))
            # Store the radius, centre and fit energy of each fitted circular arc or straight line
            if (E < linedistance and len(edge_pixels) > 3):
                self.edges.at[i,'radius'] = np.sqrt(np.power(y, 2) + np.power(L0, 2))
                self.edges.at[i,'rho'] = x0 + (y * nB)
                self.edges.at[i,'fitenergy'] = E
            else:
                self.edges.at[i,'radius'] = np.Inf
                self.edges.at[i,'rho'] = np.array([np.Inf, np.Inf])
                self.edges.at[i,'fitenergy'] = linedistance
        return


    def remove_fourfold(self):
        """

        recursively removes fourfold (or greater) vertices by splitting vertex apart in direction of greatest variance

        """

        for v in range(len(self.vertices)):
            if (np.shape(self.vertices['nverts'][v])[0] > 3  and not (0 in self.vertices['ncells'][v])):
                while (np.shape(self.vertices['nverts'][v])[0] > 3):
                    num_v = len(self.vertices)
                    num_e = len(self.edges)

                    nverts = np.array(self.vertices['nverts'][v])
                    nedges = self.vertices['edges'][v]
                    ncells = self.vertices['ncells'][v]

                    # -1 is a sentinel meaning no edge exists between two topological
                    # neighbours (set by find_edges when one vertex borders cell 0).
                    # Splitting this vertex would corrupt the edge/cell data structures.
                    if np.any(np.array(nedges, dtype=float) < 0):
                        break

                    R = np.array([self.vertices['coords'][vert] for vert in nverts])
                    rV = np.array(self.vertices['coords'][v])

                    R = R - np.mean(R, axis=0)
                    I = np.matmul(R.T,R)

                    W, V = np.linalg.eig(I)
                    direction = V[:,np.argmax(W)]

                    # create two new vertices, positive and negative
                    rV1 = rV + (direction/2)
                    rV2 = rV - (direction/2)

                    # set positive neighbour vertices to the 2 vertices closest to the direction of vertex movement
                    # all other neighbour vertices are negative
                    indices = np.argsort(np.dot(R, direction))[-2:]
                    pos_verts = np.zeros_like(nverts)
                    pos_verts[indices] = 1
                    neg_verts = 1 - pos_verts
                    

                    # change vertex with current index to negative vertex
                    self.vertices.at[v,'coords'] = rV2.tolist()
                    set_array_at(self.vertices, v, 'nverts', np.concatenate((nverts[neg_verts.astype('bool')], np.array([num_v]))))
                    self.vertices.at[v,'fourfold'] = (np.shape(self.vertices['nverts'][v])[0] > 3)

                    neg_cells = ncells[[(sum(np.isin(nverts[neg_verts.astype('bool')], self.cells['nverts'][cell]))==2) for cell in ncells]]

                    # add positive vertex
                    self.vertices = pd.concat(
                        [
                            self.vertices, 
                            pd.DataFrame({
                                'coords': [np.array([0,0])],
                                'ncells': [np.array([])],
                                'nverts': [np.array([])],
                                'edges': [np.array([])],
                                'fourfold': False})
                        ], ignore_index=True)
                    self.vertices.at[num_v,'coords'] = rV1.tolist()
                    set_array_at(self.vertices, num_v, 'nverts', np.concatenate((nverts[pos_verts.astype('bool')], np.array([v]))))
                    self.vertices.at[num_v,'fourfold'] = False

                    pos_cell = ncells[[(sum(np.isin(nverts[pos_verts.astype('bool')], self.cells['nverts'][cell]))==2) for cell in ncells]]

                    # update new positive vertex index for neighbour vertices
                    for vert in nverts[pos_verts.astype('bool')]:
                        self.vertices.at[vert, 'nverts'][self.vertices['nverts'][vert] == v] = num_v

                    joint_cells = ncells[np.invert(np.isin(ncells, np.concatenate([pos_cell, neg_cells])))]

                    set_array_at(self.vertices, v, 'ncells', np.concatenate((joint_cells, neg_cells)))
                    set_array_at(self.vertices, num_v, 'ncells', np.concatenate((joint_cells, pos_cell)))

                    # update current edges
                    # this requires edges to be in the same order as vertices
                    neg_edges = nedges[neg_verts.astype('bool')]
                    pos_edges = nedges[pos_verts.astype('bool')]

                    self.edges.at[pos_edges[0], 'verts'][self.edges.at[pos_edges[0], 'verts'] == v] = num_v
                    self.edges.at[pos_edges[1], 'verts'][self.edges.at[pos_edges[1], 'verts'] == v] = num_v

                    # create new edge between new vertices
                    # edge is only one pixel long so no need to add pixels
                    self.edges = pd.concat(
                        [
                            self.edges, 
                            pd.DataFrame({
                                'pixels': [np.array([])],
                                'verts': [np.array([])],
                                'cells': [np.array([])],
                                'radius': [np.array([])],
                                'rho': [np.array([])],
                                'fitenergy': np.Inf,
                                'tension': float(0)})
                        ], ignore_index=True)
                    self.edges.at[num_e,'verts'] = np.array([v, num_v])
                    set_array_at(self.edges, num_e, 'cells', joint_cells)
                    self.edges.at[num_e,'pixels'] = np.array([])
                    self.edges.at[num_e,'radius'] = np.Inf
                    self.edges.at[num_e,'rho'] = np.array([np.Inf, np.Inf])

                    # update edges of new vertices
                    set_array_at(self.vertices, v, 'edges', np.concatenate((neg_edges, np.array([num_e]))))
                    set_array_at(self.vertices, num_v, 'edges', np.concatenate((pos_edges, np.array([num_e]))))

                    # update cells

                    # update pos cells
                    for cell in pos_cell:
                        self.cells.at[cell,'nverts'][self.cells['nverts'][cell] == v] = num_v
                        set_array_at(self.cells, cell, 'ncells', self.cells.at[cell, 'ncells'][np.isin(self.cells.at[cell,'ncells'], neg_cells, invert=True)])
                    # update neg cells
                    for cell in neg_cells:
                        set_array_at(self.cells, cell, 'ncells', self.cells.at[cell, 'ncells'][np.isin(self.cells.at[cell,'ncells'], pos_cell, invert=True)])
                    # update joint cells
                    for cell in joint_cells:
                        set_array_at(self.cells, cell, 'nverts', np.concatenate((self.cells.at[cell,'nverts'], np.array([num_v]))))
                        self.cells.at[cell,'numv'] = self.cells.at[cell,'numv']+1
        return


    def make_convex(self):
        """

        remove concave vertices by moving vertex to ensure all angles < pi

        """

        # find boundary vertices
        boundary_cells = np.where(self.cells['holes'].to_numpy())[0]
        boundary_verts = np.unique(np.concatenate(self.cells.loc[boundary_cells,'nverts'].tolist()))

        # Iterate through all non-boundary verts
        for v in range(len(self.vertices)):
            if (v not in boundary_verts and len(self.vertices.at[v,'nverts']) == 3):

                rv = np.array(self.vertices['coords'][v])
                nverts = np.array(self.vertices['nverts'][v])

                n = np.array([self.vertices['coords'][nverts[0]],
                              self.vertices['coords'][nverts[1]],
                              self.vertices['coords'][nverts[2]]])

                n_centered = n - np.mean(n, axis=0)
                theta = np.mod(np.arctan2(n_centered[:,1], n_centered[:,0]), 2*np.pi)
                n = n[np.argsort(theta),:]

                r = n - rv
                r = np.divide(r.T,(np.linalg.norm(r, axis=1))).T

                z12 = np.cross(np.concatenate((r[0,:], np.array([0]))), np.concatenate((r[1,:], np.array([0]))))
                z23 = np.cross(np.concatenate((r[1,:], np.array([0]))), np.concatenate((r[2,:], np.array([0]))))
                z31 = np.cross(np.concatenate((r[2,:], np.array([0]))), np.concatenate((r[0,:], np.array([0]))))

                # Determine angle between all pairs of edges
                theta12 = np.mod(np.arctan2(z12[2], np.dot(r[0,:], r[1,:])), 2*np.pi)
                theta23 = np.mod(np.arctan2(z23[2], np.dot(r[1,:], r[2,:])), 2*np.pi)
                theta31 = np.mod(np.arctan2(z31[2], np.dot(r[2,:], r[0,:])), 2*np.pi)

                # If any theta>pi, move vertex until it's not
                # This accounts for measurement noise in junctions where one angle is close to pi
                if theta12 > np.pi:
                    deltaR = np.dot(n[0,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[0,:]-n[1,:])) / np.dot(n[2,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[0,:]-n[1,:]))
                    nrv = rv + 1.5*deltaR*(n[2,:]-rv)
                elif theta23 > np.pi:
                    deltaR = np.dot(n[1,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[1,:]-n[2,:])) / np.dot(n[0,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[1,:]-n[2,:]))
                    nrv = rv + 1.5*deltaR*(n[0,:]-rv)
                elif theta31 > np.pi:
                    deltaR = np.dot(n[2,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[2,:]-n[0,:])) / np.dot(n[1,:]-rv, np.matmul(np.array([[0,-1],[1,0]]), n[2,:]-n[0,:]))
                    nrv = rv + 1.5*deltaR*(n[1,:]-rv)
                elif theta12 == np.pi:
                    nrv = rv + 0.5*r[2,:]
                elif theta23 == np.pi:
                    nrv = rv + 0.5*r[0,:]
                elif theta31 == np.pi:
                    nrv = rv + 0.5*r[1,:]
                else:
                    nrv = rv

                self.vertices.at[v,'coords'] = nrv.tolist()
        return


    def prepare_data(self):
        """

        Prepare data for tension inference:
        Remove fourfold vertices, fit circles, make vertices convex

        """

        # Fit circular arcs to each edge
        self.fit_circle()

        # Recursively remove n-fold vertices (for n>3) by splitting and moving resulting (n-1)-fold vertices apart
        self.remove_fourfold()

        # Inference cannot handle concave vertices (with one angle greater than pi) so remove these
        self.make_convex()

        return


    def classify_cells(self):
        """

        Determine which cells are involved in tension inference
        Initialize q as cell centroids

        """

        self.bulk_cells = np.array(range(0,len(self.cells)))

        boundary_cells = np.unique(np.concatenate([self.cells.at[0, 'ncells'], np.where(self.cells['holes'].to_numpy()[1:])[0]]))

        # Remove boundary cells and cells surrounded by boundary cells from bulk cells
#        all_threefold = np.array([not(any(self.vertices.loc[self.cells.at[cell, 'nverts'], 'fourfold'].to_numpy())) for cell in range(len(self.cells))])
#        self.bulk_cells = self.bulk_cells[all_threefold]
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, boundary_cells, invert=True)]
        bad_cells = np.array([])
        for cell in self.bulk_cells:
            if np.sum(np.isin(self.cells.at[cell, 'ncells'], self.bulk_cells)) == 0:
                bad_cells = np.append(bad_cells, cell)
        self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, bad_cells, invert=True)]

        if len(self.bulk_cells) == 0:
            raise ValueError(
                "No bulk (fully interior) cells found - every cell in this mask either touches the "
                "background directly, or only touches other cells that themselves touch the background. "
                "VMSI's vertex-network inference needs some genuinely interior, confluent tissue to solve "
                "tensions/pressures against; a mask made up of isolated cells and/or small touching "
                "clusters with no deep interior can't provide that. Use run_isolated_cells (or run_VMSI, "
                "which falls back to it automatically) to infer each cell's pressure directly from its own "
                "boundary curvature instead."
            )

        # This excludes vertices surrounded by boundary cells; edges at these vertices are not constrained enough for accurate inference
        self.bulk_vertices = np.unique(np.concatenate([self.cells.at[cell, 'nverts'] for cell in self.bulk_cells]))

        self.involved_cells = np.unique(np.concatenate([self.vertices.at[vert, 'ncells'] for vert in self.bulk_vertices]))
        self.ext_cells = self.involved_cells[np.isin(self.involved_cells, self.bulk_cells, invert=True)]
        self.involved_cells = np.concatenate((self.bulk_cells, self.ext_cells))

        self.involved_vertices = np.unique(np.concatenate([self.vertices.at[vert, 'nverts'] for vert in self.bulk_vertices]))
        self.ext_vertices = self.involved_vertices[np.isin(self.involved_vertices, self.bulk_vertices, invert=True)]
        self.involved_vertices = np.concatenate((self.bulk_vertices, self.ext_vertices))

        x0 = np.vstack([np.stack(self.cells['centroids'][self.involved_cells]).T,
                        np.zeros(len(self.involved_cells))]).T

        return x0


    def build_diff_operators(self):
        """

        Compute difference operators to enable vectorized operations

        """
        # Build cell adjacency matrix
        adj_mat = np.zeros((len(self.involved_cells), len(self.involved_cells)))
        num_edges = 0
        # A genuine cell-cell edge always borders exactly 2 cells; some edges
        # (segmentation noise, degenerate junctions) end up with 0/1/3+ due to
        # np.intersect1d over noisy topology, so exclude those before building
        # a uniform array - otherwise np.array() on a ragged list below raises
        # "inhomogeneous shape".
        edge_cells = [c for c in self.edges.cells.to_list() if len(c) == 2]
        edge_cells = np.array(edge_cells) if len(edge_cells) > 0 else np.empty((0, 2))

        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            for ncell in self.cells.at[cell, 'ncells']:
                j = np.ravel(np.where(self.involved_cells == ncell))
                # Ensure that edge between neighbouring cells actually exists
                if j.size > 0 and adj_mat[i, j] == 0 and np.any(np.all(np.sort(np.array([cell, ncell])) == edge_cells, axis=1)):
                    adj_mat[i, j] = 1
                    adj_mat[j, i] = 1
                    num_edges += 1

        # Compute difference operators
        self.dC = np.zeros((num_edges, len(self.involved_cells)))
        self.dV = np.zeros((num_edges, len(self.involved_vertices)))
        self.cell_pairs = np.zeros((num_edges, 2), dtype=int)

        diff_index = 0
        for i in range(len(self.involved_cells)):
            ncells = np.ravel(np.where(adj_mat[i,:] == 1))
            ncells = ncells[ncells > i]

            for cell in ncells:
                self.dC[diff_index, i] = 1
                self.dC[diff_index, cell] = -1
                self.cell_pairs[diff_index] = np.array([i, cell])

                verts = np.intersect1d(self.cells['nverts'][self.involved_cells[i]], self.cells['nverts'][self.involved_cells[cell]])

                if (len(verts) == 2):
                    self.dV[diff_index, np.where(self.involved_vertices == verts[0])] = 1
                    self.dV[diff_index, np.where(self.involved_vertices == verts[1])] = -1

                diff_index += 1

        # Check for bad vertices and edges
        bad_verts = np.invert(np.sum(np.abs(self.dV), axis=0) == 0)
        self.dV = self.dV[:,bad_verts]
        self.involved_vertices = self.involved_vertices[bad_verts]

        bad_edges = np.invert(np.sum(np.abs(self.dV), axis=1) < 2)
        self.dV = self.dV[bad_edges,:]
        self.dC = self.dC[bad_edges,:]
        self.cell_pairs = self.cell_pairs[bad_edges,:]

        if self.dV.shape[0] == 0 or len(self.involved_vertices) == 0:
            raise ValueError(
                "No usable edges or vertices remain after filtering in build_diff_operators - "
                "classify_cells found some cells that aren't directly touching the background, but "
                "none of them form a well-constrained enough local network (e.g. edges shared by "
                "exactly 2 real cells, vertices with enough incident edges) for the main vertex-network "
                "inference. This is the same underlying issue as having no bulk cells at all - a mask "
                "made of small touching clusters with no genuinely confluent interior. Use "
                "run_isolated_cells (or run_VMSI, which falls back to it automatically) instead."
            )
        return


    def estimate_tau(self):
        """

        Estimate tension vector tau, which is tangent to the edge

        """

        self.build_diff_operators()

        v_coords = np.concatenate([self.vertices['coords'][self.involved_vertices].tolist()])

        e_chord = np.matmul(self.dV, v_coords)

        # Some edges end up with coincident endpoint coordinates (e.g. a
        # degenerate vertex split in remove_fourfold, or duplicate points from
        # noisy segmentation topology), giving a zero-length chord. tau below
        # normalises this vector, so a zero chord silently produces NaN
        # (warnings are suppressed in run_VMSI) that later makes SVD in
        # estimate_pressure's lstsq fail to converge. Drop these edges here,
        # same as the under-connected edges already dropped above.
        good_edges = np.linalg.norm(e_chord, axis=1) > 0
        self.dV = self.dV[good_edges, :]
        self.dC = self.dC[good_edges, :]
        self.cell_pairs = self.cell_pairs[good_edges, :]
        e_chord = e_chord[good_edges, :]

        # classify_cells marks as "involved" any cell sharing a vertex with a
        # bulk cell, but this network is edge-based: a cell meeting a bulk
        # vertex only at a single point (no shared edge with anything here) has
        # an all-zero dC column and never appears in cell_pairs. Left in
        # involved_cells it still gets an x0/pressure row yet contributes zero
        # gradient to the energy term, so the sum-of-pressures linear
        # constraint is free to push its pressure to an arbitrary extreme
        # (physically meaningless - it has no edges to satisfy Young-Laplace
        # against). Drop these cells and compact the operators; the caller
        # (initial_minimization) rebuilds x0 to match the smaller set.
        used = np.unique(self.cell_pairs)
        if 0 < used.size < len(self.involved_cells):
            remap = np.full(len(self.involved_cells), -1, dtype=int)
            remap[used] = np.arange(used.size)
            self.involved_cells = self.involved_cells[used]
            self.bulk_cells = self.bulk_cells[np.isin(self.bulk_cells, self.involved_cells)]
            self.ext_cells = self.ext_cells[np.isin(self.ext_cells, self.involved_cells)]
            self.dC = self.dC[:, used]
            self.cell_pairs = remap[self.cell_pairs]

        self.involved_edges = -1 * np.ones(self.dC.shape[0], dtype=int)

        for i in range(len(self.edges)):
            edge_cells = self.edges.at[i, 'cells']
            # Skip edges that don't border exactly 2 cells (see build_diff_operators)
            if len(edge_cells) != 2:
                continue
            idx = np.where((self.dC[:,np.where(edge_cells[0]==self.involved_cells)[0]] != 0) & (self.dC[:,np.where(edge_cells[1]==self.involved_cells)[0]] != 0))[0]
            self.involved_edges[idx] = i

        # initialise variables
        tau_1 = np.zeros((self.dV.shape[0], 2))
        tau_2 = np.zeros((self.dV.shape[0], 2))

        e_cells = np.zeros((self.dV.shape[0], 2), dtype=int)
        r1 = np.zeros((self.dV.shape[0], 2))
        r2 = np.zeros((self.dV.shape[0], 2))

        for e in range(self.dV.shape[0]):

            e_verts = np.ravel([np.where(self.dV[e,:] == 1), np.where(self.dV[e,:] == -1)])
            e_cells[e,:] = np.ravel([np.where(self.dC[e,:] == 1), np.where(self.dC[e,:] == -1)])

            r1[e,:] = v_coords[e_verts[0],:]
            r2[e,:] = v_coords[e_verts[1],:]

            # If edge is curved, rotate t by pi/2 to get tau
            # t is a vector between the centre of curvature and vertex
            if self.involved_edges[e] >= 0 and self.edges.at[self.involved_edges[e], 'radius'] < np.inf:
                rho = self.edges.at[self.involved_edges[e], 'rho']

                t1 = np.divide(r1[e,:] - rho, np.linalg.norm(r1[e,:] - rho))
                t2 = np.divide(r2[e,:] - rho, np.linalg.norm(r2[e,:] - rho))

                if np.linalg.det(np.array((t1,t2))) > 0:
                    tau_1[e,:] = np.matmul(np.array([[0,-1],[1,0]]), t1)
                    tau_2[e,:] = np.matmul(np.array([[0,1],[-1,0]]), t2)
                else:
                    tau_1[e,:] = -np.matmul(np.array([[0,1],[-1,0]]), t1)
                    tau_2[e,:] = -np.matmul(np.array([[0,-1],[1,0]]), t2)
            # If edge is straight, simply take the edge vector to get tau
            else:
                tau_1[e,:] = -np.divide(e_chord[e,:], np.linalg.norm(e_chord[e,:]))
                tau_2[e,:] = np.divide(e_chord[e,:], np.linalg.norm(e_chord[e,:]))
        return e_cells, tau_1, tau_2, r1, r2


    def estimate_pressure(self, q, e_cells, tau_1, tau_2, r1, r2):
        """

        Initial estimate of pressure
        This provides the initial values for the initial minimisation step

        :param q: (numpy array) generating points q as defined under the VMSI formulation
        :param e_cells: (numpy array) indexes of cell pairs at each edge
        :param tau_1: (numpy array) unit tension vectors at vertex 1 for each edge
        :param tau_2: (numpy array) unit tension vectors at vertex 2 for each edge
        :param r1: (numpy array) co-ordinates of vertex 1 for each edge
        :param r2: (numpy array) co-ordinates of vertex 2 for each edge

        """

        L1 = np.zeros((e_cells.shape[0], q.shape[0]))
        L2 = np.zeros((e_cells.shape[0], q.shape[0]))

        for i in range(e_cells.shape[0]):
            L1[i, e_cells[i,0]] = np.dot(q[e_cells[i,0],:] - r1[i], tau_1[i])
            L1[i, e_cells[i,1]] = -np.dot(q[e_cells[i,1],:] - r1[i], tau_1[i])
            L2[i, e_cells[i,0]] = np.dot(q[e_cells[i,0],:] - r2[i], tau_2[i])
            L2[i, e_cells[i,1]] = -np.dot(q[e_cells[i,1],:] - r2[i], tau_2[i])

        scale = np.mean(np.linalg.norm(q, axis=1))
        b = np.zeros(2*L1.shape[0] + 1)
        b[-1] = scale

        # Initial estimates generated by minimising sum of
        # p_a((q_a-r_i) * tau_i) = p_b((q_b-r_i) * tau_i) for cells a and b at vertex i
        # Overall scale maintained such that mean pressure = 1

        L = np.vstack((L1, L2, np.array(np.divide(np.ones(q.shape[0]), q.shape[0]))))

        p = np.linalg.lstsq(L,b)[0]
        p = np.divide(p, np.mean(p))
        return p


    def generate_circular_arcs(self):
        """

        Construct circular arc for each edge and use these instead of raw segmented edges for minimization

        """

        # Each arc has a number of pixels equal to the average edge length
        self.avg_edge_length = int(np.median([self.edges.at[edge, 'pixels'].shape[0] for edge in self.involved_edges]))
        self.edgearc_x = np.zeros((len(self.involved_edges), self.avg_edge_length))
        self.edgearc_y = np.zeros((len(self.involved_edges), self.avg_edge_length))

        for i in range(len(self.involved_edges)):
            r = np.array([self.vertices.at[self.edges.at[self.involved_edges[i], 'verts'][0], 'coords'],
                          self.vertices.at[self.edges.at[self.involved_edges[i], 'verts'][1], 'coords']])
            # Construct circular arc for curved edges
            if self.edges.at[self.involved_edges[i], 'radius'] < np.inf:
                r_centered = np.subtract(r, self.edges.at[self.involved_edges[i], 'rho'])

                # This returns between [0, pi] so we should aways get a convex angle as expected
                theta = np.arccos(np.divide(np.dot(r_centered[0,:], r_centered[1,:]),
                                            np.multiply(np.linalg.norm(r_centered[0,:]), np.linalg.norm(r_centered[1,:]))))
                if np.linalg.det(r_centered) < 0:
                    r_centered = r_centered[[1,0],:]
                theta_range = np.linspace(0, theta, self.avg_edge_length)

                self.edgearc_x[i] = self.edges.at[self.involved_edges[i], 'rho'][0] + (r_centered[0,0]*np.cos(theta_range) - r_centered[0,1]*np.sin(theta_range))
                self.edgearc_y[i] = self.edges.at[self.involved_edges[i], 'rho'][1] + (r_centered[0,0]*np.sin(theta_range) + r_centered[0,1]*np.cos(theta_range))
            # Construct straight line for straight edges
            else :
                chord = r[1,:] - r[0,:]

                spacing = np.linspace(0, 1, self.avg_edge_length)

                self.edgearc_x[i] = r[0,0] + spacing*chord[0]
                self.edgearc_y[i] = r[0,1] + spacing*chord[1]
        return


    def estimate_theta(self, x):
        """
        Initialize theta, defined as p_a * z^2_a for each cell a

        :param x: (numpy array) with dimensions [num_cells x 3] containing q in x[0:2,:] and p in x[2,:]
        :return: (numpy array) with dimensions [num_cells x 1] containing initialized values of theta
        """
        q = x[:,0:2]
        p = x[:,2]

        dQ = gather_pairs(q, self.cell_pairs)
        q_sq = np.sum(np.power(dQ, 2), axis=1)

        dP = gather_pairs(p, self.cell_pairs)
        rho = gather_pairs(np.multiply(q.T, p).T, self.cell_pairs) / dP[:,None]

        # Per-edge vertex coordinates for involved_edges, vectorised instead of a
        # per-edge Python loop with repeated pandas .at[] lookups.
        verts_arr = np.stack(self.edges.loc[self.involved_edges, 'verts'].to_numpy())
        coords_arr = np.array(self.vertices['coords'].tolist())
        r1 = coords_arr[verts_arr[:,0]]
        r2 = coords_arr[verts_arr[:,1]]
        # 0.5*(a^2+b^2) is exactly mean(power([a,b],2)) for the two-element case above.
        r = 0.5 * (np.sum(np.power(r1 - rho, 2), axis=1) + np.sum(np.power(r2 - rho, 2), axis=1))
        r_flat = p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * q_sq

        r = np.multiply(r, np.power(dP, 2))

        A = dense_jacobian_pairs(dP, self.cell_pairs, len(self.involved_cells))
        b = r_flat - r

        theta = np.linalg.lstsq(np.vstack((A, np.ones(A.shape[1]))),np.concatenate((b, np.array([0]))))[0]

        return theta


    def initial_minimization(self):
        """

        Initial minimisation step to ensure that vector t (between q_a and r_i for vertex i at cell a) is orthogonal
        to tension vector tau.
        This prevents the main minimisation step converging on the trivial solution q_a=q_b, p_a=p_b

        """
        # Initialize q
        x0 = self.classify_cells()

        # Initialize tau
        e_cells, tau_1, tau_2, r1, r2 = self.estimate_tau()

        # estimate_tau (via build_diff_operators) may have dropped degree-0
        # cells from involved_cells after classify_cells already sized x0 to
        # the old count - rebuild x0's rows against the pruned set so every
        # downstream array (q0/p0/theta0, dC columns, cell_pairs indices) stays
        # consistently sized.
        if x0.shape[0] != len(self.involved_cells):
            x0 = np.vstack([np.stack(self.cells['centroids'][self.involved_cells]).T,
                            np.zeros(len(self.involved_cells))]).T

        # Initialize pressure
        x0[:,2] = self.estimate_pressure(x0[:,0:2], e_cells, tau_1, tau_2, r1, r2)

        q0 = x0[:,0:2]
        p0 = x0[:,2]

        b0 = gather_pairs(np.multiply(q0.T,p0).T, self.cell_pairs)
        delta_p0 = gather_pairs(p0, self.cell_pairs)

        # Get initial values for t_i and t_j
        t1_0 = b0 - (np.multiply(r1.T,delta_p0).T)
        t2_0 = b0 - (np.multiply(r2.T,delta_p0).T)

        if self.optimiser == 'nlopt':
            import nlopt
            scale = 0.5 * (np.mean(np.linalg.norm(t1_0, axis=1)) + np.mean(np.linalg.norm(t2_0, axis=1)))

            # nlopt doesn't reliably return its best-found point (see the
            # try/except around init_opt.optimize below), so the last
            # evaluated x is captured here instead of round-tripping through
            # a CSV file on every single evaluation (up to thousands of times
            # per fit) - same information, no disk I/O, and no precision loss
            # from np.savetxt's default text formatting.
            last_x = [None]
            # Holds this stage's tqdm bar (or None) so the objective can tick it
            # once per evaluation; set just before optimize(), cleared after.
            _pbar = [None]

            # energy() and nonlinear_con() below are both evaluated by nlopt at the
            # same trial point on essentially every iteration (a gradient-based
            # constrained solver needs the objective and constraint - and often
            # their gradients - at the same x before it can take a step), and both
            # independently recomputed the same b/delta_p from scratch. Cache it
            # keyed on the raw x bytes so whichever of the two runs second reuses
            # the first's result instead of redoing the same gather.
            _bp_cache = {'key': None, 'b': None, 'delta_p': None}
            def _compute_bp(x_flat, q, p):
                key = x_flat.tobytes()
                if _bp_cache['key'] != key:
                    _bp_cache['key'] = key
                    _bp_cache['b'] = gather_pairs(np.multiply(q.T,p).T, self.cell_pairs)
                    _bp_cache['delta_p'] = gather_pairs(p, self.cell_pairs)
                return _bp_cache['b'], _bp_cache['delta_p']

            # Define energy function for initial minimisation
            def energy(x, grad=np.array([])):
                last_x[0] = x.copy()
                x_flat = x
                x = x.reshape(x0.shape, order='F')
                q = x[:,0:2]
                p = x[:,2]

                b, delta_p = _compute_bp(x_flat, q, p)

                t1 = np.divide((b - (np.multiply(r1.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)).T
                t2 = np.divide((b - (np.multiply(r2.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)).T

                E = 0.5 * np.mean(np.power(np.sum(t1 * tau_1, axis=1), 2) +
                                  np.power(np.sum(t2 * tau_2, axis=1), 2))

                if grad.size > 0:
                    ip1 = np.sum(t1 * tau_1, axis=1)
                    ip2 = np.sum(t2 * tau_2, axis=1)

                    drX1 = sx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r1[:,0], r1[:,1])
                    drX2 = sx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r2[:,0], r2[:,1])
                    drY1 = sy_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r1[:,0], r1[:,1])
                    drY2 = sy_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], r2[:,0], r2[:,1])

                    dE = np.multiply(ip1 * tau_1[:,0], drX1.T).T + np.multiply(ip1 * tau_1[:,1], drY1.T).T + \
                         np.multiply(ip2 * tau_2[:,0], drX2.T).T + np.multiply(ip2 * tau_2[:,1], drY2.T).T
                    dE = dE / self.dC.shape[0]

                    rows = np.concatenate([self.cell_pairs[:,0],self.cell_pairs[:,0]+self.dC.shape[1],self.cell_pairs[:,0]+2*self.dC.shape[1],
                                           self.cell_pairs[:,1],self.cell_pairs[:,1]+self.dC.shape[1],self.cell_pairs[:,1]+2*self.dC.shape[1]])

                    dE = np.bincount(rows, weights=np.ravel(dE,order='F'))
                    grad[:] = dE

                if _pbar[0] is not None:
                    _pbar[0].update(1)
                    _pbar[0].set_postfix_str(f"E={E:.6g}", refresh=False)
                elif self.verbose:
                    print(E)
                return E

            # Define nonlinear constraint for initial optimisation
            def nonlinear_con(x, grad=np.array([])):
                x_flat = x
                x = x.reshape(x0.shape, order='F')
                q = x[:,0:2]
                p = x[:,2]

                b, delta_p = _compute_bp(x_flat, q, p)

                l1 = np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)
                l2 = np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)

                E = 0.5*(np.mean(l1) + np.mean(l2)) - scale

                if grad.size > 0:
                    t1 = np.divide((b - (np.multiply(r1.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r1.T,delta_p).T), axis=1)).T
                    t2 = np.divide((b - (np.multiply(r2.T,delta_p).T)).T,np.linalg.norm(b - (np.multiply(r2.T,delta_p).T), axis=1)).T

                    drX1 = rx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,1],0], r1[:,0])
                    drX2 = rx_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,1],0], r2[:,0])
                    drY1 = ry_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],1], r1[:,1])
                    drY2 = ry_grad(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],1], r2[:,1])


                    dE = 0.5 * (np.multiply(t1[:,0],drX1.T).T + np.multiply(t1[:,1],drY1.T).T) + \
                         0.5 * (np.multiply(t2[:,0],drX2.T).T + np.multiply(t2[:,1],drY2.T).T)

                    rows = np.concatenate([self.cell_pairs[:,0],self.cell_pairs[:,0]+self.dC.shape[1],self.cell_pairs[:,0]+2*self.dC.shape[1],
                                           self.cell_pairs[:,1],self.cell_pairs[:,1]+self.dC.shape[1],self.cell_pairs[:,1]+2*self.dC.shape[1]])

                    grad[:] = np.bincount(rows, weights=np.ravel(dE/self.dC.shape[0],order='F'))
                return E

            # Define linear constraint for initial optimisation
            def linear_con(x, grad=np.array([])):
                Aeq = np.concatenate((np.zeros(x0.shape[0]*2), np.ones(x0.shape[0])))
                E = np.dot(Aeq, x) - (np.mean(p0)*p0.shape[0])

                if grad.size > 0:
                    grad[:] = np.concatenate((np.zeros(2*x0.shape[0]), np.ones(x0.shape[0])))
                return float(E)

            # Configure optimiser
            local_opt = nlopt.opt(nlopt.LD_LBFGS, x0.size)
            # AUGLAG delegates each penalised subproblem to this local optimiser.
            # Without its own stopping criteria, a single (possibly
            # ill-conditioned) inner LBFGS solve can run for a very long time
            # before the outer AUGLAG loop's maxeval is even checked again -
            # in practice indistinguishable from a hang. Give it explicit,
            # bounded stopping criteria.
            local_opt.set_ftol_rel(1e-6)
            local_opt.set_maxeval(500)
            init_opt = nlopt.opt(nlopt.AUGLAG, x0.size)
            init_opt.set_local_optimizer(local_opt)
            init_opt.set_min_objective(energy)
            lb = np.concatenate((-np.inf*np.ones(x0.shape[0]), -np.inf*np.ones(x0.shape[0]), 0.001*np.ones(x0.shape[0])))
            ub = np.concatenate((np.inf*np.ones(x0.shape[0]), np.inf*np.ones(x0.shape[0]), 2000*np.ones(x0.shape[0])))
            init_opt.set_lower_bounds(lb)
            init_opt.set_upper_bounds(ub)
            init_opt.add_inequality_constraint(nonlinear_con, 1e-6)
            init_opt.add_equality_constraint(linear_con, 1e-6)
            init_maxeval = 2000
            init_opt.set_maxeval(init_maxeval)

            # Optimisation
            # Nlopt can't handle initial values outside bounds so clip values before optimisation
            # Nlopt can raise a bare RuntimeError (e.g. roundoff-limited: it
            # can no longer improve given floating-point precision) instead
            # of returning normally once converged. The code already doesn't
            # trust optimize()'s return value - it uses the last evaluated
            # point captured by energy() above regardless - so treat that
            # exception the same way: not a failure, just an early
            # (already-converged) exit from this particular call.
            # nlopt's own exception class (its C++ std::runtime_error,
            # exposed as a plain lowercase "runtime_error" type) isn't a
            # subclass of Python's builtin RuntimeError, so catch broadly -
            # any exception here means "stop trying to optimize further",
            # which is exactly what falling back to last_x below already assumes.
            _pbar[0] = _make_stage_pbar("Initial minimization (q, p)", init_maxeval, self.verbose)
            try:
                init_opt.optimize(np.clip(x0.ravel(order='F'), lb, ub))
            except Exception:
                pass
            if _pbar[0] is not None:
                _pbar[0].close()
            _pbar[0] = None

            # For larger systems, the nlopt optimiser will not converge to the desired tolerance and does not return the results obtained at the final step
            # To get around this, use the last point energy() evaluated
            x = last_x[0].reshape(x0.shape, order='F')

            q = x[:,0:2]
            p = x[:,2]

            # theta_energy below derives arc curvature from the pressure
            # difference between neighbouring cells (Young-Laplace), dividing
            # by dP = p[a] - p[b] (and 1/dP^2) both in its value and in its
            # gradient (radius_grad_theta). For a relatively homogeneous
            # tissue it's physically normal for most neighbouring cells to
            # have very similar pressure - that's not a data defect, it's
            # dP -> 0 approaching the correct straight-edge (infinite
            # radius) limit. The bug is that this formulation divides by dP
            # directly instead of handling that limit gracefully. Rather than
            # trying to nudge p apart (which doesn't scale: with pressure
            # this uniform, most cells sit in multiple too-close pairs at
            # once, so a single vectorised pass silently drops all but the
            # last correction per cell - confirmed by diagnostics showing the
            # fix had no effect), dP is now clamped at the point of use in
            # theta_energy and inside radius_grad_theta/radius_grad
            # themselves (see below), which is conflict-free by construction.
            min_dp = 1e-2

            if self.verbose:
                dP_diag = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
                dQ_diag = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
                QL_diag = np.sum(np.power(dQ_diag, 2), axis=1)
                print(f"[diag] cell_pairs: {len(dP_diag)}, min|dP|: {np.min(np.abs(dP_diag)):.6g}, "
                      f"pairs<min_dp: {int(np.sum(np.abs(dP_diag) < min_dp))}, "
                      f"p range: [{np.min(p):.6g}, {np.max(p):.6g}], "
                      f"q finite: {np.all(np.isfinite(q))}, p finite: {np.all(np.isfinite(p))}, "
                      f"QL range: [{np.min(QL_diag):.6g}, {np.max(QL_diag):.6g}]")

            self.generate_circular_arcs()

            # Once q, p are optimised, perform initial optimisation for theta
            theta0 = self.estimate_theta(x)

            # See last_x above for why this is captured in memory instead of
            # written to disk on every evaluation.
            last_theta = [None]
            # See _pbar above - same per-stage tqdm holder for the theta stage.
            _pbar = [None]

            # p and q are fixed for this whole stage (only theta is being
            # optimised), so everything below that depends on p/q alone - not
            # theta - only needs computing once here, rather than on every one
            # of the (potentially hundreds of) evaluations of theta_energy/
            # theta_neqlincon below. Note theta_energy clamps dP away from its
            # zero-singularity but theta_neqlincon never did - that's a
            # pre-existing inconsistency between the two, kept exactly as-is
            # (dP_clamped vs dP_raw) rather than unified as a side effect of
            # this hoisting.
            dP_raw = p[self.cell_pairs[:,0]] - p[self.cell_pairs[:,1]]
            dP_clamped = np.where(dP_raw >= 0, np.maximum(dP_raw, min_dp), np.minimum(dP_raw, -min_dp))
            dQ_fixed = q[self.cell_pairs[:,0],:] - q[self.cell_pairs[:,1],:]
            QL_fixed = np.sum(np.power(dQ_fixed, 2), axis=1)
            pp_QL_fixed = p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]] * QL_fixed
            rho_fixed = gather_pairs(np.multiply(p, q.T).T, self.cell_pairs) / dP_clamped[:,None]
            # theta_neqlincon's Jacobian doesn't depend on theta either (it's
            # ±dP_raw at each edge's two cell columns, same every call), so
            # its dense (n_edges, n_cells) output can be built once too.
            theta_con_jacobian = dense_jacobian_pairs(dP_raw, self.cell_pairs, len(self.involved_cells))

            # Define energy function for theta optimisation
            def theta_energy(theta, grad=np.array([])):
                last_theta[0] = theta.copy()

                dP = dP_clamped
                dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]

                rho = rho_fixed
                r_sq = np.divide((pp_QL_fixed - (dP * dT)),np.power(dP, 2))
                # r_sq<0 doesn't catch NaN (any comparison with NaN is False),
                # so a stray NaN would otherwise pass straight through
                # unclamped into sqrt() below. ~(r_sq>=0) treats NaN the same
                # as a degenerate/negative discriminant.
                ind_z = ~(r_sq>=0)
                r_sq[ind_z] = 0

                r = np.sqrt(r_sq)

                delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
                delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

                dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

                E = 0.5 * np.mean(np.sum(np.power(np.subtract(dMag.T, r).T, 2), axis=1))

                if grad.size > 0:
                    d = np.subtract(dMag.T, r).T

                    avg_d = np.sum(d, axis=1)
                    dR = radius_grad_theta(p[self.cell_pairs[:,0]], p[self.cell_pairs[:,1]], q[self.cell_pairs[:,0],0], q[self.cell_pairs[:,0],1], q[self.cell_pairs[:,1],0], q[self.cell_pairs[:,1],1], theta[self.cell_pairs[:,0]], theta[self.cell_pairs[:,1]])
                    dR[ind_z,:] = 0
                    # radius_grad_theta recomputes the r_sq<0 discriminant
                    # internally via a reciprocal-then-multiply (1/dP^2 * ...)
                    # rather than the direct division used for r_sq/ind_z
                    # above; near a r_sq=0 crossing these two paths can round
                    # to opposite signs, so sqrt(negative) = NaN can slip
                    # through the ind_z mask. Catch any leftover non-finite
                    # entries directly - they correspond to the same
                    # degenerate (near-zero-discriminant) edges ind_z already
                    # intends to zero out.
                    dR[~np.isfinite(dR)] = 0

                    dE = np.divide(-np.multiply(avg_d, dR.T).T,self.dC.shape[0])
                    rows = np.concatenate([self.cell_pairs[:,0], self.cell_pairs[:,1]])

                    grad[:] = np.bincount(rows, weights=np.ravel(dE,order='F'))
                if _pbar[0] is not None:
                    _pbar[0].update(1)
                    _pbar[0].set_postfix_str(f"E={E:.6g}", refresh=False)
                elif self.verbose:
                    print(E)
                return float(E)

            def theta_eqlincon(theta, grad=np.array([])):
                Aeq = np.ones((1, len(self.involved_cells)))
                E = np.dot(Aeq, theta)

                if grad.size > 0:
                    grad[:] = np.ones(theta0.shape[0])
                return float(E)

            # Define nonlinear constraint for theta optimisation
            def theta_neqlincon(result, theta, grad=np.array([])):
                # np.dot(A, theta) where A = dense_jacobian_pairs(dP_raw, ...) is
                # exactly dP_raw * dT for each edge (A's row e is +dP_raw[e] at
                # cell_pairs[e,0], -dP_raw[e] at cell_pairs[e,1]), so this never
                # needs to materialise A itself just to compute the constraint value.
                dT = theta[self.cell_pairs[:,0]] - theta[self.cell_pairs[:,1]]
                E = (dP_raw * dT) - pp_QL_fixed
                result[:] = E
                if grad.size > 0:
                    grad[:] = theta_con_jacobian
                return

            if (theta_energy(np.zeros_like(theta0)) < theta_energy(theta0)):
                theta0 = np.zeros_like(theta0)

            # Configure optimiser
            theta_local_opt = nlopt.opt(nlopt.LD_LBFGS, theta0.size)
            # See the analogous local_opt above: without its own stopping
            # criteria this inner solve can run effectively forever.
            theta_local_opt.set_ftol_rel(1e-6)
            theta_local_opt.set_maxeval(500)
            theta_opt = nlopt.opt(nlopt.AUGLAG, theta0.size)
            theta_opt.set_local_optimizer(theta_local_opt)
            theta_opt.set_ftol_abs(1e-5)
            theta_opt.set_min_objective(theta_energy)
            theta_opt.add_inequality_mconstraint(theta_neqlincon, 1e-5*np.ones(self.dC.shape[0]))
            # This was previously disabled due to NLopt generic failures.
            # Re-enabling it (once the NaN leak from radius_grad_theta's
            # gradient was fixed - see the dR non-finite guard above) turned
            # out not to change the optimisation trajectory at all for a
            # single, non-tiled run: it only pins the scale of theta, and
            # theta_neqlincon's own gradient already keeps that scale from
            # drifting in practice. Kept for correctness (relevant once
            # tiling recombines multiple fits), but it is not what was
            # causing the inf/nan runaway below.
            theta_opt.add_equality_constraint(theta_eqlincon, 1e-5)
            # theta itself has no bounds, unlike q/p above, so nothing stops
            # the local LBFGS solver from pushing some theta difference
            # toward an unconstrained direction until r_sq genuinely
            # overflows float64 (inf), with a transitional inf-inf giving the
            # nan seen right before it. Give theta generously wide bounds -
            # far too loose to affect any real solution, but tight enough to
            # hard-stop the runaway well before it reaches actual overflow.
            theta_bound = 1e8
            theta_opt.set_lower_bounds(-theta_bound * np.ones(theta0.size))
            theta_opt.set_upper_bounds(theta_bound * np.ones(theta0.size))
            theta_maxeval = 2000
            theta_opt.set_maxeval(theta_maxeval)

            # Optimise
            # Nlopt can't handle initial values outside bounds so clip values before optimisation
            # See the matching try/except around init_opt.optimize above -
            # same reasoning, and same broad Exception catch since nlopt's
            # own exception class isn't a Python RuntimeError subclass.
            _pbar[0] = _make_stage_pbar("Theta minimization", theta_maxeval, self.verbose)
            try:
                theta_opt.optimize(np.clip(theta0, -theta_bound, theta_bound))
            except Exception:
                pass
            if _pbar[0] is not None:
                _pbar[0].close()
            _pbar[0] = None

            theta = last_theta[0]

        elif self.optimiser == 'matlab':
            # Equivalent optimisation steps for matlab optimiser instead
            import matlab
            p_mat = matlab.double(p0.tolist())
            q_mat = matlab.double((q0).tolist())
            d0_mat = matlab.double(self.dC.tolist())
            bCells_mat = matlab.double((self.cell_pairs+1).tolist())
            r1_mat = matlab.double((r1).tolist())
            r2_mat = matlab.double((r2).tolist())
            t1_mat = matlab.double(tau_1.tolist())
            t2_mat = matlab.double(tau_2.tolist())

            x = self.eng.initial_optimization(p_mat, q_mat, d0_mat, bCells_mat, r1_mat, r2_mat, t1_mat, t2_mat)
            x = np.array(x)

            q = x[:,0:2]
            p = x[:,2]

            self.generate_circular_arcs()
            theta0 = self.estimate_theta(x)

            p_mat = matlab.double(p.tolist())
            q_mat = matlab.double((q).tolist())
            theta0_mat = matlab.double(theta0.tolist())
            rBX_mat = matlab.double((self.edgearc_x).tolist())
            rBY_mat = matlab.double((self.edgearc_y).tolist())

            theta = self.eng.theta_optimization(theta0_mat, q_mat, p_mat, d0_mat, bCells_mat, rBX_mat, rBY_mat)
            theta = np.array(theta)
        return q, p, theta


    def fit(self):

        """

        Perform the minimization of equation 5 with respect to the variables (q, z, p)

        """
        self.prepare_data()

        if self.verbose:
            print("Initial minimization")
        # Perform initial minimization for p, q and theta
        q0, p0, theta0 = self.initial_minimization()
        X0 = np.vstack([q0.T, theta0.squeeze(), p0.squeeze()]).T

        q0 = X0[:,0:2]
        theta0 = X0[:,2]
        p0 = X0[:,3]

        if self.verbose:
            print("Main minimization")

        if self.optimiser == 'nlopt':
            import nlopt
            # See the matching clamp/comment in initial_minimization's
            # theta_energy - same floor, kept consistent so both stages
            # regularise the dP=0 singularity the same way.
            min_dp = 1e-2
            # See last_x in initial_minimization for why this is captured in
            # memory instead of written to disk on every evaluation.
            last_X = [None]
            # See _pbar in initial_minimization - per-stage tqdm holder.
            _pbar = [None]

            # objective() and nonlinear_con() below are both evaluated by nlopt
            # at the same trial point on essentially every iteration, and both
            # independently gather the same dP/dT/dQ/QL/pp from X. Cache them
            # keyed on the raw X bytes (same pattern as initial_minimization's
            # _compute_bp) so whichever runs second reuses the first's work.
            # Note objective() clamps dP away from zero but nonlinear_con()
            # never did - a pre-existing inconsistency between the two, kept
            # exactly as-is: this cache holds the shared *raw* dP, and each
            # function does its own clamping (or not) on top.
            _shared_cache = {'key': None}
            def _compute_shared(x_flat, q, p, theta):
                key = x_flat.tobytes()
                if _shared_cache['key'] != key:
                    _shared_cache['key'] = key
                    _shared_cache['dP_raw'] = gather_pairs(p, self.cell_pairs)
                    _shared_cache['dT'] = gather_pairs(theta, self.cell_pairs)
                    dQ = gather_pairs(q, self.cell_pairs)
                    _shared_cache['dQ'] = dQ
                    _shared_cache['QL'] = np.sum(np.power(dQ, 2), axis=1)
                    _shared_cache['pp'] = p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]]
                return (_shared_cache['dP_raw'], _shared_cache['dT'], _shared_cache['dQ'],
                        _shared_cache['QL'], _shared_cache['pp'])

            def objective(X, grad=np.array([])):
                last_X[0] = X.copy()
                X_flat = X
                X = X.reshape(X0.shape, order='F')

                q = X[:,0:2]
                p = X[:,3]
                theta = X[:,2]

                dP_raw, dT, dQ, QL, pp = _compute_shared(X_flat, q, p, theta)
                # Clamp away from the dP=0 singularity - dP is a live
                # optimisation variable here (unlike in theta_energy), so
                # this also protects against the solver transiently passing
                # through near-equal-pressure configurations mid-search. Kept
                # consistent with the same clamp in radius_grad's gradient.
                dP = np.where(dP_raw >= 0, np.maximum(dP_raw, min_dp), np.minimum(dP_raw, -min_dp))

                rho = gather_pairs(np.multiply(p, q.T).T, self.cell_pairs) / dP[:,None]
                r_sq = np.divide((pp * QL) - (dP * dT), np.power(dP, 2))
                # r_sq<=0 doesn't catch NaN (any comparison with NaN is
                # False), so a stray NaN would otherwise pass straight
                # through unclamped into sqrt() below. ~(r_sq>0) treats NaN
                # the same as a degenerate/non-positive discriminant.
                ind_z = ~(r_sq>0)
                r_sq[ind_z] = 0

                r = np.sqrt(r_sq)

                delta_x = np.subtract(rho[:,0],self.edgearc_x.T).T
                delta_y = np.subtract(rho[:,1],self.edgearc_y.T).T

                dMag = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

                E = 0.5 * np.mean(np.sum(np.power(np.subtract(dMag.T, r).T, 2), axis=1))

                if grad.size>0:

                    d = np.subtract(dMag.T, r).T

                    dRhoX = rho_x_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,1],0])
                    dRhoY = rho_y_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],1])
                    dR = radius_grad(p[self.cell_pairs[:,0]],p[self.cell_pairs[:,1]],q[self.cell_pairs[:,0],0],q[self.cell_pairs[:,0],1],q[self.cell_pairs[:,1],0],q[self.cell_pairs[:,1],1],theta[self.cell_pairs[:,0]],theta[self.cell_pairs[:,1]])

                    dNormX = np.sum(np.multiply(delta_x,np.divide(d, dMag)),axis=1)
                    dNormY = np.sum(np.multiply(delta_y,np.divide(d, dMag)),axis=1)

                    avg_d = np.sum(d, axis=1)
                    dR[ind_z] = 0
                    # See the analogous guard in theta_energy: radius_grad
                    # recomputes the r_sq<=0 discriminant via a
                    # reciprocal-then-multiply, which can round to a
                    # different sign than the direct division used for
                    # r_sq/ind_z above right at a zero crossing, letting
                    # sqrt(negative) = NaN slip past the ind_z mask.
                    dR[~np.isfinite(dR)] = 0

                    dE = np.divide(np.multiply(dNormX, dRhoX.T).T+np.multiply(dNormY, dRhoY.T).T-np.multiply(avg_d, dR.T).T,self.dC.shape[0])
                    rows = np.concatenate([self.cell_pairs[:,0],self.cell_pairs[:,0]+self.dC.shape[1],self.cell_pairs[:,0]+2*self.dC.shape[1],self.cell_pairs[:,0]+3*self.dC.shape[1],
                                           self.cell_pairs[:,1],self.cell_pairs[:,1]+self.dC.shape[1],self.cell_pairs[:,1]+2*self.dC.shape[1],self.cell_pairs[:,1]+3*self.dC.shape[1]])
                    dE = np.bincount(rows, weights=np.ravel(dE,order='F'))
                    grad[:] = dE.ravel()

                if _pbar[0] is not None:
                    _pbar[0].update(1)
                    _pbar[0].set_postfix_str(f"E={E:.6g}", refresh=False)
                elif self.verbose:
                    print(E)
                return E

            def nonlinear_con(result, X, grad=np.array([])):
                X_flat = X
                X = X.reshape(X0.shape, order='F')

                q = X[:,0:2]
                p = X[:,3]
                theta = X[:,2]

                dP_raw, dT, dQ, QL, pp = _compute_shared(X_flat, q, p, theta)
                result[:] = (dP_raw * dT) - (pp * QL)

                if grad.size>0:
                    n_cells = len(self.involved_cells)
                    # Calculate jacobian of nonlinear constraints. gX/gY/gTh use the
                    # ±w-at-the-edge's-two-cells pattern (dense_jacobian_pairs); gP's
                    # second term instead needs the *same* +w at both cells (from
                    # np.abs(dC) rather than dC itself), so it's built directly.
                    gX = dense_jacobian_pairs(-2*pp*dQ[:,0], self.cell_pairs, n_cells)
                    gY = dense_jacobian_pairs(-2*pp*dQ[:,1], self.cell_pairs, n_cells)
                    gTh = dense_jacobian_pairs(dP_raw, self.cell_pairs, n_cells)

                    QLpp = np.zeros((len(dP_raw), n_cells))
                    edge_idx = np.arange(len(dP_raw))
                    QLpp[edge_idx, self.cell_pairs[:,0]] = QL * pp
                    QLpp[edge_idx, self.cell_pairs[:,1]] = QL * pp
                    gP = dense_jacobian_pairs(dT, self.cell_pairs, n_cells) - np.divide(QLpp, p)
                    grad[:] = np.hstack([gX,gY,gTh,gP])
                return

            def linear_con(result, X, grad=np.array([])):
                Aeq = np.vstack((np.concatenate((np.zeros(3*theta0.shape[0]), np.ones(theta0.shape[0])/theta0.shape[0])),
                                 np.concatenate((np.zeros(2*theta0.shape[0]), np.ones(theta0.shape[0])/theta0.shape[0], np.zeros(theta0.shape[0])))))
                E = np.dot(Aeq, X) - np.array([np.mean(X0[:,3]), np.mean(X0[:,2])])
                if grad.size > 0:
                    grad[:] = np.vstack((np.concatenate((np.zeros(3*self.dC.shape[1]), np.ones(self.dC.shape[1])/self.dC.shape[1])),
                                         np.concatenate((np.zeros(2*self.dC.shape[1]), np.ones(self.dC.shape[1])/self.dC.shape[1], np.zeros(self.dC.shape[1])))))
                result[:] = E
                return

            local_opt = nlopt.opt(nlopt.LD_LBFGS, X0.size)
            # See the analogous local_opt in initial_minimization: without its
            # own stopping criteria this inner solve can run effectively forever.
            local_opt.set_ftol_rel(1e-6)
            local_opt.set_maxeval(500)

            main_opt = nlopt.opt(nlopt.AUGLAG, X0.size)
            main_opt.set_local_optimizer(local_opt)
            main_opt.set_min_objective(objective)
            lb = np.concatenate((-np.inf*np.ones(3*theta0.shape[0]), 0.001*np.ones(theta0.shape[0])))
            ub = np.concatenate((np.inf*np.ones(3*theta0.shape[0]), 1000*np.ones(theta0.shape[0])))
            main_opt.set_lower_bounds(lb)
            main_opt.set_upper_bounds(ub)
            main_opt.add_inequality_mconstraint(nonlinear_con, 1e-6*np.ones(self.dC.shape[0]))
            main_opt.add_equality_mconstraint(linear_con, 1e-6*np.ones(2))
            main_maxeval = 2000
            main_opt.set_maxeval(main_maxeval)

            # See the matching try/except around init_opt/theta_opt.optimize
            # in initial_minimization - same reasoning: X falls back to the
            # last point objective() evaluated regardless, so an exception
            # from nlopt itself (e.g. roundoff-limited, which is what the
            # value flatlining right before the crash indicates) isn't a
            # failure here. Caught broadly since nlopt's own exception class
            # isn't a Python RuntimeError subclass (confirmed by this exact
            # crash escaping a narrower `except RuntimeError`).
            _pbar[0] = _make_stage_pbar("Main minimization", main_maxeval, self.verbose)
            try:
                main_opt.optimize(np.clip(X0.ravel(order='F'),lb,ub))
            except Exception:
                pass
            if _pbar[0] is not None:
                _pbar[0].close()
            _pbar[0] = None

            X = last_X[0].reshape(X0.shape, order='F')
        elif self.optimiser == 'matlab':
            import matlab

            X0_mat = matlab.double(X0.tolist())
            bCells_mat = matlab.double((self.cell_pairs+1).tolist())
            d0_mat = matlab.double(self.dC.tolist())
            rBX_mat = matlab.double((self.edgearc_x).tolist())
            rBY_mat = matlab.double((self.edgearc_y).tolist())

            res = self.eng.main_minimization(bCells_mat, d0_mat, rBX_mat, rBY_mat, X0_mat, float(self.avg_edge_length))

            X = np.array(res)
            X = X.reshape(X0.shape, order='F')

            self.eng.quit()

        q = X[:,0:2]
        p = X[:,3]
        theta = X[:,2]

        # compute the tensions
        T = self.get_tensions(q, p, theta)
        self.upload_mechanics(p, T, q, theta)
        self.infer_isolated_cells()
        self.infer_cluster_cells()
        return


    def get_tensions(self, q, p, theta):

        """

        applies the Young-Laplace law to obtain the tensions at every edge

        """
        T = gather_pairs(q, self.cell_pairs)
        T = np.sum(np.power(T, 2), axis=1)
        T = T * np.abs(p[self.cell_pairs[:,0]] * p[self.cell_pairs[:,1]])
        T = T - np.multiply(gather_pairs(p, self.cell_pairs), gather_pairs(theta, self.cell_pairs))
        # Residual optimizer imprecision (nlopt's ftol_rel=1e-6 tolerance) can leave this
        # term slightly negative at convergence even though the nonlinear constraint keeps
        # it >=0 in theory; sqrt(negative) silently gives NaN, which then poisons the
        # per-cell stress tensor for every cell sharing that edge. Clamp at the same
        # tolerance used elsewhere in this solve rather than let it go negative.
        T = np.sqrt(np.maximum(T, 0))
        return T

    def upload_mechanics(self, p, T, q, theta):
        """

        Match tensions and pressures to edges and cells

        """

        self.cells.pressure[np.sort(self.involved_cells)] = p[np.argsort(self.involved_cells)]
        self.cells.qx[np.sort(self.involved_cells)] = q[np.argsort(self.involved_cells),0]
        self.cells.qy[np.sort(self.involved_cells)] = q[np.argsort(self.involved_cells),1]
        self.cells.theta[np.sort(self.involved_cells)] = theta[np.argsort(self.involved_cells),]

        for i in range(len(self.edges)):
            edge_cells = self.edges.at[i, 'cells']

            # Skip edges that don't border exactly 2 cells (see build_diff_operators)
            if len(edge_cells) == 2 and all(np.isin(edge_cells, self.involved_cells)) and np.isin(i, self.involved_edges):
                cell_ind1 = np.where(self.involved_cells == edge_cells[0])[0]
                cell_ind2 = np.where(self.involved_cells == edge_cells[1])[0]

                edge_ind = np.where(np.squeeze((self.dC[:,cell_ind1] != 0) & (self.dC[:,cell_ind2] != 0), axis=1))[0]
                self.edges.at[i, 'tension'] = T[edge_ind]
        return

    def return_tensions(self):
        return self.edges.tension.values

    def return_pressures(self):
        return self.cells.pressure.values[1:]

    def infer_isolated_cells(self, background_pressure=None, tension=None):
        """
        Directly infer pressure (and a representative tension) for cells that
        touch only the background (cell 0) — i.e. suspended / isolated cells
        with no real neighbours to jointly solve tension/pressure against.

        A genuinely isolated cell's boundary curvature only constrains the
        ratio of its pressure difference from the background to its own
        interfacial tension (delta_p = tension * curvature) - not each
        independently, since there's no cell-cell junction network to
        resolve them jointly (see run_isolated_cells for the full derivation
        and rationale for this convention). This assumes a fixed, uniform
        interfacial tension across all isolated cells and solves for each
        cell's relative pressure from how tightly curved its own full
        contour is: delta_p = tension / R, fit with a single circle to the
        cell's whole boundary (self.mask, if available) rather than the
        piecewise arcs used elsewhere in the main VMSI fit - a single clean
        fit to the true contour is more robust than stitching together
        several short, noisy synthetic-edge arcs around the same shape.

        If self.mask isn't available (e.g. constructed without one), falls
        back to the coarser synthetic-edge/fit_circle-based radius already
        computed elsewhere in the pipeline, using the same delta_p=tension
        convention.

        :param background_pressure: reference pressure assigned to cell 0. Defaults to
                                    self.isolated_background_pressure (set at construction).
        :param tension: assumed uniform interfacial tension for every isolated cell (arbitrary
                        units unless independently measured). Defaults to self.isolated_tension.
        :return: list of cell indices that were processed.
        """
        if background_pressure is None:
            background_pressure = self.isolated_background_pressure
        if tension is None:
            tension = self.isolated_tension

        isolated = [
            c for c in range(1, len(self.cells))
            if (len(self.cells.at[c, 'ncells']) == 1
                and int(self.cells.at[c, 'ncells'][0]) == 0)
        ]

        for cell_idx in isolated:
            radius = np.inf
            circularity = np.nan

            if self.mask is not None:
                cell_mask = (self.mask == cell_idx).astype(float)
                contours = measure.find_contours(cell_mask, 0.5)
                if contours:
                    contour = max(contours, key=len)
                    contour_xy = contour[:, ::-1]
                    _, radius, circularity = _fit_circle_to_contour(contour_xy)

            pressure = background_pressure + (tension / radius if np.isfinite(radius) and radius > 0 else 0.0)
            self.cells.at[cell_idx, 'pressure'] = pressure
            self.cells.at[cell_idx, 'circularity'] = circularity

            cell_edges = self.cells.at[cell_idx, 'edges']
            if hasattr(cell_edges, '__len__') and len(cell_edges) > 0:
                cell_edges = np.array(cell_edges, dtype=int)
                cell_edges = cell_edges[cell_edges >= 0]
                for e in cell_edges:
                    r = self.edges.at[e, 'radius'] if not np.isfinite(radius) else radius
                    self.edges.at[e, 'tension'] = tension if np.isfinite(r) else 0.0

        self.isolated_cells = np.array(isolated, dtype=int)
        return isolated

    def _edge_curvature_sign(self, rho, cell_a, cell_b):
        """
        Determine which side of a fitted arc (centre rho) is the higher-pressure
        side, following the same soap-film convention as everywhere else in VMSI:
        for an interface separating regions of pressure p_a > p_b, the arc bulges
        away from its own centre of curvature into the lower-pressure region, so
        the centre of curvature always lies on the higher-pressure side.

        :return: +1 if cell_a is the higher-pressure side, -1 if cell_b is,
                 or +1 by default if this can't be determined (e.g. no mask
                 available and one side is background) - matching the outward-
                 bulge assumption already used by infer_isolated_cells.
        """
        if not np.all(np.isfinite(rho)):
            # Flat edge (radius=inf) - sign is irrelevant since curvature is 0
            return 1

        if self.mask is not None:
            row, col = int(round(rho[1])), int(round(rho[0]))
            if 0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]:
                label = self.mask[row, col]
                if label == cell_a:
                    return 1
                if label == cell_b:
                    return -1
            # rho fell outside the mask or on neither cell (thin/noisy arc) -
            # fall through to the centroid-distance heuristic below

        if cell_a != 0 and cell_b != 0:
            dist_a = np.linalg.norm(rho - np.array(self.cells.at[cell_a, 'centroids']))
            dist_b = np.linalg.norm(rho - np.array(self.cells.at[cell_b, 'centroids']))
            return 1 if dist_a <= dist_b else -1

        # One side is background with no reliable centroid to compare against -
        # default to assuming the real cell bulges outward (higher pressure),
        # the same convention infer_isolated_cells uses for isolated cells.
        return 1 if cell_b == 0 else -1

    def infer_cluster_cells(self, background_pressure=None, tension=None):
        """
        Generalizes infer_isolated_cells() to small clusters of mutually-touching
        cells that never reach a bulk (fully interior) cell - so they're excluded
        from self.involved_cells (see classify_cells) - but that also aren't
        purely background-facing singletons either, so infer_isolated_cells
        skips them too (they're absent from self.isolated_cells).

        Uses the same fixed, uniform interfacial tension convention as
        infer_isolated_cells, but generalizes the single-circle-per-cell fit to
        a per-edge Young-Laplace system solved jointly across each cluster:
        every edge already has a fitted radius/centre of curvature from
        fit_circle() (run in prepare_data(), before this is called), giving one
        linear equation p_a - p_b = tension / radius_edge per edge, with the
        sign resolved by which side of the arc's centre of curvature falls on
        (see _edge_curvature_sign). Solving this (typically over-determined,
        since most cluster cells border several edges) linear system per
        cluster in a least-squares sense gives every cell in it a pressure
        relative to background_pressure - this exactly reduces to
        infer_isolated_cells's own equation for a cluster of size 1 bordering
        only background, and background_pressure anchors the (rare) cluster
        that happens not to border background directly via a weak Tikhonov
        regularization term, so the system is never rank-deficient.

        :param background_pressure: reference pressure assigned to cell 0. Defaults to
                                    self.isolated_background_pressure (set at construction).
        :param tension: assumed uniform interfacial tension for every edge in a
                        cluster (arbitrary units unless independently measured).
                        Defaults to self.isolated_tension.
        :return: list of cell indices that were processed.
        """
        if background_pressure is None:
            background_pressure = self.isolated_background_pressure
        if tension is None:
            tension = self.isolated_tension

        all_cells = np.arange(1, len(self.cells))
        gap_cells = np.setdiff1d(np.setdiff1d(all_cells, self.involved_cells), self.isolated_cells)
        gap_set = set(gap_cells.tolist())

        # Group gap cells into connected clusters via mutual (non-background) adjacency
        visited = set()
        clusters = []
        for c in gap_cells:
            if c in visited:
                continue
            visited.add(c)
            stack = [c]
            cluster = []
            while stack:
                cur = stack.pop()
                cluster.append(cur)
                for n in np.array(self.cells.at[cur, 'ncells'], dtype=int):
                    if n in gap_set and n not in visited:
                        visited.add(n)
                        stack.append(n)
            clusters.append(np.array(cluster, dtype=int))

        processed = []
        anchor_weight = 1e-3

        for cluster in clusters:
            cluster_index = {int(cell): i for i, cell in enumerate(cluster)}
            k = len(cluster)

            # Gather every edge touching any cell in this cluster exactly once
            cluster_edges = set()
            for cell in cluster:
                cell_edges = self.cells.at[cell, 'edges']
                if hasattr(cell_edges, '__len__'):
                    for e in np.array(cell_edges, dtype=int):
                        if e >= 0:
                            cluster_edges.add(int(e))

            rows = []
            rhs = []
            used_edges = []
            for e in cluster_edges:
                edge_cells = self.edges.at[e, 'cells']
                if len(edge_cells) != 2:
                    continue
                a, b = int(edge_cells[0]), int(edge_cells[1])
                a_in, b_in = a in cluster_index, b in cluster_index
                if not (a_in or b_in):
                    continue
                if not (a_in and b_in) and a != 0 and b != 0:
                    # Borders a cell outside this cluster/background - shouldn't
                    # happen (gap cells only ever touch background or other gap
                    # cells), but skip defensively rather than assume.
                    continue

                radius = self.edges.at[e, 'radius']
                kappa = 1.0/radius if np.isfinite(radius) and radius > 0 else 0.0
                rho = np.array(self.edges.at[e, 'rho'], dtype=float)
                sign = self._edge_curvature_sign(rho, a, b)

                row = np.zeros(k)
                if a_in:
                    row[cluster_index[a]] += 1
                if b_in:
                    row[cluster_index[b]] -= 1
                value = sign * tension * kappa - (background_pressure if a == 0 else 0.0) + (background_pressure if b == 0 else 0.0)

                rows.append(row)
                rhs.append(value)
                used_edges.append((e, radius))

            if k == 0:
                continue

            # Weak per-cell anchor towards background_pressure - keeps the system
            # well-posed (non-rank-deficient) for the rare cluster with no direct
            # background-facing edge, while barely perturbing normal, better-
            # constrained clusters.
            rows.extend(anchor_weight * np.eye(k))
            rhs.extend(anchor_weight * background_pressure * np.ones(k))

            A = np.array(rows)
            b_vec = np.array(rhs)
            # Pressure is a power/weight in VMSI's underlying power-diagram geometry
            # (see get_tensions's |p_a*p_b| term, and the p>=0.001 box constraint the
            # main solver imposes) - it isn't physically meaningful below background,
            # so bound the solve rather than let curvature disagreement between edges
            # push an individual cell negative.
            pressures = lsq_linear(A, b_vec, bounds=(background_pressure, np.inf)).x

            for cell, p in zip(cluster, pressures):
                self.cells.at[cell, 'pressure'] = p

            for e, radius in used_edges:
                self.edges.at[e, 'tension'] = tension if np.isfinite(radius) else 0.0

            processed.extend(cluster.tolist())

        self.cluster_cells = np.array(processed, dtype=int)
        return processed

    def compute_stresstensor(self):
        """

        Compute stress tensor for each cell from tensions and pressures.
        This uses a small-angle approximation assuming edge curvatures are small.

        """

        p = np.array([self.cells.at[cell, 'pressure'] for cell in self.involved_cells])
        T = np.zeros(self.cell_pairs.shape[0])
        edge_verts = np.array(self.edges.verts.to_list())

        i1 = -1*np.ones_like(T)
        for e in range(len(T)):
            verts = self.involved_vertices[self.dV[e,:] != 0]

            if len(verts) == 2:
                ind = np.where(np.all(verts == np.sort(edge_verts, axis=1), axis=1))[0]
            else:
                ind = np.array([])

            # ind can contain more than one match if two edges happen to
            # share the same vertex pair (e.g. duplicate/synthetic edges
            # from isolated-cell topology injection or vertex splitting) -
            # int(ind) would then raise "only length-1 arrays can be
            # converted to Python scalars". Just take the first match.
            if ind.size>0 and self.edges.at[int(ind[0]), 'tension'].size>0:
                T[e] = self.edges.at[int(ind[0]), 'tension']
                i1[e] = int(ind[0])
            else:
                T[e] = 1

        rv = np.array([self.vertices.at[vertex, 'coords'] for vertex in self.involved_vertices])


        rb = np.matmul(self.dV, rv)
        D = np.sqrt(np.sum(np.power(rb, 2), 1))
        D[D==0] = 1
        rb = np.divide(rb.T, D).T
        Rot = np.array([[0,-1],[1,0]])
        nb = np.matmul(rb, Rot.T)
        dP = gather_pairs(p, self.cell_pairs)

        sigmaB = np.zeros([rb.shape[0], 3])
        sigmaB[:,0] = rb[:,0] * T * rb[:,0]
        sigmaB[:,1] = rb[:,0] * T * rb[:,1]
        sigmaB[:,2] = rb[:,1] * T * rb[:,1]

        sigmaP = np.zeros([rb.shape[0], 3])
        sigmaP[:,0] = nb[:,0] * dP * D * nb[:,0]
        sigmaP[:,1] = nb[:,0] * dP * D * nb[:,1]
        sigmaP[:,2] = nb[:,1] * dP * D * nb[:,1]

        n_cells = len(self.involved_cells)
        sigma = symmetric_scatter_pairs(sigmaB, self.cell_pairs, n_cells) + 0.5 * scatter_pairs(sigmaP, self.cell_pairs, n_cells)

        A = np.array(self.cells.area.to_list())[self.involved_cells]
        sigma = np.divide(sigma.T, A)
        for c in range(len(self.involved_cells)):
            self.cells.at[self.involved_cells[c], 'stress'] = sigma[:,c]
        return

    def _resolve_cell_values(self, column):
        """
        Look up a per-cell array of values for `column`, indexed 0..len(self.cells)-1 -
        the same indexing as self.cells/self.mask - so plot() can colour cells by it the
        same way it already does for 'pressure'.

        Checks self.cells first. If not found there, falls back to self.adata.obs (set by
        attach_adata/run_VMSI(adata=...)), mapping each adata row back onto its cell via
        adata.obs['cell_id'] - the inverse of attach_adata's own row-matching, so this
        works for columns computed in adata *after* run_VMSI (e.g. sc.tl.leiden), which
        were never copied into self.cells in the first place.

        :param column: (str) column name to look up.
        :return: (numpy array) values indexed 0..len(self.cells)-1; numeric dtype (with NaN
                 for missing) if the source column is numeric, else object dtype (with None
                 for missing).
        """
        if column in self.cells.columns:
            return self.cells[column].to_numpy()

        if self.adata is not None and column in self.adata.obs.columns:
            if 'cell_id' not in self.adata.obs.columns:
                raise ValueError(
                    f"self.adata.obs has no 'cell_id' column, so '{column}' can't be mapped "
                    f"back onto cells - this should have been set automatically by "
                    f"run_VMSI(adata=...)/attach_adata()."
                )
            col_series = self.adata.obs[column]
            if pd.api.types.is_numeric_dtype(col_series):
                values = np.full(len(self.cells), np.nan)
            else:
                values = np.full(len(self.cells), None, dtype=object)
            cell_ids = self.adata.obs['cell_id'].to_numpy()
            in_range = (cell_ids >= 0) & (cell_ids < len(self.cells))
            values[cell_ids[in_range]] = col_series.to_numpy()[in_range]
            return values

        raise ValueError(
            f"'{column}' is not a column in self.cells, and self.adata is "
            f"{'set but has no matching adata.obs column' if self.adata is not None else 'None'} - "
            f"nothing to plot. Available self.cells columns: {list(self.cells.columns)}."
        )

    def plot(self, options='', mask=np.array([]), line_thickness=5, size=10, file=None):
        """

        Plots results of stress inference.

        :param mask: (numpy array) image on which to overlay plotted objects. If plotting pressure
                     or a generic attribute (see below), must be a labelled, segmented image -
                     pass model.mask.
        :param options: (list) list of options for plotting. The built-in options are: 'stress',
                         'pressure', 'tension', 'cap'. Any other string is treated as a per-cell
                         attribute to colour cells by (like 'pressure', but for any column) -
                         looked up first in self.cells, then in self.adata.obs if an adata is
                         attached (see attach_adata/run_VMSI(adata=...)), e.g.
                         model.plot(['leiden'], model.mask) for a Leiden clustering computed in
                         adata after run_VMSI. Numeric columns get a sequential colourmap and
                         colourbar (same normalisation as 'pressure': 90th percentile clipped);
                         non-numeric columns (e.g. cluster labels) get one discrete colour per
                         category and a legend instead. Only the first non-built-in option found
                         is used for cell colouring; combine it with 'tension'/'stress'/'cap' in
                         the same call to overlay those too (e.g. ['leiden', 'tension']).
        :param line_thickness: (int) thickness of lines used for tension and CAP plotting. Default: 5.
        :param size: (int) text size for legends. Default: 10.
        :param file: (str) filename to save plot to. If none provided, outputs plot to console.
        """
        options = [option.lower() for option in options]

        if mask.size > 0 and self.mask is not None and mask.shape != self.mask.shape:
            raise ValueError(
                f"mask passed to plot() has shape {mask.shape}, but this model was fit "
                f"against a mask of shape {self.mask.shape} (self.mask) - cell centroids "
                f"and coordinates are only meaningful in that coordinate space. This "
                f"mismatch commonly happens after run_VMSI(..., expand_distance>0), which "
                f"internally relabels/resizes the mask before fitting. Pass mask=model.mask "
                f"(this model's own stored mask) instead of your original input array."
            )

        # Options that aren't one of the built-in keywords are treated as an arbitrary
        # per-cell attribute to colour cells by - a column in self.cells, or (via
        # _resolve_cell_values) in self.adata.obs, e.g. a Leiden cluster computed after
        # run_VMSI(adata=...). Only the first one found is used for area colouring (a cell
        # can only be shaded one way at a time); the rest are ignored here, same as any
        # other unrecognised option.
        reserved_options = ('stress', 'pressure', 'tension', 'cap')
        generic_options = [opt for opt in options if opt not in reserved_options]

        # Pressure first as requires remapping cell area colours
        if mask.size > 0:
            if np.isin('pressure', options):
                img = np.zeros_like(mask).astype(float)
                colourmap = cm.get_cmap('plasma')
                # Include isolated cells and small disconnected clusters (pressure inferred
                # directly via infer_isolated_cells/infer_cluster_cells, not part of the main
                # confluent network) so they aren't whited/greyed out below.
                plot_cells = np.union1d(np.union1d(self.involved_cells, self.isolated_cells), self.cluster_cells)
                centroids = np.array(self.cells.loc[plot_cells, 'centroids'].tolist())
                pressures = self.cells.pressure.to_numpy()[plot_cells]
                props = measure.regionprops(mask)
                img_centroids = np.array([np.flip(regionprops.centroid) for regionprops in measure.regionprops(mask)])

                maxP = np.percentile(pressures, 90)
                indices = np.argmin(cdist(centroids, img_centroids), axis=1)
                involvedcells_labels = np.zeros(len(plot_cells))
                for i in range(len(plot_cells)):
                    img_index = indices[i]
                    img_label = props[img_index].label
                    involvedcells_labels[i] = img_label
                    p_norm = np.divide(pressures[i], maxP)
                    if p_norm > 1:
                        p_norm = 1
                    img[mask==img_label] = p_norm
                img = colourmap(img)
                # True background (label 0) stays white. Cells present in the mask but with
                # no computed pressure (neither in the confluent network nor isolated - e.g.
                # small disconnected touching-clusters with no bulk interior, see classify_cells)
                # are shown in neutral grey instead, so they're visibly "not modeled" rather
                # than indistinguishable from background.
                uncomputed = ~np.isin(mask, involvedcells_labels) & (mask != 0)
                img[~np.isin(mask, involvedcells_labels),:] = (1,1,1,1)
                img[uncomputed,:] = (0.75,0.75,0.75,1)
            elif len(generic_options) > 0:
                generic_option = generic_options[0]
                values = self._resolve_cell_values(generic_option)
                is_numeric = pd.api.types.is_numeric_dtype(pd.Series(values))
                valid_cells = np.array([c for c in range(1, len(self.cells)) if not pd.isna(values[c])])
                if valid_cells.size == 0:
                    raise ValueError(f"No cells have a non-missing value for '{generic_option}' - nothing to plot.")

                centroids = np.array(self.cells.loc[valid_cells, 'centroids'].tolist())
                props = measure.regionprops(mask)
                img_centroids = np.array([np.flip(regionprops.centroid) for regionprops in props])
                indices = np.argmin(cdist(centroids, img_centroids), axis=1)
                cell_labels = np.array([props[i].label for i in indices])

                if is_numeric:
                    generic_colourmap = cm.get_cmap('plasma')
                    cell_values = values[valid_cells].astype(float)
                    generic_vmin = np.min(cell_values)
                    generic_vmax = np.percentile(cell_values, 90)
                    span = generic_vmax - generic_vmin
                    img = np.zeros_like(mask).astype(float)
                    for i, lbl in enumerate(cell_labels):
                        norm = np.divide(cell_values[i] - generic_vmin, span) if span > 0 else 0.0
                        img[mask == lbl] = np.clip(norm, 0, 1)
                    img = generic_colourmap(img)
                else:
                    generic_categories = pd.unique(values[valid_cells])
                    cat_colourmap = cm.get_cmap('tab20')
                    generic_cat_colours = {cat: cat_colourmap(i % 20) for i, cat in enumerate(generic_categories)}
                    img = np.zeros(mask.shape + (4,))
                    for i, lbl in enumerate(cell_labels):
                        img[mask == lbl] = generic_cat_colours[values[valid_cells[i]]]

                # Same convention as pressure: true background stays white, cells present
                # in the mask but missing a value for this attribute are shown grey.
                uncomputed = ~np.isin(mask, cell_labels) & (mask != 0)
                img[~np.isin(mask, cell_labels), :] = (1,1,1,1)
                img[uncomputed, :] = (0.75,0.75,0.75,1)
            else:
                colourmap = cm.get_cmap('Set3')
                colours = np.array([colourmap(np.mod(i,12)) for i in range(len(np.unique(mask)))])
                img = color.label2rgb(mask, mask, colors=colours, alpha=0.7, bg_label=0, bg_color=(0, 0, 0))
        else:
            if np.isin('pressure', options):
                return("Error: 'pressure' plotting specified without providing labelled, segmented image.")
            mask = np.ones((self.height, self.width))
            # Need to map zeros to black otherwise background will be purple
            img = color.label2rgb(mask, mask, colors=[(1,1,1)], alpha=1)

        # Set up figure
        fig, ax = plt.subplots(1,1,figsize=np.divide(mask.shape,72), dpi=72)
        ax.imshow(img)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)

        # Now we add colourbar for pressure if it was specified
        if np.isin('pressure', options):
            cax1 = divider.append_axes("right", size="5%", pad=0.5)
            p_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(np.min(pressures), maxP), cmap=colourmap), cax=cax1)
            p_cb.set_label('Pressure (a.u.)', size=size)
            p_cb.ax.tick_params(labelsize=size)

        # Colourbar (numeric) or legend (categorical) for a generic per-cell attribute,
        # mirroring the pressure colourbar above - only rendered when mask.size > 0 took
        # the generic_options branch (not the 'pressure' or plain-Set3 branches).
        if mask.size > 0 and not np.isin('pressure', options) and len(generic_options) > 0:
            generic_option = generic_options[0]
            if is_numeric:
                cax_g = divider.append_axes("right", size="5%", pad=0.5)
                g_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(generic_vmin, generic_vmax), cmap=generic_colourmap), cax=cax_g)
                g_cb.set_label(generic_option, size=size)
                g_cb.ax.tick_params(labelsize=size)
            else:
                legend_handles = [patches.Patch(color=generic_cat_colours[cat], label=str(cat)) for cat in generic_categories]
                ax.legend(handles=legend_handles, title=generic_option, fontsize=size*0.6,
                          title_fontsize=size*0.7, loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Every option that isn't one of these four keywords is already handled above (as
        # the base cell-area colouring) or ignored - unrecognised entries here are simply
        # skipped by the if/elif chain below, so there's no need to gate the whole loop on
        # every option being recognised (that would also skip valid ones, e.g. requesting
        # ['leiden', 'tension'] together used to silently draw neither).
        for option in options:
                if option == 'stress':
                    stress = np.array([self.cells.at[cell, 'area'] * np.array([[self.cells.at[cell, 'stress'][0], self.cells.at[cell, 'stress'][1]],[self.cells.at[cell, 'stress'][1], self.cells.at[cell, 'stress'][2]]]) for cell in range(len(self.cells))])

                    # A single non-finite cell (e.g. a NaN tension from get_tensions
                    # propagated through compute_stresstensor) would otherwise fail
                    # np.linalg.eig for the entire batch. Compute eigenvalues only for
                    # finite rows and leave the rest as NaN so they're skipped below.
                    finite_rows = np.isfinite(stress).all(axis=(1,2))
                    eigvals = np.full((stress.shape[0], 2), np.nan)
                    eigvects = np.full((stress.shape[0], 2, 2), np.nan)
                    if finite_rows.any():
                        eigvals[finite_rows], eigvects[finite_rows] = np.linalg.eig(stress[finite_rows])
                    scalefct = np.sqrt(np.nanmedian(np.multiply(eigvals[:,0], eigvals[:,1])))

                    for i in range(len(self.involved_cells)):
                        cell = self.involved_cells[i]
                        if not finite_rows[i]:
                            continue
                        if np.max(stress[i] > 0):
                            centroid = self.cells.at[cell, 'centroids']
                            eigval = eigvals[i,:]
                            eigvect = eigvects[i,:,:]
                            idx = np.flip(np.argsort(eigval))
                            eigval = eigval[idx]
                            eigvect = eigvect[:,idx]

                            # scale eigenvalues
                            eigval = np.divide(eigval, scalefct)
                            eigval[eigval>3] = 3
                            eigval[eigval<0] = 0
                            eigval = eigval * 0.4 * np.mean(np.sqrt(np.divide(np.array(self.cells.area.to_list())[1:], np.pi)))

                            # calculate angle of rotation
                            theta = np.arctan2(eigvect[1,0], eigvect[0,0])
                            if theta < 0:
                                theta = theta + 2*np.pi
                            theta = np.degrees(theta)
                            stress_ellipse = patches.Ellipse(centroid, eigval[0], eigval[1], angle=theta, fill=False, color='red', lw=3)
                            ax.add_patch(stress_ellipse)
                elif option == 'tension':
                    maxT = np.percentile(self.return_tensions(), 95)
                    minT = np.percentile(self.return_tensions(), 1)
                    colourmap = cm.get_cmap('hot')

                    for e in range(len(self.edges)):
                        if self.edges.at[e, 'tension'] > 0:
                            radius = self.edges.at[e, 'radius']
                            tension_norm = np.divide(self.edges.at[e, 'tension'], maxT-minT)
                            if tension_norm > 1:
                                tension_norm = 1
                            colour = colourmap(tension_norm)

                            if np.isinf(radius):

                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                                   np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                                ax.plot(points[:,0], points[:,1], lw=line_thickness, color=colour)
                            else:
                                v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                                rho = self.edges.at[e, 'rho']

                                theta = np.arctan2(v[:,1]-rho[1], v[:,0]-rho[0])
                                theta[theta<0] = theta[theta<0] + 2*np.pi
                                theta = np.sort(theta)

                                if theta[1] - theta[0] > np.pi:
                                    theta[1] = theta[1] - 2*np.pi

                                theta_range = np.linspace(theta[0], theta[1], len(self.edges.at[e, 'pixels']))
                                points = np.array([rho[0] + radius*np.cos(theta_range),
                                                   rho[1] + radius*np.sin(theta_range)]).T
                                ax.plot(points[:,0], points[:,1], lw=line_thickness, color=colour)
                    cax2 = divider.append_axes("right", size="5%", pad=0.5)
                    t_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=colors.Normalize(minT, maxT), cmap=colourmap), cax=cax2)
                    t_cb.set_label('Tension (a.u.)', size=size)
                    t_cb.ax.tick_params(labelsize=size)
                elif option == 'cap':
                    for e in range(len(self.edges)):
                        radius = self.edges.at[e, 'radius']

                        if np.isinf(radius):

                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            points = np.array([np.linspace(v[0,0], v[1,0], len(self.edges.at[e, 'pixels'])),
                                               np.linspace(v[0,1], v[1,1], len(self.edges.at[e, 'pixels']))]).T
                            ax.plot(points[:,0], points[:,1], lw=line_thickness, color='b')
                        else:
                            v = np.array([self.vertices.at[self.edges.at[e, 'verts'][0], 'coords'], self.vertices.at[self.edges.at[e, 'verts'][1], 'coords']])
                            rho = self.edges.at[e, 'rho']

                            theta = np.arctan2(v[:,1]-rho[1], v[:,0]-rho[0])
                            theta[theta<0] = theta[theta<0] + 2*np.pi
                            theta = np.sort(theta)

                            if theta[1] - theta[0] > np.pi:
                                theta[1] = theta[1] - 2*np.pi

                            theta_range = np.linspace(theta[0], theta[1], len(self.edges.at[e, 'pixels']))
                            points = np.array([rho[0] + radius*np.cos(theta_range),
                                               rho[1] + radius*np.sin(theta_range)]).T
                            ax.plot(points[:,0], points[:,1], lw=line_thickness, color='b')
        if file is not None:
            ax.set_facecolor((1,1,1))
            ax.set_alpha(1.0)
            plt.savefig(file, facecolor=ax.get_facecolor())
        else:
            plt.show()
        return

    def smooth_stress(self, smoothsize=10):
        """

        Compute smoothed stress tensor and stores it in the stress_smooth field
        This reduces variability in stress between individual cells and makes it easier to spot tissue-wide trends

        :param smoothsize: (int) Degree of smoothing; controls width of gaussian kernel over cell neighbours
        """
        self.cells['stress_smooth'] = [np.array([0,0,0]) for _ in range(self.cells.shape[0])]

        rc = np.zeros((len(self.involved_cells), 2))
        stress = np.zeros((len(self.involved_cells), 3))
        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            rc[i,:] = self.cells.at[cell, 'centroids']
            stress[i,:] = self.cells.at[cell, 'stress']

        # scaling factor based on mean cell area
        A = np.array(self.cells.area.to_list()[1:])
        r0 = 0.5 * np.mean(np.sqrt(np.divide(A, np.pi)))

        dist = cdist(rc, rc)
        smk = np.exp(-np.power(dist, 2)/(2*np.power(smoothsize*r0,2)))
        smk = np.divide(smk, np.sum(smk, axis=1))

        stress[:,0] = smk @ stress[:,0]
        stress[:,1] = smk @ stress[:,1]
        stress[:,2] = smk @ stress[:,2]

        for i in range(len(self.involved_cells)):
            cell = self.involved_cells[i]
            self.cells.at[cell, 'stress_smooth'] = stress[i,:]
        return

    def output_results(self, metrics=['centroids','pressure','stress','inertia','perimeter','polygon_perimeter','feret_d','moments_hu','bbox', 'area','label'], neighbours=False):
        """
        Outputs force inference/morphometric quantities and cell adjacency matrix as Pandas Dataframes.

        :param metrics: (list) list of metrics to output. Default: all metrics.
        :param neighbours: (bool) whether to output adjacency matrix. Nonzero values in matrix represent junction tension. Default: False.
        """

        # Assign actual id's to each cell instead of just using labels
        cell_ids = {label:id for label, id in zip(self.involved_cells, ['cell_' + str(number+1) for number in range(len(self.involved_cells))])}

        if metrics is not None:
            out_df = self.cells.loc[self.involved_cells, metrics]
            if 'centroids' in out_df.columns:
                out_df['centroid_x'] = np.array(out_df['centroids'].tolist())[:,0]
                out_df['centroid_y'] = np.array(out_df['centroids'].tolist())[:,1]
                out_df.drop('centroids', axis=1, inplace=True)
            if 'stress' in out_df.columns:
                # We want each value to make sense on its own
                # Therefore, compute eigenvalues, orientation and anisotropy of stress tensor
                stress = np.array([np.array([[out_df.at[cell, 'stress'][0], out_df.at[cell, 'stress'][1]],[out_df.at[cell, 'stress'][1], out_df.at[cell, 'stress'][2]]]) for cell in out_df.index])
                eigvals, eigvects = np.linalg.eig(stress)
                eigvals = np.abs(eigvals)
                idx = np.flip(np.argsort(eigvals, axis=1), axis=1)
                eigvals = eigvals[np.arange(eigvals.shape[0])[:,None], idx]
                eigvects = eigvects[np.arange(eigvects.shape[0])[:,None], idx]
                out_df['stresstensor_eigval1'] = eigvals[:,0]
                out_df['stresstensor_eigval2'] = eigvals[:,1]
                # define orientation as absolute angle between largest eigenvector and x-axis
                out_df['stresstensor_orientation'] = np.arctan2(np.abs(eigvects[:,0,:])[:,1], np.abs(eigvects[:,0,:])[:,0])
                # define anisotropy using definition from Hartkamp et al., J. Chem. Phys. 2012
                out_df['stresstensor_anisotropy'] = np.divide(eigvals[:,0] - eigvals[:,1], (eigvals[:,0] + eigvals[:,1]))
                out_df.drop('stress', axis=1, inplace=True)
            if 'inertia' in out_df.columns:
                # We want each value to make sense on its own
                # Therefore, compute eigenvalues, orientation and anisotropy of inertia tensor
                inertia = np.array([np.array([[out_df.at[cell, 'inertia'][0], out_df.at[cell, 'inertia'][1]],[out_df.at[cell, 'inertia'][1], out_df.at[cell, 'inertia'][2]]]) for cell in out_df.index])
                eigvals, eigvects = np.linalg.eig(inertia)
                eigvals = np.abs(eigvals)
                idx = np.flip(np.argsort(eigvals, axis=1), axis=1)
                eigvals = eigvals[np.arange(eigvals.shape[0])[:,None], idx]
                eigvects = eigvects[np.arange(eigvects.shape[0])[:,None], idx]
                out_df['inertiatensor_eigval1'] = eigvals[:,0]
                out_df['inertiatensor_eigval2'] = eigvals[:,1]
                # define orientation as absolute angle between largest eigenvector and x-axis
                out_df['inertiatensor_orientation'] = np.arctan2(np.abs(eigvects[:,0,:])[:,1], np.abs(eigvects[:,0,:])[:,0])
                out_df['inertiatensor_anisotropy'] = np.divide(eigvals[:,0] - eigvals[:,1], (eigvals[:,0] + eigvals[:,1]))
                out_df.drop('inertia', axis=1, inplace=True)
            if 'moments_hu' in out_df.columns:
                # Hu moments 1 and 3 (2 is omitted as it is equivalent to inertia tensor anisotropy)
                out_df['moments_hu_1'] = np.array(out_df['moments_hu'].tolist())[:,0]
                out_df['moments_hu_3'] = np.array(out_df['moments_hu'].tolist())[:,2]
                out_df.drop('moments_hu', axis=1, inplace=True)
            if 'bbox' in out_df.columns:
                out_df['bbox_x'] = np.array(out_df['bbox'].tolist())[:,0]
                out_df['bbox_y'] = np.array(out_df['bbox'].tolist())[:,1]
                out_df.drop('bbox', axis=1, inplace=True)
            out_df.rename(index=cell_ids, inplace=True)

        if neighbours:
            # Generate adjacency matrix.
            # Magnitude of non-zero values represents the tension (in arbitrary units) along the junction.
            adj_mat = pd.DataFrame(np.zeros([len(self.involved_cells), len(self.involved_cells)]), index=self.involved_cells, columns=self.involved_cells)

            for cell in self.involved_cells:
                adj_cells = self.cells.at[cell, 'ncells']
                for adj_cell in adj_cells[np.isin(adj_cells, self.involved_cells)]:
                    adj_mat.loc[cell, adj_cell] = self.edges[self.edges.cells.apply(tuple) == tuple(np.sort([cell, adj_cell]))]['tension'].values
            adj_mat.rename(index=cell_ids, columns=cell_ids, inplace=True)

        if neighbours and metrics is not None:
            return out_df, adj_mat
        elif metrics is None:
            return adj_mat
        elif not neighbours:
            return out_df
        else:
            return False

def _fit_tile(args):
    """

    Build topology and fit a single VMSI model for one tile. Each tile is
    completely independent of every other one until run_VMSI's later
    pairwise-overlap merge step, so this is the unit of work farmed out to
    worker processes when tile=True - kept as a plain module-level function
    (rather than a closure inside run_VMSI) since that's what
    ProcessPoolExecutor needs to be able to pickle and ship to a worker.

    :param args: (tile, tile_holes_mask, is_labelled, verbose, optimiser) tuple.
    :return: fitted VMSI object for this tile.

    """
    tile, tile_holes_mask, is_labelled, verbose, optimiser = args
    seg = Segmenter(masks=tile, labelled=is_labelled)
    VMSI_obj, labelled_mask = seg.process_segmented_image(holes_mask=tile_holes_mask)
    model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df,
                 height=tile.shape[0], width=tile.shape[1], verbose=verbose, optimiser=optimiser)
    model.fit()
    return model

def _remove_small_regions(img, min_size, fill_distance):
    """
    Remove connected components smaller than min_size px from a labelled image (the same
    connectivity=1 relabel + drop-small-regions + expand_labels cleanup the README documents
    for manual use), letting a real neighbour grow back over the gap where one is within
    fill_distance.

    A region relabelled with connectivity=1 catches a disconnected noise speck even if it
    happens to share its raw label ID with an unrelated real cell elsewhere in the image -
    naively checking areas on the input image directly would miss that.

    A region too far from any real neighbour for expand_labels to reach keeps whatever value
    it gets filled with but stays physically disconnected from that neighbour's component, so
    it re-splits into its own (still-too-small) component under the final relabel rather than
    truly merging. Rather than leave that behind under a new label, verify afterwards and
    delete any leftover fragment directly instead of retrying the fill.
    """
    labels = measure.label(img, connectivity=1)
    areas = pd.DataFrame(measure.regionprops_table(label_image=labels, properties=('label', 'area')))
    small = areas.loc[areas['area'] < min_size, 'label'].to_numpy()
    if small.size == 0:
        return img, 0

    to_remove = np.isin(labels, small)
    original = img
    img = img.copy()
    img[to_remove] = 0
    img = expand_labels(img, distance=fill_distance)
    # Only keep the fill at the positions we actually removed - everything else (including
    # any real cell pixels expand_labels might have grown across) reverts to its exact
    # original value.
    img = np.where(to_remove, img, original)
    img = measure.label(img)

    # Verify the fill actually merged each removed region into a real neighbour's component -
    # img is already correctly labelled at this point (by the call directly above), so just
    # measure areas on it directly rather than relabelling with connectivity=1 again: that
    # stricter connectivity would flag (and then wrongly delete) a fragment that only merged
    # diagonally, even though it's now legitimately part of a real cell under the same default
    # connectivity used for every other measure.label(img) call in this pipeline.
    areas2 = pd.DataFrame(measure.regionprops_table(label_image=img, properties=('label', 'area')))
    still_small = areas2.loc[areas2['area'] < min_size, 'label'].to_numpy()
    if still_small.size > 0:
        img[np.isin(img, still_small)] = 0
        img = measure.label(img)

    return img, small.size

def detect_holes(img, hole_size_threshold=2):
    """
    Detect internal tissue holes in an already boundary-separated, labelled mask -
    the "Process mask and detect holes in tissue" step from
    notebooks/00_run_tensionmap.ipynb, factored out so run_VMSI can apply it
    automatically on every call instead of requiring it as a manual step.

    Holes are identified exactly as the notebook does: any connected region in
    img (real cells and background alike) whose area exceeds
    hole_size_threshold times the mean area across all such regions is flagged -
    a normal cell-cell junction is never that large, so an oversized background
    region enclosed by tissue is a genuine interior hole, not exterior
    background. This will also flag the tissue's true exterior background if
    there is one and it's larger than that threshold (typically true) - this
    mirrors the notebook's own algorithm exactly, and just means cells touching
    that background get correctly treated as boundary-adjacent downstream (the
    same thing clear_border already does for cells touching the image frame
    edge specifically).

    Doesn't carve cell boundaries itself - img must already be in that format
    (1px, 4-connected boundaries, unique label per cell). run_VMSI calls this
    after any boundary-carving it does itself (expand_distance), so it's never
    called on a raw touching-label mask.

    :param img: (numpy array) labelled, boundary-separated segmentation mask.
    :param hole_size_threshold: (float) a region is flagged as a hole if its area exceeds
                                 this many times the mean area across all regions in img.
                                 Lower this if genuine holes are being missed (e.g. small
                                 lumens not much bigger than a typical cell); raise it if
                                 normal large cells are being wrongly flagged. Default: 2
                                 (matches notebooks/00_run_tensionmap.ipynb).
    :return: (numpy array) binary mask, same shape as img - 1 where an internal
             hole was detected, 0 elsewhere.
    """
    holes_mask = np.zeros_like(img)
    areas = pd.DataFrame(measure.regionprops_table(label_image=img, properties=('label', 'area')))
    if len(areas) > 0:
        thresh = areas['area'].mean() * hole_size_threshold
        hole_labels = areas.loc[areas['area'] > thresh, 'label'].to_numpy()
        holes_mask[np.isin(img, hole_labels)] = 1

    return holes_mask

def attach_adata(model, adata):
    """
    Attach an AnnData object to a fitted VMSI model, copying every column
    currently in model.cells into adata.obs - factored out of run_VMSI's
    adata handling so it can be redone standalone (e.g. after editing
    model.cells, or after calling model.output_results()-style post-processing).

    Rows are matched via adata.obs['cell_id'] (set by run_VMSI as
    adata.obs.reset_index().index + 1, i.e. 1-indexed row order) against
    model.cells' row index - which is the same value as each cell's label in
    model.mask, so this only gives meaningful results if cell_id values still
    correspond to mask labels (true right after run_VMSI(..., bbox=...), since
    cropping doesn't renumber labels; not true if cells were relabelled
    in between, e.g. by min_cell_size/hole detection/expand_distance removing
    or renumbering cells - run_VMSI already accounts for this by assigning
    cell_id before any of those steps run). Any cell_id with no matching row
    in model.cells (out of range, or a cell that got dropped along the way)
    gets NaN for every model.cells column rather than raising.

    :param model: (VMSI) a fitted VMSI model.
    :param adata: (AnnData) must already have a 'cell_id' column in adata.obs.
    :return: adata, with model.cells' columns added to adata.obs. model.adata is also
             set to adata as a side effect, so the model can be reached from either side.
    """
    model.adata = adata
    cell_ids = adata.obs['cell_id'].to_numpy()
    matched = model.cells.reindex(cell_ids)
    matched.index = adata.obs.index
    for col in matched.columns:
        adata.obs[col] = matched[col]
    return adata

def run_VMSI(img, is_labelled=False, tile=False, cells_per_tile=150, verbose=False, overlap=0.3,
             optimiser='nlopt', isolated_tension=1.0, isolated_background_pressure=0.0, expand_distance=0,
             min_cell_size=0, hole_size_threshold=2, bbox=None, adata=None, n_jobs=None):
    """
    Main function to run stress inference in a single step.

    Cells with no real neighbours (only touching the background) are handled
    via infer_isolated_cells - their pressure is inferred directly from their
    own boundary curvature (Young-Laplace, assuming uniform interfacial
    tension across cells - see infer_isolated_cells/run_isolated_cells for
    the full rationale), rather than through the main vertex-network fit,
    since a genuinely isolated cell has no shared junctions to jointly solve
    tension/pressure against. If *no* cell in the image has any real
    neighbour at all (a fully non-confluent/dispersed segmentation), there is
    no vertex-network topology whatsoever to build - this function detects
    that case and falls back entirely to run_isolated_cells(), which returns
    a plain DataFrame instead of a VMSI object (see below).

    Internal tissue holes are detected automatically (detect_holes(), the
    "Process mask and detect holes in tissue" step from
    notebooks/00_run_tensionmap.ipynb) on every call unless hole_size_threshold
    is set to None - no separate flag or manual holes_mask needed. This runs
    after any boundary-carving this function does itself (expand_distance), so
    it always sees the mask in its final, correctly-resolutioned form. If you
    need to inspect or override the detected holes, call detect_holes(img)
    yourself.

    :param verbose: (bool) whether to provide detailed output. Default:False.
    :param img: (numpy array) segmented image.
    :param is_labelled: (bool) whether all cells in image are labelled. Default: False.
    :param tile: (bool) whether to break image into tiles for faster inference. Not recommended for images with tissue-scale anisotropy. Default: False.
    :param overlap: (float) fraction of overlap between tiles. Default: 0.3.
    :param optimiser: (str) which optimiser to use. Currently available options are 'nlopt' (default) , 'matlab'.
    :param isolated_tension: (float) assumed uniform interfacial tension used to infer pressure for any
                             isolated cells (arbitrary units unless independently measured). Default: 1.0.
    :param isolated_background_pressure: (float) reference pressure assigned to the background for isolated
                                          cells' Young-Laplace inference. Default: 0.0.
    :param expand_distance: (int) if > 0, close thin background gaps between cells that should actually be
                             touching (e.g. a resolution-limited segmentation artifact) before running
                             inference, via skimage.segmentation.expand_labels(distance=expand_distance)
                             followed by re-carving an explicit boundary with
                             find_boundaries(mode='subpixel') - the same two-step approach documented in
                             the README for delineating touching cell labels, just applied automatically.
                             Keep this small (just above your actual gap width) - too large a distance will
                             merge cells that are genuinely meant to stay separate. Note find_boundaries in
                             'subpixel' mode doubles the image's resolution as a side effect; hole detection
                             (see above) runs after this step so it's unaffected either way.
                             Default: 0 (disabled).
    :param min_cell_size: (int) if > 0, remove segmentation regions smaller than this many pixels
                           before inference - the same connectivity=1 relabel + drop-small-regions +
                           expand_labels(distance=5) cleanup documented in the README for manual use,
                           just applied automatically. Tiny stray regions (a few px, e.g. segmentation
                           noise or sub-pixel boundary artifacts) otherwise still enter the vertex-network
                           fit and can get wildly ill-constrained pressures/tensions from their near-
                           degenerate geometry, skewing results for the whole tissue. There's no
                           universally correct threshold - it depends on your image's resolution and
                           actual cell size (a real small cell shouldn't be caught by this), so this is
                           opt-in rather than defaulting to some fixed size. Default: 0 (disabled).
    :param hole_size_threshold: (float or None) passed straight to detect_holes() - a region is
                                 flagged as an internal hole if its area exceeds this many times the
                                 mean area across all regions in the processed mask. Lower this if
                                 genuine holes (e.g. small lumens) are being missed; raise it if normal
                                 large cells are being wrongly flagged as holes. Set to None to skip
                                 hole detection entirely (holes_mask=None, the pre-detect_holes
                                 behaviour). Default: 2 (matches notebooks/00_run_tensionmap.ipynb).
    :param n_jobs: (int) only used when tile=True. Number of tiles to fit in parallel, each in its own
                   process (tiles are independent until the pairwise-overlap merge step afterwards, so
                   this parallelises cleanly). None uses all available CPU cores (os.cpu_count()); 1 fits
                   tiles sequentially in the current process, same as before this option existed. Default: None.
    :param bbox: (list/tuple of 4 ints, or None) if given, [row_start, row_end, col_start, col_end] -
                 img is cropped to img[row_start:row_end, col_start:col_end] before anything else runs
                 (labelling, min_cell_size, hole detection, expand_distance, tiling, etc.), so the rest
                 of the pipeline only ever sees the cropped region. Useful for testing/debugging on a
                 small region of a much larger mask without pre-cropping it yourself. Default: None
                 (use the whole image).
    :param adata: (AnnData, optional) if given, adata.obs['cell_id'] is set to
                  adata.obs.reset_index().index + 1 (1-indexed row order, matching mask label
                  values) before anything else runs. If bbox is also given, adata is then
                  subset to only the cell_ids still present in the cropped mask. A copy is
                  made - your original adata object is never modified in place. Once the model
                  is fit, adata is stored as model.adata, and every column currently in
                  model.cells is copied into adata.obs, row-matched by cell_id against
                  model.cells' row index (see attach_adata() for the standalone version of this
                  step, e.g. to redo it after further editing model.cells). This only happens
                  for the normal VMSI-object return path - if the mask has too little confluent
                  tissue and run_VMSI falls back to run_isolated_cells() (a plain DataFrame, not
                  a VMSI object), adata is left untouched. Default: None (disabled).

    :return: VMSI object containing the inferred tensions, pressures and stress, as well as cell morphology
             metrics - UNLESS the image is fully non-confluent (no cell has any real neighbour), in which case
             a pandas DataFrame is returned instead (see run_isolated_cells), since there is no vertex/edge
             network to attach to a VMSI object at all.
    """

    warnings.filterwarnings('ignore')

    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"bbox must be [row_start, row_end, col_start, col_end] (4 values), got {bbox}")
        row_start, row_end, col_start, col_end = bbox
        img = img[row_start:row_end, col_start:col_end]

    if adata is not None:
        # Copy rather than mutate the caller's object in place - cell_id below is
        # added either way, but the bbox-driven subsetting further down should
        # never surprise a caller who's still holding onto their original adata.
        adata = adata.copy()
        adata.obs['cell_id'] = adata.obs.reset_index().index + 1
        if bbox is not None:
            # img is already cropped above - cell_id values are mask label values,
            # so keep only the cells whose label actually survived the crop.
            present_ids = np.unique(img)
            adata = adata[adata.obs['cell_id'].isin(present_ids)].copy()

    # If cells are not labelled, label them
    if not is_labelled:
        img = measure.label(img)
        is_labelled = True

    if min_cell_size > 0:
        img, n_removed = _remove_small_regions(img, min_cell_size, fill_distance=5)
        if verbose and n_removed > 0:
            print(f"Removed {n_removed} regions under {min_cell_size}px...")

    if expand_distance > 0:
        if verbose:
            print(f"Closing background gaps up to {expand_distance}px between touching cells...")
        img = expand_labels(img, distance=expand_distance)
        boundaries = find_boundaries(img, mode='subpixel')
        img = measure.label(1 - boundaries)
        is_labelled = True

        # find_boundaries(mode='subpixel') doubles the image's resolution by inserting an
        # interpolated pixel between every pair of neighbours; at diagonal/corner junctions
        # where several cells meet, this can strand a single interpolated pixel that belongs
        # to none of them, which measure.label() then turns into its own 1-3px "cell". These
        # are pure artifacts of this resampling step (not present in the input mask, so the
        # user's own pre-filtering can't catch them) - clean them up the same way the README
        # recommends for real segmentation noise: drop tiny regions, then let their neighbours
        # expand back over the gap.
        sliver_min_size = 10
        img, n_slivers = _remove_small_regions(img, sliver_min_size, fill_distance=expand_distance + 1)
        if verbose and n_slivers > 0:
            print(f"Removed {n_slivers} sub-pixel boundary artifacts (<{sliver_min_size}px)...")

    # Detect internal tissue holes (notebooks/00_run_tensionmap.ipynb's hole-detection
    # step), now that img is in its final boundary-separated form - whether that's as
    # originally supplied, or via expand_distance just above. hole_size_threshold=None
    # opts out entirely (holes_mask=None, same as pre-detect_holes behaviour) rather
    # than running detect_holes just to get an all-zero mask back.
    if hole_size_threshold is None:
        holes_mask = None
    else:
        if verbose:
            print("Detecting internal holes...")
        holes_mask = detect_holes(img, hole_size_threshold=hole_size_threshold)

    # Test whether there are enough cells to tile image
    if not len(np.unique(img))-1 > 2*cells_per_tile and tile:
        print('Not enough cells for tiling; proceeding with a single tile.')
        tile = False
    if tile:

        # Generating tiling
        tiles, holes_masks, offset, adj_tiles = create_image_tiles(img, holes_mask, cells_per_tile=cells_per_tile, overlap=overlap)
        models = []

        pairwise_tensions = []
        pairwise_pressures = []

        # Each tile is independent of every other one until the pairwise-
        # overlap merge step below, so fit them in parallel worker processes
        # rather than one after another in this process. n_jobs=1 keeps the
        # old sequential behaviour (e.g. if a single tile's fit already
        # saturates a core, or for easier debugging/profiling).
        tile_args = [(tiles[i], holes_masks[i], is_labelled, verbose, optimiser) for i in range(len(tiles))]
        if n_jobs == 1:
            models = [_fit_tile(a) for a in tile_args]
        else:
            if verbose:
                print(f"Fitting {len(tiles)} tiles in parallel (n_jobs={n_jobs or 'all cores'})...")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # map() preserves input order in its output, so models[i]
                # still lines up with tiles[i]/offset[i] as adj_tiles assumes.
                models = list(executor.map(_fit_tile, tile_args))
        # For each pair of adjacent tiles, determine cell overlap and record overlapping tensions and pressures
        for i in range(len(adj_tiles)):
            pair = adj_tiles[i]
            model_1 = models[pair[0]]
            model_2 = models[pair[1]]

            # Match pressures
            model1_cells = np.array(model_1.cells.centroids.tolist())[1:] + offset[pair[0]]
            model2_cells = np.array(model_2.cells.centroids.tolist())[1:] + offset[pair[1]]

            cell_pwdist = cdist(model1_cells, model2_cells)
            # We expect matching cells to have assigned centroids that are <=2px apart in the two tiles
            # This should be the same regardless of image size
            matching_cells = np.where(cell_pwdist<=2)
            pressures = np.array([model_1.return_pressures()[matching_cells[0]], model_2.return_pressures()[matching_cells[1]]])
            pairwise_pressures.append(pressures[:,np.all(pressures>0, axis=0)])

            # Match edges
            model1_edge_r1 = np.array(model_1.vertices.coords[np.array(model_1.edges.verts.tolist())[:,0]].tolist()) + offset[pair[0]]
            model1_edge_r2 = np.array(model_1.vertices.coords[np.array(model_1.edges.verts.tolist())[:,1]].tolist()) + offset[pair[0]]
            model2_edge_r1 = np.array(model_2.vertices.coords[np.array(model_2.edges.verts.tolist())[:,0]].tolist()) + offset[pair[1]]
            model2_edge_r2 = np.array(model_2.vertices.coords[np.array(model_2.edges.verts.tolist())[:,1]].tolist()) + offset[pair[1]]

            r1_pwdist = cdist(model1_edge_r1, model2_edge_r1)
            r2_pwdist = cdist(model1_edge_r2, model2_edge_r2)

            matching_edges = np.where(np.logical_and(r1_pwdist <= 2, r2_pwdist <= 2))
            tensions = np.array([model_1.return_tensions()[matching_edges[0]], model_2.return_tensions()[matching_edges[1]]])
            pairwise_tensions.append(tensions[:,np.all(tensions>0, axis=0)])

        # Scale tensions and pressures globally
        def p_energy(x):
            E = np.sum([np.sum(np.power(pairwise_pressures[i][0,:] + x[adj_tiles[i][0]] - pairwise_pressures[i][1,:] - x[adj_tiles[i][1]], 2)) for i in range(len(adj_tiles))])
            return E
        constr = LinearConstraint(np.ones(len(tiles))/len(tiles), 0, 0)
        res = minimize(p_energy, np.zeros(len(tiles)), tol=1e-8, constraints=[constr])
        p_scale = res.x

        def t_energy(x):
            E = np.sum([np.sum(np.power(pairwise_tensions[i][0,:]*x[adj_tiles[i][0]] - pairwise_tensions[i][1,:]*x[adj_tiles[i][1]], 2)) for i in range(len(adj_tiles))])
            return E
        constr = LinearConstraint(np.ones(len(tiles))/len(tiles), 1, 1)
        res = minimize(t_energy, np.ones(len(tiles)), tol=1e-8, constraints=[constr])
        t_scale = res.x

        model = merge_models(models, p_scale, t_scale, offset, img, verbose=verbose, holes_mask=holes_mask)
    else:
        # There turn out to be several distinct places (Segmenter.find_vertices,
        # Segmenter.find_edges, VMSI.classify_cells, VMSI.build_diff_operators,
        # and plausibly others not yet identified) where a mask with too
        # little confluent interior tissue - scattered isolated cells, small
        # touching clusters with no genuinely deep interior, etc. - makes some
        # intermediate array come up empty and raise a ValueError. Rather than
        # maintaining a growing whitelist of exact error strings from each
        # site individually (which needed a new entry nearly every time a
        # differently-shaped sparse mask was tried), treat *any* ValueError
        # raised while building/fitting the main confluent-tissue model as
        # evidence this mask doesn't have enough usable topology, and fall
        # back to the standalone per-cell Laplace-pressure method. The
        # underlying error is always printed (not just when verbose=True) so
        # a genuinely unrelated bug in this code path doesn't get silently
        # misattributed to "not enough topology" without you seeing why.
        try:
            # process segmented image for input into VMSI
            seg = Segmenter(masks=img, labelled=is_labelled)
            VMSI_obj, labelled_mask = seg.process_segmented_image(holes_mask=holes_mask)
            # create the model
            model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=img.shape[0], width=img.shape[1],
                          verbose=verbose, optimiser=optimiser, mask=labelled_mask,
                          isolated_tension=isolated_tension, isolated_background_pressure=isolated_background_pressure)
            # fit the model parameters
            model.fit()
            # compute stress tensor
            model.compute_stresstensor()
        except ValueError as err:
            print(f"run_VMSI: the main confluent-tissue pipeline failed ({err}); "
                  "treating this as insufficient interior topology and falling back to "
                  "run_isolated_cells (per-cell Young-Laplace pressure inference). If this "
                  "mask should have had enough confluent tissue to avoid this, the error "
                  "above is worth a closer look rather than trusting this fallback.")
            # run_isolated_cells returns a plain DataFrame, not a VMSI object, so it has
            # nowhere to attach adata to (see attach_adata's docstring) - fall through to
            # the common return below rather than attaching there.
            return run_isolated_cells(img, is_labelled=True, background_pressure=isolated_background_pressure,
                                       tension=isolated_tension, verbose=verbose)

    if adata is not None:
        adata = attach_adata(model, adata)
    return model

def _fit_circle_to_contour(contour_xy):
    """

    Algebraic (Kasa) least-squares circle fit to a closed set of 2D points.
    Unlike VMSI.fit_circle (which fits an arc between two known vertex
    endpoints), this fits a circle to an entire closed contour with no
    endpoints, which is what a spatially-isolated cell's own boundary is.

    :param contour_xy: (N,2) array of (x,y) boundary coordinates.
    :return: (centre (2,), radius (float), circularity (float)) - circularity
             is the coefficient of variation of point-to-centre distances
             (0 = perfect circle, larger = less circular / less reliable fit).

    """
    x = contour_xy[:, 0]
    y = contour_xy[:, 1]

    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r_sq = c + cx**2 + cy**2

    if r_sq <= 0:
        return np.array([cx, cy]), np.inf, np.inf

    radius = np.sqrt(r_sq)
    dists = np.sqrt((x - cx)**2 + (y - cy)**2)
    circularity = np.std(dists) / np.mean(dists) if np.mean(dists) > 0 else np.inf

    return np.array([cx, cy]), radius, circularity

def run_isolated_cells(img, is_labelled=False, background_pressure=0.0, tension=1.0, verbose=False):
    """

    Infer relative cell pressure for a segmentation with no shared cell-cell
    topology at all (e.g. spatially separated / dispersed cells), by applying
    Young-Laplace's law directly to each cell's own boundary shape.

    This is a fundamentally different, much weaker measurement than run_VMSI:
    a genuinely isolated cell's boundary curvature only constrains the ratio
    of its pressure difference (from the surrounding medium) to its own
    interfacial tension (delta_p = tension * curvature) - not each
    independently. There is no cell-cell junction network to jointly solve
    for both, unlike in a confluent tissue. This assumes interfacial tension
    is the same for every cell (a modelling choice, not something measured -
    see the fit_circle-based derivation in VMSI.infer_isolated_cells for the
    background), and infers each cell's *relative* pressure from how tightly
    curved (circular) its own boundary is. It intentionally bypasses all of
    VMSI's vertex/edge topology construction and nonlinear optimisation,
    since none of that machinery applies when cells share no boundaries.

    :param img: (numpy array) segmented image.
    :param is_labelled: (bool) whether cells are already labelled. Default: False.
    :param background_pressure: (float) reference pressure assigned to the surrounding medium. Default: 0.0.
    :param tension: (float) assumed uniform interfacial tension for every cell, in arbitrary units unless
                    independently measured - only relative pressures between cells are meaningful. Default: 1.0.
    :param verbose: (bool) print progress. Default: False.

    :return: pandas DataFrame with one row per cell: label, centroid, area, radius, circularity, pressure.
             circularity is the coefficient of variation of the fitted circle's residuals - high values mean
             the cell's shape is a poor match for a single circular arc, so its inferred pressure should be
             treated with more caution.

    """
    if not is_labelled:
        img = measure.label(img)

    labels = np.unique(img)
    labels = labels[labels != 0]

    rows = []
    for label in labels:
        cell_mask = (img == label).astype(float)
        contours = measure.find_contours(cell_mask, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        contour_xy = contour[:, ::-1]

        centre, radius, circularity = _fit_circle_to_contour(contour_xy)
        pressure = background_pressure + (tension / radius if np.isfinite(radius) and radius > 0 else np.nan)

        props = measure.regionprops((img == label).astype(int))[0]

        rows.append({
            'label': label,
            'centroid_x': props.centroid[1],
            'centroid_y': props.centroid[0],
            'area': props.area,
            'radius': radius,
            'circularity': circularity,
            'pressure': pressure,
        })

        if verbose:
            print(f"cell {label}: radius={radius:.3g}, circularity={circularity:.3g}, pressure={pressure:.3g}")

    return pd.DataFrame(rows)

def create_image_tiles(img, holes_mask, cells_per_tile=150, overlap=0.3):
    """

    Split image into overlapping tiles, each of which contain approximately the same number of cells.

    :param img: (numpy array) labelled, segmented image to be split into tiles.
    :param holes_mask: (numpy array) binary image containing interior holes in segmented image.
    :param cells_per_tile: (int) number of cells per tile. Default: 150.
    :param overlap: (float) fraction of overlap between each adjacent tiles. Default: 0.3.

    :return tiles: (list) list of tiles of original image, where each tile is a numpy array.
    :return holes_masks: (list) list of tiles of holes masks, where each tile is a numpy array.
    :return offset: (list) offset between each tile's top left corner relative to the untiled image.
    :return adj_tiles: (list) pairs of adjacent tiles with sufficient overlap to allow scaling after inference.
    """
    cell_properties = pd.DataFrame(measure.regionprops_table(img, properties=('label','bbox','centroid')))
    ncells = cell_properties.shape[0]

    # Get tile centres using K-means
    ntiles = int(2*np.round(ncells/(2*cells_per_tile)))
    cell_centroids = cell_properties[['centroid-0', 'centroid-1']].values
    kmeans = KMeans(n_clusters=ntiles, random_state=0)
    kmeans_res = kmeans.fit_predict(cell_centroids)

    # Compute bounding box for each tile
    tiles = []
    holes_masks = []
    offset = []
    tiles_bbox = np.zeros([ntiles, 4], dtype=int)
    for i in range(ntiles):
        tile_index = np.unique(kmeans_res)[i]
        cell_bbox = cell_properties[['bbox-0','bbox-1','bbox-2','bbox-3']].iloc[np.where(kmeans_res == tile_index)[0]].values
        tile_bbox = np.array([np.min(cell_bbox[:,0]), np.min(cell_bbox[:,1]), np.max(cell_bbox[:,2]), np.max(cell_bbox[:,3])])
        # Expand bounding box by overlap percentage (in each direction)
        tile_bbox += np.round((overlap/2)*np.array([-(tile_bbox[3]-tile_bbox[1]), -(tile_bbox[2]-tile_bbox[0]), (tile_bbox[3]-tile_bbox[1]), (tile_bbox[2]-tile_bbox[0])])).astype(int)
        tile_bbox = np.clip(tile_bbox, [0, 0, 0, 0], [img.shape[0], img.shape[1], img.shape[0], img.shape[1]])
        tiles_bbox[i,:] = tile_bbox

        tiles.append(img[tile_bbox[0]:tile_bbox[2], tile_bbox[1]:tile_bbox[3]])
        if holes_mask is not None:
            holes_masks.append(holes_mask[tile_bbox[0]:tile_bbox[2], tile_bbox[1]:tile_bbox[3]])
        else:
            holes_masks.append(None)
        offset.append(np.array([tile_bbox[1], tile_bbox[0]]))

    # Determine adjacent tiles using overlap; threshold by half overlap fraction multiplied by smallest tile area
    adj_tiles = []
    for tile_1 in range(ntiles):
        for tile_2 in range(tile_1+1, ntiles):
            bbox_1 = tiles_bbox[tile_1,:]
            bbox_2 = tiles_bbox[tile_2,:]
            if not (bbox_1[2]<bbox_2[0] or bbox_1[0]>bbox_2[2] or bbox_1[1]>bbox_2[3] or bbox_1[3]<bbox_2[1]):
                # If there is overlap, determine overlap area
                x_left = np.maximum(bbox_1[0], bbox_2[0])
                y_top = np.maximum(bbox_1[1], bbox_2[1])
                x_right = np.minimum(bbox_1[2], bbox_2[2])
                y_bottom = np.minimum(bbox_1[3], bbox_2[3])

                # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                # Only use overlap that contains actual cells, not holes
                intersection_area -= np.sum(holes_mask[x_left:x_right, y_top:y_bottom])
                area_1 = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
                area_2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])

                if intersection_area > (overlap/3)*np.minimum(area_1, area_2):
                    adj_tiles.append([tile_1, tile_2])


    # Output tiling for debug
    fig, ax = plt.subplots(1,1,figsize=np.divide(img.shape,72), dpi=72)
    ax.imshow(img)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for tile_index in range(tiles_bbox.shape[0]):
        bbox = tiles_bbox[tile_index,:]
        plt.hlines(bbox[0], bbox[1], bbox[3], linewidth=10, colors='black')
        plt.hlines(bbox[2], bbox[1], bbox[3], linewidth=10, colors='black')
        plt.vlines(bbox[1], bbox[0], bbox[2], linewidth=10, colors='black')
        plt.vlines(bbox[3], bbox[0], bbox[2], linewidth=10, colors='black')
    plt.tight_layout()
    plt.show()
    return tiles, holes_masks, offset, adj_tiles


def merge_models(models, p_scale, t_scale, offset, img, verbose, holes_mask=None):
    """

    Merges VMSI objects created for each tile into a single object based on scaling factors calculated for tensions and pressures

    :param models: (list) list of VMSI objects for each tile
    :param p_scale: (numpy array) additive scaling factors for cell pressures for each tile.
    :param t_scale: (numpy array) multiplicative scaling factors for junction tensions for each tile.
    :param offset: (list) offset between each tile's top left corner relative to the untiled image.
    :param img: (numpy array) labelled image.
    :param verbose: (bool) output verbosity.
    :param holes_mask: (numpy array) binary image containing interior holes in segmented image.

    :return: single VMSI object containing inferred mechanics for the entire image.
    """
    # process segmented image for input into VMSI
    seg = Segmenter(masks=img, labelled=True)
    VMSI_obj, labelled_mask = seg.process_segmented_image(holes_mask=holes_mask)
    # create the model
    merged_model = VMSI(vertices=VMSI_obj.V_df, cells=VMSI_obj.C_df, edges=VMSI_obj.E_df, height=img.shape[0], width=img.shape[1], verbose=verbose)
    merged_model.prepare_data()

    # counters for number of inferred values for each edge and cell
    ncell = np.zeros(len(merged_model.cells))
    nedge = np.zeros(len(merged_model.edges))
    for i in range(len(models)):
        model = models[i]
        model.cells.pressure = model.cells.pressure.apply(lambda x: x + p_scale[i])
        model.edges.tension = model.edges.tension.apply(lambda x: x * t_scale[i])
        model.compute_stresstensor()

        # Match cells and edges for each model to the merged model
        def match_index(coords_1, coords_2):
            pdist = cdist(coords_1, coords_2)
            indices = np.argmin(pdist, axis=1)
            return indices

        cell_indices = match_index(np.array(model.cells.loc[model.involved_cells].centroids.tolist()) + offset[i], np.array(merged_model.cells.centroids.tolist()))
        merged_model.involved_cells = (np.unique(np.concatenate((merged_model.involved_cells, cell_indices)))) if merged_model.involved_cells is not None else (cell_indices)
        merged_model.cells.loc[cell_indices, ['pressure', 'stress']] = merged_model.cells.loc[cell_indices, ['pressure', 'stress']].values + model.cells.loc[model.involved_cells, ['pressure', 'stress']].values
        ncell[cell_indices] += 1

        for edge in model.involved_edges:
            index = int(np.where(np.all(np.sort(cell_indices[np.ravel(np.argwhere(np.isin(model.involved_cells, model.edges.at[edge, 'cells'])))]) == np.sort(np.array(merged_model.edges.cells.tolist())), axis=1))[0])
            merged_model.edges.at[index, 'tension'] = merged_model.edges.at[index, 'tension'] + model.edges.at[edge, 'tension']
            nedge[index] += 1

    # Take mean over edges and cells with multiple inferred values
    merged_model.cells.loc[ncell>0, 'pressure'] = np.divide(merged_model.cells.loc[ncell>0, 'pressure'].values,ncell[ncell>0])
    merged_model.cells.loc[ncell>0, 'stress'] = np.divide(merged_model.cells.loc[ncell>0, 'stress'].values,ncell[ncell>0])
    merged_model.edges.loc[nedge>0, 'tension'] = np.divide(merged_model.edges.loc[nedge>0, 'tension'].values,nedge[nedge>0])

    return merged_model