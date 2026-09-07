import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
from scipy.ndimage import generic_filter
from scipy.optimize import minimize, leastsq
from scipy.spatial import ConvexHull
import skimage.segmentation as seg
import skimage.morphology as morph
import skimage.measure as measure
import skimage.draw as draw
from src.bwmorph import *
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix

def set_array_at(df, idx, col, arr):
    """

    Assign arr to df.at[idx, col] without pandas unboxing a length-1
    np.ndarray into a 0-d array (df.at[idx, col] = arr does this silently),
    which otherwise breaks later np.concatenate calls over that column.

    """
    df.loc[[idx], col] = pd.Series([arr], index=[idx])

class VMSI_obj:
    def __init__(self):
        self.V_df = []
        self.C_df = []
        self.E_df = []

class Segmenter:
    def __init__(self, images = None, masks = None, very_far = 300, labelled=False):
        """
        :param: images: (Numpy array) Membrane-stained images to be segmented. WARNING: currently experimental and not working as intended. Default: None.
        :param masks: (Numpy array) Segmented image with edges set to zero and cells set to non-zero. Edges must be 1px wide and 4-connected. Default: None.
        :param very_far: (Int) Maximum distance in pixels between two vertices connected by the same edge. Default: 300.
        :param labelled: (Bool) Whether the segmented cells have been labelled. Default: False.
        """
        self.images = []
        self.masks = []
        self.very_far = very_far

        if images is not None:
            self.images = images
        if masks is not None:
            self.masks = masks
        if not labelled:
            self.masks = measure.label(self.masks)

    def process_segmented_image(self, holes_mask=None):
        """
        Given a segmented mask, produce VMSI_obj for input into VMSI
        """

        # Before processing mask, obtain polygon perimeter and original image label for each cell
        polygon_perimeter = self.polygon_perimeter()
        # Process mask

        # Clear border (create external cell from all cells that run into image boundary)
        tmp1 = seg.clear_border(self.masks)
        # If we are specifying holes, also set cells bordering holes as external cell
        if holes_mask is not None:
            tmp5 = morph.binary_dilation(holes_mask, footprint=np.ones([5,5]))
            hole_adj_cells = np.unique(tmp5 * self.masks)[1:]
            tmp6 = np.isin(self.masks, hole_adj_cells)
            tmp1[tmp6] = 0
        tmp2 = ((self.masks - tmp1)>0).astype(int)
        tmp3 = tmp1 + tmp2

        # Find edge pixels that only separate external cells
        kernel = lambda neighborhood : len(set(neighborhood))
        tmp4 = generic_filter(tmp3, kernel, footprint=np.ones([3,3]))
        tmp1[np.logical_and(tmp1==0,tmp4<3)] = 1

        mask_tmp = tmp1

        # Relabel mask (may not be necessary in future, just for Matlab compatibility)
        mask_tmp = self.relabel(mask_tmp)

        # Create VMSI object to store vertex, cell and edge information
        obj = VMSI_obj()

        obj.C_df = self.find_cells(mask_tmp)
        # Add polygon perimeter and cell label information to C_df

        cell_pwdist = cdist(polygon_perimeter[['centroid_x','centroid_y']], np.array(obj.C_df['centroids'].tolist()))
        matching_cells = np.where(cell_pwdist<=2)
        obj.C_df.loc[obj.C_df.index.values[matching_cells[1]],['label','polygon_perimeter']] = polygon_perimeter[['label','polygon_perimeter']].values[matching_cells[0]]

        obj.V_df, cc = self.find_vertices(mask_tmp, obj.C_df)
        obj.E_df = self.find_edges(obj, mask_tmp, cc)
        obj = self.inject_isolated_cell_topology(obj, mask_tmp)
        self.identify_holes(obj)
        return obj, mask_tmp

    def find_vertices(self, mask, C_df):
        V_df = pd.DataFrame(columns = ['coords','ncells','nverts','edges'])

        branchpoints = self.find_branch_points(mask==0)

        cc = measure.label(branchpoints, connectivity=2)
        if cc.max() == 0:
            raise ValueError(
                "No triple-junction vertices found in this mask's background "
                "skeleton. VMSI's vertex model requires a confluent, "
                "space-filling tissue segmentation (cells touching along "
                "shared edges); it can't infer anything from a mask where "
                "cells are separated by background - there are no cell-cell "
                "junctions to analyse. If these gaps are a segmentation "
                "artifact (cells that should be adjacent but have thin gaps "
                "between them), close them first, e.g. with "
                "skimage.segmentation.expand_labels."
            )
        v = np.array([np.flip(np.round(regionprops.centroid).astype(int)) for regionprops in measure.regionprops(cc)])
        # Each region's coords array can have a different number of pixels
        # (branch points aren't all the same size/shape), so this is
        # genuinely ragged - a is only ever indexed per-region (a[i]) below,
        # never used as a single rectangular array, so dtype=object is
        # correct here rather than trying to stack them.
        a = np.array([regionprops.coords for regionprops in measure.regionprops(cc)], dtype=object)
        # regionprops returns the coordinates in numpy indexing rather than cartesian indexing - e.g.
        # (rows, cols) rather than (x, y) so flip and re-sort coordinates
        a = a[v[:,0].argsort()]
        v = v[v[:,0].argsort()]

        # R[0,:]/R[1,:] are just v's two columns transposed - no need to
        # build this element-by-element in the loop below.
        R = v.T.astype(float)

        # Each vertex's local cell-neighbourhood only needs a small slice
        # around its own branch-point region, so this loop is O(V) with a
        # small constant per iteration - not the quadratic cost below.
        rows = []
        for i in range(v.shape[0]):
            # Flip again to convert back to numpy indexing
            local = mask[min(a[i][:,0])-1:max(a[i][:,0])+2, min(a[i][:,1])-1:max(a[i][:,1])+2]
            ncells = np.unique(local[local!=0])-1
            rows.append({'coords': v[i,:], 'ncells': ncells, 'nverts': np.array([]), 'edges': np.array([])})
        # Build the DataFrame once instead of growing it one row at a time
        # via pd.concat, which copies the whole DataFrame on every
        # iteration (O(V^2) for V vertices instead of O(V)).
        V_df = pd.DataFrame(rows, columns=['coords','ncells','nverts','edges'])

        D = np.add(np.tile(np.sum(np.multiply(R, R), axis=0), (v.shape[0],1)),
                   np.tile(np.sum(np.multiply(R, R), axis=0), (v.shape[0],1)).T) - 2*np.matmul(R.T, R)

        for V in range(len(V_df)):
            for cell in V_df.at[V, 'ncells']:
                C_df.at[cell, 'numv'] += 1
                set_array_at(C_df, cell, 'nverts', np.append(C_df.at[cell, 'nverts'], np.array([V])))

        for C in range(len(C_df)):
            # If cell has no vertices, assume it must border the external cell only
            if C_df.at[C, 'nverts'].size > 0:
                ncells = np.setdiff1d(np.unique(np.hstack(V_df.loc[C_df.at[C, 'nverts'], 'ncells'].tolist())), C)
            else:
                ncells = np.array([0])
            C_df.at[C, 'ncells'] = ncells

        # Identify neighbour vertices. The original approach checked, for
        # every pair of vertices, whether they share >=2 neighbouring cells
        # via np.intersect1d - an O(V^2) Python double loop. That "shared
        # cell count between every pair" is exactly what a vertex-by-cell
        # incidence matrix M gives via M @ M.T: entry (i,j) of that product
        # is sum_c M[i,c]*M[j,c], i.e. the number of cells common to vertices
        # i and j. Each vertex only touches a handful of cells, so M is very
        # sparse and this multiplication is cheap regardless of V, unlike
        # the O(V^2) loop it replaces.
        filtered_ncells = [np.unique(nc[nc != 0]).astype(int) for nc in V_df['ncells']]
        row_idx = np.repeat(np.arange(len(filtered_ncells)), [len(fc) for fc in filtered_ncells])
        col_idx = np.concatenate(filtered_ncells)
        M = coo_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(v.shape[0], len(C_df))).tocsr()
        shared_counts = (M @ M.T).toarray()

        dist_mask = D <= np.power(self.very_far, 2)
        adj = ((shared_counts >= 2) & dist_mask).astype(float)
        np.fill_diagonal(adj, 0)  # a vertex is never its own neighbour (the original loop only ever compared i against j>i)

        for i in range(v.shape[0]):
            # Ensure that the list format remains
            # Pandas will force single-element np.ndarrays to be an integer (not iterable)
            V_df['nverts'].iloc[i] = np.where(adj[i,:]==1)[0].tolist()
        return V_df, cc

    def find_branch_points(self, skel):
        # Vectorized branch point finding; faster than convolving with filter

        skel = np.array(skel, dtype=int)

        branch_points = np.zeros(skel.shape)
        branch_points[1:skel.shape[0]-1,1:skel.shape[1]-1] = skel[2:skel.shape[0],1:skel.shape[1]-1] + skel[0:skel.shape[0]-2,1:skel.shape[1]-1] + \
                                   skel[1:skel.shape[0]-1,2:skel.shape[1]] + skel[1:skel.shape[0]-1,0:skel.shape[1]-2]
        branch_points = np.multiply(branch_points,skel)
        branch_points = branch_points >= 3
        return branch_points

    def find_cells(self, mask):
        # Identify cells, record region information
        # regionprops(mask) is a real cost (an internal full scan of the
        # labelled image, computing many per-region properties) - cache it
        # once and read every property from the same list instead of calling
        # it 5 separate times for centroid/perimeter/inertia/bbox/moments_hu.
        regions = measure.regionprops(mask)

        # regionprops returns the co-ordinates in numpy indexing rather than cartesian indexing - e.g.
        # (rows, cols) rather than (x, y) so flip
        c = np.array([np.flip(r.centroid) for r in regions])
        p = np.array([r.perimeter for r in regions])
        ine = np.array([r.inertia_tensor[np.triu_indices(2)] for r in regions])
        bbox = np.array([[r.bbox[3]-r.bbox[1], r.bbox[2]-r.bbox[0]] for r in regions])
        moments_hu = np.array([r.moments_hu for r in regions])
        cell_props = pd.DataFrame(measure.regionprops_table(mask, properties=('label', 'feret_diameter_max','area')))


        # estimate very_far to be the half the maximum cell perimeter
        self.very_far = np.max(p[1:])/2

        # Build every row as a plain dict first, then construct the DataFrame
        # once at the end. pd.concat in a loop copies the entire
        # already-built DataFrame on every iteration, making this O(N^2) in
        # the number of cells for what should be an O(N) operation.
        rows = []
        for i in range(c.shape[0]):
            rows.append({'centroids': c[i,:], 'nverts': np.array([]), 'numv': 0, 'ncells': np.array([]), 'edges': np.array([]),
                        'area': cell_props.at[i,'area'], 'holes': False, 'inertia': ine[i],
                        'perimeter': p[i], 'polygon_perimeter': 0,
                        'feret_d': cell_props.at[i,'feret_diameter_max'],
                        'moments_hu': moments_hu[i,:], 'bbox': bbox[i,:], 'label': 0})
        C_df = pd.DataFrame(rows, columns=['centroids','nverts','numv','ncells','edges', 'area', 'holes',
                                            'inertia', 'perimeter','polygon_perimeter','feret_d',
                                            'moments_hu','bbox','label'])
        return C_df

    def identify_holes(self, obj):
        """
        Filter out labelled objects that have area greater than 2x the median area and are non-convex
        """
        areas = obj.C_df['area'].to_numpy()
        for i in range(obj.C_df.shape[0]):
            vcoords = np.array(obj.V_df.loc[obj.C_df.at[i, 'nverts'], 'coords'].tolist())
            if vcoords.shape[0] >= 3:
                try:
                    hull = ConvexHull(vcoords)
                    if hull.simplices.shape[0] < vcoords.shape[0] and obj.C_df.at[i, 'area'] > 2*np.median(areas):
                        obj.C_df.at[i, 'holes'] = True
                # else:
                except:
                    pass
            else:
                obj.C_df.at[i, 'holes'] = True
        return

    def relabel(self, mask):
        """
        If cells aren't sequenctially label, relabel them
        """
        new_mask = np.zeros(mask.shape, dtype=int)
        ids = np.sort(np.unique(mask))

        for i in range(1,len(ids)):
            new_mask[mask==ids[i]] = i
        return new_mask

    def find_edges(self, obj, mask, cc):
        E_df = pd.DataFrame(columns = ['pixels','verts','cells'])

        l_dat = mask
        b_dat = (l_dat == 0).astype(int)

        rv = np.vstack(obj.V_df['coords'])
        verts = np.zeros(b_dat.shape)
        verts[rv[:,1],rv[:,0]] = 1

        b_dat[np.where(verts == 1)] = 0
#        b_end  = b_dat * morph.dilation(verts, morph.disk(1))

        b_dat[cc != 0] = 0
#        b_end = (b_end * b_dat) + (self.endpoints(b_dat) * b_dat)
        # Not sure what the Matlab code is trying to accomplish but it doesn't seem to work so try another method
        b_end = self.endpoints(b_dat) * b_dat

        re = np.argwhere(b_end.T != 0)
        D = cdist(re, rv)

        b_l = measure.label(b_dat.T, connectivity=1).T
        end_labels = b_l[re[:,1],re[:,0]]
        b_props = measure.regionprops(b_l)

        # Build every row as a plain dict first, then construct the
        # DataFrame once - see the same fix in find_cells/find_vertices for
        # why pd.concat in a loop is an O(N^2) way to do this.
        edge_rows = []
        # Diagnostics only used if this ends up finding zero edges, to
        # pinpoint which of the three acceptance conditions is failing
        # universally instead of guessing.
        rejected_no_endpoint = 0
        rejected_not_neighbours = 0
        rejected_both_exterior = 0
        rejected_neighbour_dists = []
        for i in range(1, len(np.unique(b_l))):
            end_points = np.argwhere(end_labels==i)

            v1 = -1
            v2 = -1

            # Edges with 1 endpoint are generally 1-length; ignore these
            if len(end_points) == 2:
                v1 = np.argmin(D[end_points[0],:])
                v2 = np.argmin(D[end_points[1],:])
                if (v1 == v2):
                    sort1 = np.sort(D[end_points[0],:]).squeeze()
                    sort2 = np.sort(D[end_points[1],:]).squeeze()
                    if abs(sort1[0] - sort1[1]) <= np.sqrt(3):
                        # Break the v1==v2 tie by falling back to the
                        # second-closest vertex to end_points[0] - [0] here
                        # was a pre-existing bug that just recomputed the
                        # same (closest) vertex np.argmin already found two
                        # lines above, making this whole disambiguation a
                        # no-op and leaving v1==v2 unresolved.
                        v1 = np.argsort(D[end_points[0],:]).squeeze()[1]
                    elif abs(sort2[0] - sort2[1]) <= np.sqrt(3):
                        v2 = np.argsort(D[end_points[1],:]).squeeze()[1]

            if (v1 != -1) and (v2 != -1) and (v2 in obj.V_df.at[v1, 'nverts']) and ((v1 not in obj.C_df.at[0, 'nverts']) or (v2 not in obj.C_df.at[0, 'nverts'])):
                pix = np.ravel_multi_index(np.flip(b_props[i-1].coords.T), mask.shape[::-1])
                verts = np.array([v1, v2])
                cells = np.intersect1d(obj.V_df.at[v1, 'ncells'], obj.V_df.at[v2, 'ncells'])
                edge_rows.append({'pixels': pix, 'verts': verts, 'cells': cells})
            elif v1 == -1 or v2 == -1:
                rejected_no_endpoint += 1
            elif v2 not in obj.V_df.at[v1, 'nverts']:
                rejected_not_neighbours += 1
                rejected_neighbour_dists.append(np.linalg.norm(obj.V_df.at[v1,'coords'] - obj.V_df.at[v2,'coords']))
            else:
                rejected_both_exterior += 1
        E_df = pd.DataFrame(edge_rows, columns=['pixels','verts','cells'])

        if len(E_df) == 0:
            n_verts = len(obj.V_df)
            n_touch_exterior = sum(1 for k in range(n_verts) if k in obj.C_df.at[0, 'nverts'])
            dist_info = ""
            if rejected_neighbour_dists:
                dist_info = (f" Of those, pixel distances ranged "
                             f"[{min(rejected_neighbour_dists):.1f}, {max(rejected_neighbour_dists):.1f}] "
                             f"against self.very_far={self.very_far:.1f} (i.e. a pair needs distance <= "
                             f"{self.very_far:.1f} AND >=2 shared cells to already count as neighbours).")
            n_segments = len(np.unique(b_l)) - 1
            seg_sizes = [len(p.coords) for p in b_props]
            print(f"[diag] candidate segments: {rejected_no_endpoint} had no valid 2-endpoint match, "
                  f"{rejected_not_neighbours} matched 2 real vertices that find_vertices didn't already "
                  f"consider neighbours,{dist_info} {rejected_both_exterior} matched neighbouring vertices "
                  f"that both touch the exterior cell. very_far={self.very_far:.1f}, "
                  f"{n_touch_exterior}/{n_verts} vertices touch the exterior cell.")
            print(f"[diag] {n_segments} background skeleton segments were traced for only {n_verts} real "
                  f"vertices ({n_segments/max(n_verts,1):.1f} segments/vertex - a normal 1px-wide, "
                  f"single-pixel-thick boundary tessellation should give roughly 1.5). Segment pixel-length "
                  f"range: [{min(seg_sizes) if seg_sizes else 0}, {max(seg_sizes) if seg_sizes else 0}], "
                  f"median {np.median(seg_sizes) if seg_sizes else 0:.1f}. Many short segments relative to "
                  f"vertex count usually means the background between cells isn't a clean single-pixel-wide "
                  f"skeleton (e.g. thicker or jagged boundaries), which fragments into many small spurious "
                  f"pieces instead of a handful of long ones once branch points are removed.")
            raise ValueError(
                "No edges could be traced between this mask's vertices - branch-point vertices exist, "
                "but no background skeleton segment between any pair of them satisfied both the distance "
                "threshold (Segmenter.very_far) and the already-known-neighbour check. This is the same "
                "underlying issue as having no confluent interior tissue - VMSI's vertex-network inference "
                "needs real, traceable edges to work with. Use run_isolated_cells (or run_VMSI, which "
                "falls back to it automatically) instead."
            )

        # Edit V_df and C_df with edge information
        # The original approach re-stacked ALL of E_df's verts into one
        # array (np.vstack(E_df['verts'])) and re-scanned it from scratch on
        # every single (v, nv) pair checked below - O(E) work repeated for
        # every one of the ~3V vertex-neighbour pairs, i.e. O(V*E) overall,
        # which is just as costly as the O(V^2) adjacency loop fixed above
        # for a tissue where both V and E scale with cell count. Since new
        # edges can be created mid-loop (the "Create new edge" branch) and
        # later iterations need to see them, a one-off precomputed lookup
        # isn't enough - use a dict kept up to date as edges are added, so
        # every lookup is O(1) instead of O(E).
        edge_lookup = {}
        for idx, vp in enumerate(E_df['verts']):
            edge_lookup.setdefault((vp[0], vp[1]), idx)

        for v in range(0, len(obj.V_df)):
            for nv in obj.V_df.at[v, 'nverts']:
                idx = edge_lookup.get((v, nv))
                if idx is None:
                    idx = edge_lookup.get((nv, v))

                if idx is not None:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], idx)
                elif (v not in obj.C_df.at[0, 'nverts']) and (nv not in obj.C_df.at[0, 'nverts']):
                    # Create new edge
                    line = draw.line(obj.V_df.at[v, 'coords'][1], obj.V_df.at[v, 'coords'][0], obj.V_df.at[nv, 'coords'][1], obj.V_df.at[nv, 'coords'][0])
                    pix = np.ravel_multi_index(np.flip(line,axis=0), mask.shape[::-1])
                    verts = np.array([v, nv])
                    cells = np.intersect1d(obj.V_df.at[v, 'ncells'], obj.V_df.at[nv, 'ncells'])
                    edge_df = pd.DataFrame({'pixels':[pix],'verts':[verts],'cells':[cells]})
                    E_df = pd.concat([E_df, edge_df], ignore_index=True)
                    # len(E_df) here is actually one past this new row's true
                    # 0-indexed position (len(E_df)-1) - a pre-existing
                    # off-by-one in this specific line, kept exactly as-is
                    # (not something this speed-up is meant to change).
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], len(E_df))
                    # The lookup dict, however, must hold the *true* index -
                    # it's what later re-scans of E_df via argwhere would
                    # have found for any other vertex searching for this
                    # same edge (argwhere always finds the real position;
                    # the off-by-one above only ever affected this creating
                    # vertex's own stored value, not the array itself).
                    # Using the off-by-one value here instead would silently
                    # propagate v's bug into every other vertex that later
                    # looks up this edge, which the original code never did.
                    edge_lookup[(v, nv)] = len(E_df) - 1
                else:
                    obj.V_df.at[v, 'edges'] = np.append(obj.V_df.at[v, 'edges'], np.array([-1]))

        for c in range(1, len(obj.C_df)):
            c_verts = obj.C_df.at[c, 'nverts']

            # np.ndarray uses size
            if c_verts.size > 1:
                c_coords = np.vstack(obj.V_df.loc[c_verts, 'coords'].to_list())
                c_coords = c_coords - np.mean(c_coords, axis=0)

                # Sort vertices in clockwise direction
                theta = np.mod(np.arctan2(c_coords[:,1], c_coords[:,0]), 2*np.pi)
                c_verts = c_verts[np.argsort(theta)]
                c_verts = np.append(c_verts, c_verts[0])


                if c not in obj.C_df.at[0, 'ncells']:
                    for v in range(0, len(c_verts)-1):
                        if (c_verts[v+1] in obj.V_df.at[c_verts[v], 'nverts']):
                            obj.C_df.at[c, 'edges'] = np.append(obj.C_df.at[c, 'edges'], np.intersect1d(obj.V_df.at[c_verts[v], 'edges'], obj.V_df.at[c_verts[v+1], 'edges']))
                        else:
                            obj.C_df.at[c, 'edges'] = np.append(obj.C_df.at[c, 'edges'], -1)

        return E_df

    def endpoints(self, image):
        # Define endpoint as pixel with only 1 4-connected neighbor
        # This requires the skeletonized image to be 4-connnected
        image = image.astype(int)
        k = np.array([[0,1,0],[1,0,1],[0,1,0]])
        neighborhood_count = ndi.convolve(image,k, mode='constant', cval=1)
        neighborhood_count[~image.astype(bool)] = 0
        return neighborhood_count == 1

    def inject_isolated_cell_topology(self, obj, mask_tmp, n_verts=6):
        """
        For cells with no real-cell neighbors (isolated / suspended cells), sample
        n_verts artificial vertices along the cell contour and create edges shared
        with the background (cell 0).  This lets VMSI infer their mechanics via
        infer_isolated_cells() using Young-Laplace directly.

        :param obj: VMSI_obj populated by find_cells / find_vertices / find_edges.
        :param mask_tmp: labelled mask (same coordinate system used throughout).
        :param n_verts: number of artificial vertices to place around each isolated cell.
        """
        for cell_idx in range(1, len(obj.C_df)):
            if obj.C_df.at[cell_idx, 'nverts'].size > 0:
                continue  # cell already has topology from real junctions

            cell_mask = (mask_tmp == cell_idx).astype(float)
            contours = measure.find_contours(cell_mask, 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)      # (N, 2) in (row, col)
            contour_xy = contour[:, ::-1].copy()  # convert to (x, y)

            # Arc-length parameterisation of the contour
            diffs = np.diff(contour_xy, axis=0)
            arc_lens = np.concatenate([[0.0],
                                       np.cumsum(np.linalg.norm(diffs, axis=1))])
            total_len = arc_lens[-1]
            if total_len == 0:
                continue

            # Interpolate n_verts positions at equal arc-length spacing
            targets = np.linspace(0, total_len, n_verts, endpoint=False)
            sampled_pts = np.stack([
                np.interp(targets, arc_lens, contour_xy[:, 0]),
                np.interp(targets, arc_lens, contour_xy[:, 1])
            ], axis=1)

            # --- vertices ---
            start_v = len(obj.V_df)
            v_indices = np.arange(start_v, start_v + n_verts, dtype=int)

            for i, pt in enumerate(sampled_pts):
                obj.V_df = pd.concat([obj.V_df, pd.DataFrame({
                    'coords': [pt.tolist()],
                    'ncells': [np.array([0, cell_idx])],
                    'nverts': [np.array([v_indices[(i - 1) % n_verts],
                                         v_indices[(i + 1) % n_verts]])],
                    'edges':  [np.array([])]
                })], ignore_index=True)

            # --- edges ---
            start_e = len(obj.E_df)

            for i in range(n_verts):
                v1 = int(v_indices[i])
                v2 = int(v_indices[(i + 1) % n_verts])
                pt1 = sampled_pts[i]
                pt2 = sampled_pts[(i + 1) % n_verts]

                # draw.line expects (row, col) = (y, x)
                line = draw.line(int(round(pt1[1])), int(round(pt1[0])),
                                 int(round(pt2[1])), int(round(pt2[0])))
                pix = np.ravel_multi_index(np.flip(line, axis=0),
                                           mask_tmp.shape[::-1])
                e_idx = start_e + i

                obj.E_df = pd.concat([obj.E_df, pd.DataFrame({
                    'pixels': [pix],
                    'verts':  [np.array([v1, v2])],
                    'cells':  [np.array([0, cell_idx])]
                })], ignore_index=True)

                obj.V_df.at[v1, 'edges'] = np.append(obj.V_df.at[v1, 'edges'], e_idx)
                obj.V_df.at[v2, 'edges'] = np.append(obj.V_df.at[v2, 'edges'], e_idx)

            # --- update C_df ---
            obj.C_df.at[cell_idx, 'nverts'] = v_indices
            obj.C_df.at[cell_idx, 'numv']   = n_verts
            set_array_at(obj.C_df, cell_idx, 'ncells', np.array([0]))
            obj.C_df.at[cell_idx, 'edges']  = np.arange(start_e, start_e + n_verts,
                                                          dtype=int)

            set_array_at(obj.C_df, 0, 'ncells', np.append(obj.C_df.at[0, 'ncells'], cell_idx))
            set_array_at(obj.C_df, 0, 'nverts', np.append(obj.C_df.at[0, 'nverts'], v_indices))

        return obj

    def segment_image(self, diameter=None, channels=[0,0], use_model='default'):
        """
        :param diameter: estimated diameter (px) for cells in image. If not specified, this will be estimated from the image
        :param channels: channels containing membrane and nuclear staining of image. 0 - Grayscale, 1 - R, 2 - G, 3 - B
        :param use_model: which Cellpose neural network to use. 'default' - Cyto2, 'custom' - custom trained model.
        :return: segmented image
        """
        from cellpose import models, utils, plot
        image = self.images.copy()
        image = image.astype(float)

        # Assuming membrane staining instead of cytoplasm, invert image before segmenting with Cellpose
        def normalise_image(image):
            image_norm = image.copy()
            ub = np.percentile(image, 99)
            lb = np.percentile(image, 1)
            image_norm[image_norm>ub] = ub
            image_norm[image_norm<lb] = lb
            image_norm = np.divide(image_norm-lb, ub - lb)
            return image_norm

        if channels != [0,0]:
            image[:,:,channels[0]-1] = 1-normalise_image(image[:,:,channels[0]-1])
            image[:,:,channels[1]-1] = normalise_image(image[:,:,channels[1]-1])
        else:
            image = normalise_image(image)

        if use_model == 'default':
            model = models.Cellpose(model_type='cyto2')
        elif use_model == 'custom':
            import pathlib
            src_path = str(pathlib.Path(__file__).parent.resolve())
            modeldir = f'{src_path}/cellpose_models/cellpose_residual_on_style_on_concatenation_off_train_folder_2022_03_24_00_26_36.748195'
            model = models.Cellpose(model_dir=modeldir, net_avg=False)
        else:
            return "Invalid use_model option. Available models are 'default', 'custom'."

        masks, flows, styles, diams = model.eval(image, diameter=diameter, channels=channels, progress=True)
        segmented_image = masks

        return segmented_image

    def polygon_perimeter(self):
        """

        Identify vertices and calculate polygon perimeter for each cell

        :return:
        """

        branchpoints = self.find_branch_points(self.masks==0)
        labels = np.unique(self.masks)
        labels = labels[labels!=0]
        res = pd.DataFrame(np.zeros([len(labels),1]), index=labels, columns=['polygon_perimeter'])

        # A 3x3 dilation can only ever turn a pixel True if it's within 1px
        # of an already-True pixel, so its effect on a single cell's mask is
        # entirely contained within that cell's bounding box expanded by 1px
        # on each side - dilating (and multiplying by branchpoints) on the
        # full image every time is exact but wasteful, since almost every
        # pixel touched is nowhere near the cell in question. Crop to that
        # bbox instead (same result, verified directly against the
        # full-image version), so each of the N cells does work proportional
        # to its own size instead of the whole image's.
        bboxes = {r.label: r.bbox for r in skimage.measure.regionprops(self.masks)}
        height, width = self.masks.shape

        for label in res.index.values:
            min_row, min_col, max_row, max_col = bboxes[label]
            r0, r1 = max(min_row-1, 0), min(max_row+1, height)
            c0, c1 = max(min_col-1, 0), min(max_col+1, width)

            local_cell = self.masks[r0:r1, c0:c1] == label
            local_branch = branchpoints[r0:r1, c0:c1]
            local_dilated = skimage.morphology.binary_dilation(local_cell, footprint=np.ones([3,3]))
            vertices = np.array(np.where((local_dilated * local_branch) > 0)).T + np.array([r0, c0])

            # calculate polygon perimeter
            v_norm = vertices - np.mean(vertices, axis=0)
            theta = np.mod(np.arctan2(v_norm[:,1], v_norm[:,0]), 2*np.pi)
            vertices = vertices[np.argsort(theta),:]
            perim = 0
            for i in range(vertices.shape[0]):
                v1 = vertices[i,:]
                v2 = vertices[np.mod(i+1, vertices.shape[0]),:]
                perim += np.linalg.norm(v1-v2)
            res.at[label, 'polygon_perimeter'] = perim
        centroids = pd.DataFrame(skimage.measure.regionprops_table(self.masks, properties=['label','centroid']))
        centroids.columns = ['label','centroid_y','centroid_x']
        centroids.index = centroids['label']
        res = pd.concat([res, centroids], axis=1)
        return res