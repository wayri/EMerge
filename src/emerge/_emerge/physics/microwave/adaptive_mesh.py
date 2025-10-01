# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

from .microwave_data import MWField
import numpy as np
from ...mth.optimized import matmul, outward_normal, cross_c
from numba import njit, f8, c16, i8, types, prange # type: ignore, p
from loguru import logger

def print_sparam_matrix(pre: str, S: np.ndarray):
    """
    Print an N x N complex S-parameter matrix in dB∠deg format.
    Magnitude in dB rounded to 2 decimals, phase in degrees with 1 decimal.
    """
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square (N x N) complex matrix")

    N = S.shape[0]
    logger.debug(pre+"S-parameter matrix (dB ∠ deg):")

    for i in range(N):
        row_str = []
        for j in range(N):
            mag_db = 20 * np.log10(np.abs(S[i, j]) + np.finfo(float).eps)
            phase_deg = np.degrees(np.angle(S[i, j]))
            row_str.append(f"{mag_db:6.2f} dB ∠ {phase_deg:6.1f}°")
        logger.debug(" | ".join(row_str))
        
def compute_convergence(Sold: np.ndarray, Snew: np.ndarray) -> float:
    """
    Return a single scalar: max |Snew - Sold|.
    Works for shapes (N,N) or (..., N, N); reduces over all axes.
    """
    
    Sold = np.asarray(Sold)
    Snew = np.asarray(Snew)
    print_sparam_matrix('Old:',Sold)
    print_sparam_matrix('New',Snew)
    if Sold.shape != Snew.shape:
        raise ValueError("Sold and Snew must have identical shapes")
    #amp_conv = float(np.abs(np.abs(Snew) - np.abs(Sold)).max())
    mag_conv = float(np.abs(np.abs(Snew)-np.abs(Sold)).max())
    amp_conv = float(np.abs(Snew - Sold).max())
    phase_conv = float(np.abs(np.angle(np.diag(Sold)/np.diag(Snew))).max()) * 180/np.pi
    return amp_conv, mag_conv, phase_conv

def select_refinement_indices(errors: np.ndarray, refine: float) -> np.ndarray:
    """
    Pick indices to refine based on two rules, then take the smaller set:
      (A) All indices with |error| >= (1 - refine) * max(|error|)
      (B) The top ceil(refine * N) indices by |error|
    Returns indices sorted from largest to smallest |error|.
    """
    errs = np.abs(np.ravel(errors))
    N = errs.size
    if N == 0:
        return np.array([], dtype=int)
    refine = float(np.clip(refine, 0.0, 1.0))
    if refine == 0.0:
        return np.array([], dtype=int)

    # Sorted indices by decreasing amplitude
    sorted_desc = np.argsort(errs)[::-1]

    # Rule A: threshold by amplitude
    thresh = (1.0 - refine) * errs[sorted_desc[0]]
    A = np.flatnonzero(errs >= thresh)

    # Rule B: top-k by count
    k = int(np.ceil(refine * N))
    B = sorted_desc[:k]

    # Choose the smaller set (tie-breaker: use B)
    chosen = B#A if A.size > B.size else B
    
    # Return chosen indices sorted from largest to smallest amplitude
    mask = np.zeros(N, dtype=bool)
    mask[chosen] = True
    return sorted_desc[mask[sorted_desc]]

@njit(f8[:](i8, f8[:,:], f8, f8, f8[:]), cache=True, nogil=True, parallel=False)
def compute_size(id: int, coords: np.ndarray, q: float, scaler: float, dss: np.ndarray) -> float:
    """Optimized function to compute the size impressed by size constraint points on each other size constraint point.

    Args:
        id (int): _description_
        coords (np.ndarray): _description_
        q (float): _description_
        scaler (float): _description_
        dss (np.ndarray): _description_

    Returns:
        float: _description_
    """
    N = dss.shape[0]
    sizes = np.zeros((N,), dtype=np.float64)-1.0
    x, y, z = coords[:,id]
    for n in prange(N):
        if n == id:
            sizes[n] = dss[n]*scaler
            continue
        nsize = scaler*dss[n]/q - (1-q)/q * ((x-coords[0,n])**2 + (y-coords[1,n])**2 + (z-coords[2,n])**2)**0.5
        sizes[n] = nsize
    return sizes

@njit(f8[:](f8[:,:], i8, i8[:]), cache=True, nogil=True, parallel=True)
def nbmin(matrix, axis, include):
    
    if axis==0:
        N = matrix.shape[1]
        out = np.empty((N,), dtype=np.float64)
        for n in prange(N):
            out[n] = np.min(matrix[include==1,n])
        return out
    if axis==1:
        N = matrix.shape[0]
        out = np.empty((N,), dtype=np.float64)
        for n in prange(N):
            out[n] = np.min(matrix[n,include==1])
        return out
    else:
        out = np.empty((N,), dtype=np.float64)
        return out   
    
@njit(i8(i8, f8[:,:], f8, i8[:]), cache=True, nogil=True, parallel=False)
def can_remove(index: int, M: np.ndarray, scaling: float, include: np.ndarray) -> int:

    ratio = np.min(M[index,:] / nbmin(M, 1, include))
    
    if ratio > scaling:
        return 1
    return 0

@njit(i8[:](f8[:,:], f8, f8[:], f8, f8), cache=True, nogil=True, parallel=False)
def reduce_point_set(coords: np.ndarray, q: float, dss: np.ndarray, scaler: float, keep_percentage: float) -> list[int]:
    N = dss.shape[0]
    impressed_size = np.zeros((N,N), np.float64)
    
    include = np.ones((N,), dtype=np.int64)
    
    for n in range(N):
        impressed_size[:,n] = compute_size(n, coords, q, scaler, dss)
    
    current_min = nbmin(impressed_size, 1, include)
    
    counter = 0
    for i in range(N):
        
        if include[i]==0:
            continue

        if (N-counter)/N < keep_percentage:
            break
        
        n_imposed = np.sum(impressed_size[:,i] <= (current_min*1.01))
        
        if n_imposed == 0:
            include[i] = 0
            counter = counter + 1
        
        
    
    ids = np.arange(N)
    output = ids[include==1]
    return output


@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex 0.1005964238ers of one tetrahedron, in *its* order
        (v0, v1, v2, v3).

    triangle_ids : 2-D int64 array (nTri × 3)
        Each row is a global-ID triple of one face that belongs to this tet.

    Returns
    -------
    local_tris   : 2-D int64 array (nTri × 3)
        Same triangles, but every entry replaced by the local index
        0,1,2,3 that the vertex has inside this tetrahedron.
        (Guaranteed to be ∈{0,1,2,3}; no -1 ever appears if the input
        really belongs to the tet.)
    """
    ndim = triangle_ids.shape[0]
    ntri = triangle_ids.shape[1]
    out  = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):                 # each triangle
        for j in range(ndim):                # each vertex in that triangle
            gid = triangle_ids[j, t]      # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k         # store local index 0-3
                    break                 # stop the k-loop

    return out

@njit(f8(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def compute_volume(xs, ys, zs):
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    return np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)

@njit(types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def tet_coefficients(xs, ys, zs):
    ## THIS FUNCTION WORKS
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)
    
    aas[0] = x2*y3*z4 - x2*y4*z3 - x3*y2*z4 + x3*y4*z2 + x4*y2*z3 - x4*y3*z2
    aas[1] = -x1*y3*z4 + x1*y4*z3 + x3*y1*z4 - x3*y4*z1 - x4*y1*z3 + x4*y3*z1
    aas[2] = x1*y2*z4 - x1*y4*z2 - x2*y1*z4 + x2*y4*z1 + x4*y1*z2 - x4*y2*z1
    aas[3] = -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x2*y3*z1 - x3*y1*z2 + x3*y2*z1
    bbs[0] = -y2*z3 + y2*z4 + y3*z2 - y3*z4 - y4*z2 + y4*z3
    bbs[1] = y1*z3 - y1*z4 - y3*z1 + y3*z4 + y4*z1 - y4*z3
    bbs[2] = -y1*z2 + y1*z4 + y2*z1 - y2*z4 - y4*z1 + y4*z2
    bbs[3] = y1*z2 - y1*z3 - y2*z1 + y2*z3 + y3*z1 - y3*z2
    ccs[0] = x2*z3 - x2*z4 - x3*z2 + x3*z4 + x4*z2 - x4*z3
    ccs[1] = -x1*z3 + x1*z4 + x3*z1 - x3*z4 - x4*z1 + x4*z3
    ccs[2] = x1*z2 - x1*z4 - x2*z1 + x2*z4 + x4*z1 - x4*z2
    ccs[3] = -x1*z2 + x1*z3 + x2*z1 - x2*z3 - x3*z1 + x3*z2
    dds[0] = -x2*y3 + x2*y4 + x3*y2 - x3*y4 - x4*y2 + x4*y3
    dds[1] = x1*y3 - x1*y4 - x3*y1 + x3*y4 + x4*y1 - x4*y3
    dds[2] = -x1*y2 + x1*y4 + x2*y1 - x2*y4 - x4*y1 + x4*y2
    dds[3] = x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2

    return aas, bbs, ccs, dds, V

@njit(c16[:](f8[:], f8[:,:], c16[:], i8[:,:], i8[:,:]), cache=True, nogil=True)
def compute_field(coords: np.ndarray, 
                 vertices: np.ndarray,
                 Etet: np.ndarray, 
                 l_edge_ids: np.ndarray, 
                 l_tri_ids: np.ndarray):
    
    x = coords[0]
    y = coords[1]
    z = coords[2]
    
    xvs = vertices[0,:]
    yvs = vertices[1,:]
    zvs = vertices[2,:]
    
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    
    Em1s = Etet[0:6]
    Ef1s = Etet[6:10]
    Em2s = Etet[10:16]
    Ef2s = Etet[16:20]
    
    Exl = 0.0 + 0.0j
    Eyl = 0.0 + 0.0j
    Ezl = 0.0 + 0.0j
    
    V1 = (216*V**3)
    for ie in range(6):
        Em1, Em2 = Em1s[ie], Em2s[ie]
        edgeids = l_edge_ids[:, ie]
        a1, a2 = a_s[edgeids]
        b1, b2 = b_s[edgeids]
        c1, c2 = c_s[edgeids]
        d1, d2 = d_s[edgeids]
        x1, x2 = xvs[edgeids]
        y1, y2 = yvs[edgeids]
        z1, z2 = zvs[edgeids]
        F1 = (a1 + b1*x + c1*y + d1*z)
        F2 = (a2 + b2*x + c2*y + d2*z)
        L = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        ex =  L*(Em1*F1 + Em2*F2)*(b1*F2 - b2*F1)/V1
        ey =  L*(Em1*F1 + Em2*F2)*(c1*F2 - c2*F1)/V1
        ez =  L*(Em1*F1 + Em2*F2)*(d1*F2 - d2*F1)/V1

        Exl += ex
        Eyl += ey
        Ezl += ez
    
    for ie in range(4):
        Em1, Em2 = Ef1s[ie], Ef2s[ie]
        triids = l_tri_ids[:, ie]
        a1, a2, a3 = a_s[triids]
        b1, b2, b3 = b_s[triids]
        c1, c2, c3 = c_s[triids]
        d1, d2, d3 = d_s[triids]

        x1, x2, x3 = xvs[l_tri_ids[:, ie]]
        y1, y2, y3 = yvs[l_tri_ids[:, ie]]
        z1, z2, z3 = zvs[l_tri_ids[:, ie]]

        L1 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
        L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        
        F1 = (a1 + b1*x + c1*y + d1*z)
        F2 = (a2 + b2*x + c2*y + d2*z)
        F3 = (a3 + b3*x + c3*y + d3*z)
        
        Q1 = Em1*L1*F2
        Q2 = Em2*L2*F3
        ex =  (-Q1*(b1*F3 - b3*F1) + Q2*(b1*F2 - b2*F1))/V1
        ey =  (-Q1*(c1*F3 - c3*F1) + Q2*(c1*F2 - c2*F1))/V1
        ez =  (-Q1*(d1*F3 - d3*F1) + Q2*(d1*F2 - d2*F1))/V1
        
        Exl += ex
        Eyl += ey
        Ezl += ez

    out = np.zeros((3,), dtype=np.complex128)
    out[0] = Exl
    out[1] = Eyl
    out[2] = Ezl
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], c16[:], i8[:,:], i8[:,:]), cache=True, nogil=True)
def compute_curl(coords: np.ndarray, 
                 vertices: np.ndarray,
                 Etet: np.ndarray, 
                 l_edge_ids: np.ndarray, 
                 l_tri_ids: np.ndarray):
    
    x = coords[0,:]
    y = coords[1,:]
    z = coords[2,:]
    
    xvs = vertices[0,:]
    yvs = vertices[1,:]
    zvs = vertices[2,:]
    
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    
    Em1s = Etet[0:6]
    Ef1s = Etet[6:10]
    Em2s = Etet[10:16]
    Ef2s = Etet[16:20]
    
    Exl = np.zeros((x.shape[0],), dtype=np.complex128)
    Eyl = np.zeros((x.shape[0],), dtype=np.complex128)
    Ezl = np.zeros((x.shape[0],), dtype=np.complex128)
    
    V1 = (216*V**3)
    V2 = (72*V**3)
    
    for ie in range(6):
        Em1, Em2 = Em1s[ie], Em2s[ie]
        edgeids = l_edge_ids[:, ie]
        a1, a2 = a_s[edgeids]
        b1, b2 = b_s[edgeids]
        c1, c2 = c_s[edgeids]
        d1, d2 = d_s[edgeids]
        x1, x2 = xvs[edgeids]
        y1, y2 = yvs[edgeids]
        z1, z2 = zvs[edgeids]

        L = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        C1 = Em1*a1
        C2 = Em1*b1
        C3 = Em1*c1
        C4 = Em1*c2
        C5 = Em2*a2
        C6 = Em2*b2
        C7 = Em2*c1
        C8 = Em2*c2
        C9 = Em1*b2
        C10 = Em2*b1
        D1 = c1*d2
        D2 = c2*d1
        D3 = d1*d2
        D4 = d1*d1
        D5 = c2*d2
        D6 = d2*d2
        D7 = b1*d2
        D8 = b2*d1
        D9 = c1*d1
        D10 = b2*d2
        D11 = b1*c2
        D12 = b2*c1
        D13 = c1*c2
        D14 = c1*c1
        D15 = b2*c2
        
        ex =  L*(-C1*D1 + C1*D2 - C2*D1*x + C2*D2*x - C3*D1*y + C3*D2*y - C3*D3*z + C4*D4*z - C5*D1 + C5*D2 - C6*D1*x + C6*D2*x - C7*D5*y - C7*D6*z + C8*D2*y + C8*D3*z)/V2
        ey =  L*(C1*D7 - C1*D8 + C2*D7*x - C2*D8*x + C2*D1*y + C2*D3*z - C9*D9*y - C9*D4*z + C5*D7 - C5*D8 + C10*D10*x + C10*D5*y + C10*D6*z - C6*D8*x - C6*D2*y - C6*D3*z)/V2
        ez =  L*(-C1*D11 + C1*D12 - C2*D11*x + C2*D12*x - C2*D13*y - C2*D2*z + C9*D14*y + C9*D9*z - C5*D11 + C5*D12 - C10*D15*x - C10*c2*c2*y - C10*D5*z + C6*D12*x + C6*D13*y + C6*D1*z)/V2
        Exl += ex
        Eyl += ey
        Ezl += ez
    
    for ie in range(4):
        Em1, Em2 = Ef1s[ie], Ef2s[ie]
        triids = l_tri_ids[:, ie]
        a1, a2, a3 = a_s[triids]
        b1, b2, b3 = b_s[triids]
        c1, c2, c3 = c_s[triids]
        d1, d2, d3 = d_s[triids]

        x1, x2, x3 = xvs[l_tri_ids[:, ie]]
        y1, y2, y3 = yvs[l_tri_ids[:, ie]]
        z1, z2, z3 = zvs[l_tri_ids[:, ie]]

        L1 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
        L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        F1 = (a3 + b3*x + c3*y + d3*z)
        F2 = (a1 + b1*x + c1*y + d1*z)
        F3 = (a2 + b2*x + c2*y + d2*z)
        N1 = (d1*F1 - d3*F2)
        N2 = (c1*F1 - c3*F2)
        N3 = (c1*d3 - c3*d1)
        N4 = (d1*F3 - d2*F2)
        N5 = (c1*F3 - c2*F2)
        D1 = c1*d2
        D2 = c2*d1
        N6 = (D1 - D2)
        N7 = (b1*F1 - b3*F2)
        N8 = (b1*d3 - b3*d1)
        N9 = (b1*F3 - b2*F2)
        D7 = b1*d2
        D8 = b2*d1
        N10 = (D7 - D8)
        D11 = b1*c2
        D12 = b2*c1
        ex =  (Em1*L1*(-c2*N1 + d2*N2 + 2*N3*F3) - Em2*L2*(-c3*N4 + d3*N5 + 2*N6*F1))/V1
        ey =  (-Em1*L1*(-b2*N1 + d2*N7 + 2*N8*F3) + Em2*L2*(-b3*N4 + d3*N9 + 2*N10*F1))/V1
        ez =  (Em1*L1*(-b2*N2 + c2*N7 + 2*(b1*c3 - b3*c1)*F3) - Em2*L2*(-b3*N5 + c3*N9 + 2*(D11 - D12)*F1))/V1
        
        Exl += ex
        Eyl += ey
        Ezl += ez

    out = np.zeros((3,x.shape[0]), dtype=np.complex128)
    out[0,:] = Exl
    out[1,:] = Eyl
    out[2,:] = Ezl
    return out

@njit(c16[:](f8[:], f8[:,:], c16[:], i8[:,:], i8[:,:], c16[:,:]), cache=True, nogil=True)
def compute_curl_curl(coords: np.ndarray, 
                    vertices: np.ndarray,
                    Etet: np.ndarray, 
                    l_edge_ids: np.ndarray, 
                    l_tri_ids: np.ndarray,
                    Um: np.ndarray):
    
    uxx, uxy, uxz = Um[0,0], Um[0,1], Um[0,2]
    uyx, uyy, uyz = Um[1,0], Um[1,1], Um[1,2]
    uzx, uzy, uzz = Um[2,0], Um[2,1], Um[2,2]
    
    x = coords[0]
    y = coords[1]
    z = coords[2]
    
    xvs = vertices[0,:]
    yvs = vertices[1,:]
    zvs = vertices[2,:]
    
    Exl = 0.0 + 0.0j
    Eyl = 0.0 + 0.0j
    Ezl = 0.0 + 0.0j
    
    a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
    
    Em1s = Etet[0:6]
    Ef1s = Etet[6:10]
    Em2s = Etet[10:16]
    Ef2s = Etet[16:20]
    
    V1 = (216*V**3)
    
    for ie in range(6):
        Em1, Em2 = Em1s[ie], Em2s[ie]
        edgeids = l_edge_ids[:, ie]
        a1, a2 = a_s[edgeids]
        b1, b2 = b_s[edgeids]
        c1, c2 = c_s[edgeids]
        d1, d2 = d_s[edgeids]
        x1, x2 = xvs[edgeids]
        y1, y2 = yvs[edgeids]
        z1, z2 = zvs[edgeids]

        L1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        ex = -3*L1*(Em1*(b1*c1*c2*uzz - b1*c1*d2*uzy - b1*c2*d1*uyz + b1*d1*d2*uyy - b2*c1**2*uzz + b2*c1*d1*uyz + b2*c1*d1*uzy - b2*d1**2*uyy + c1**2*d2*uzx - c1*c2*d1*uzx - c1*d1*d2*uyx + c2*d1**2*uyx) + Em2*(b1*c2**2*uzz - b1*c2*d2*uyz - b1*c2*d2*uzy + b1*d2**2*uyy - b2*c1*c2*uzz + b2*c1*d2*uyz + b2*c2*d1*uzy - b2*d1*d2*uyy + c1*c2*d2*uzx - c1*d2**2*uyx - c2**2*d1*uzx + c2*d1*d2*uyx))
        ey = 3*L1*(Em1*(b1**2*c2*uzz - b1**2*d2*uzy - b1*b2*c1*uzz + b1*b2*d1*uzy + b1*c1*d2*uzx - b1*c2*d1*uxz - b1*c2*d1*uzx + b1*d1*d2*uxy + b2*c1*d1*uxz - b2*d1**2*uxy - c1*d1*d2*uxx + c2*d1**2*uxx) + Em2*(b1*b2*c2*uzz - b1*b2*d2*uzy - b1*c2*d2*uxz + b1*d2**2*uxy - b2**2*c1*uzz + b2**2*d1*uzy + b2*c1*d2*uxz + b2*c1*d2*uzx - b2*c2*d1*uzx - b2*d1*d2*uxy - c1*d2**2*uxx + c2*d1*d2*uxx))
        ez = -3*L1*(Em1*(b1**2*c2*uyz - b1**2*d2*uyy - b1*b2*c1*uyz + b1*b2*d1*uyy - b1*c1*c2*uxz + b1*c1*d2*uxy + b1*c1*d2*uyx - b1*c2*d1*uyx + b2*c1**2*uxz - b2*c1*d1*uxy - c1**2*d2*uxx + c1*c2*d1*uxx) + Em2*(b1*b2*c2*uyz - b1*b2*d2*uyy - b1*c2**2*uxz + b1*c2*d2*uxy - b2**2*c1*uyz + b2**2*d1*uyy + b2*c1*c2*uxz + b2*c1*d2*uyx - b2*c2*d1*uxy - b2*c2*d1*uyx - c1*c2*d2*uxx + c2**2*d1*uxx))
        
        Exl += ex*V1
        Eyl += ey*V1
        Ezl += ez*V1
    
    for ie in range(4):
        Em1, Em2 = Ef1s[ie], Ef2s[ie]
        triids = l_tri_ids[:, ie]
        a1, a2, a3 = a_s[triids]
        b1, b2, b3 = b_s[triids]
        c1, c2, c3 = c_s[triids]
        d1, d2, d3 = d_s[triids]

        x1, x2, x3 = xvs[l_tri_ids[:, ie]]
        y1, y2, y3 = yvs[l_tri_ids[:, ie]]
        z1, z2, z3 = zvs[l_tri_ids[:, ie]]

        L1 = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
        L2 = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        
        ex = L1*(Em1*L1*(3*c2*uzx*(c1*d3 - c3*d1) + 3*c2*uzz*(b1*c3 - b3*c1) - 3*d2*uyx*(c1*d3 - c3*d1) + 3*d2*uyy*(b1*d3 - b3*d1) - uyz*(-b2*(c1*d3 - c3*d1) + c2*(b1*d3 - b3*d1) + 2*d2*(b1*c3 - b3*c1)) - uzy*(b2*(c1*d3 - c3*d1) + 2*c2*(b1*d3 - b3*d1) + d2*(b1*c3 - b3*c1))) - Em2*L2*(3*c3*uzx*(c1*d2 - c2*d1) + 3*c3*uzz*(b1*c2 - b2*c1) - 3*d3*uyx*(c1*d2 - c2*d1) + 3*d3*uyy*(b1*d2 - b2*d1) - uyz*(-b3*(c1*d2 - c2*d1) + c3*(b1*d2 - b2*d1) + 2*d3*(b1*c2 - b2*c1)) - uzy*(b3*(c1*d2 - c2*d1) + 2*c3*(b1*d2 - b2*d1) + d3*(b1*c2 - b2*c1))))
        ey = L1*(Em1*L1*(3*b2*uzy*(b1*d3 - b3*d1) - 3*b2*uzz*(b1*c3 - b3*c1) + 3*d2*uxx*(c1*d3 - c3*d1) - 3*d2*uxy*(b1*d3 - b3*d1) + uxz*(-b2*(c1*d3 - c3*d1) + c2*(b1*d3 - b3*d1) + 2*d2*(b1*c3 - b3*c1)) - uzx*(2*b2*(c1*d3 - c3*d1) + c2*(b1*d3 - b3*d1) - d2*(b1*c3 - b3*c1))) - Em2*L2*(3*b3*uzy*(b1*d2 - b2*d1) - 3*b3*uzz*(b1*c2 - b2*c1) + 3*d3*uxx*(c1*d2 - c2*d1) - 3*d3*uxy*(b1*d2 - b2*d1) + uxz*(-b3*(c1*d2 - c2*d1) + c3*(b1*d2 - b2*d1) + 2*d3*(b1*c2 - b2*c1)) - uzx*(2*b3*(c1*d2 - c2*d1) + c3*(b1*d2 - b2*d1) - d3*(b1*c2 - b2*c1))))
        ez = -L1*(Em1*L1*(3*b2*uyy*(b1*d3 - b3*d1) - 3*b2*uyz*(b1*c3 - b3*c1) + 3*c2*uxx*(c1*d3 - c3*d1) + 3*c2*uxz*(b1*c3 - b3*c1) - uxy*(b2*(c1*d3 - c3*d1) + 2*c2*(b1*d3 - b3*d1) + d2*(b1*c3 - b3*c1)) - uyx*(2*b2*(c1*d3 - c3*d1) + c2*(b1*d3 - b3*d1) - d2*(b1*c3 - b3*c1))) - Em2*L2*(3*b3*uyy*(b1*d2 - b2*d1) - 3*b3*uyz*(b1*c2 - b2*c1) + 3*c3*uxx*(c1*d2 - c2*d1) + 3*c3*uxz*(b1*c2 - b2*c1) - uxy*(b3*(c1*d2 - c2*d1) + 2*c3*(b1*d2 - b2*d1) + d3*(b1*c2 - b2*c1)) - uyx*(2*b3*(c1*d2 - c2*d1) + c3*(b1*d2 - b2*d1) - d3*(b1*c2 - b2*c1)))) 
        
        Exl += ex*V1
        Eyl += ey*V1
        Ezl += ez*V1

    out = np.zeros((3,), dtype=np.complex128)
    out[0] = Exl
    out[1] = Eyl
    out[2] = Ezl
    return out

@njit(types.Tuple((f8[:], f8[:]))(f8[:,:], i8[:,:], i8[:,:], i8[:,:], f8[:,:],
                                  c16[:], f8[:], f8[:], i8[:,:], i8[:,:],
                                  f8[:,:], i8[:,:], i8[:,:], c16[:], c16[:], f8), cache=True, nogil=True)
def compute_error_single(nodes, tets, tris, edges, centers, 
                         Efield, 
                         edge_lengths,
                         areas,
                         tet_to_edge, 
                         tet_to_tri, 
                         tri_centers, 
                         tri_to_tet,
                         tet_to_field, 
                         er, 
                         ur,
                         k0,) -> np.ndarray:


    # UNPACK DATA
    ntet = tets.shape[1]
    nedges = edges.shape[1]
    
    
    # INIT POSTERIORI ERROR ESTIMATE
    error = np.zeros((ntet,), dtype=np.float64)
    max_elem_size = np.zeros((ntet,), dtype=np.float64)
    
    hks = np.zeros((ntet,), dtype=np.float64)
    hfs = np.zeros((4,ntet), dtype=np.float64)
    face_error1 = np.zeros((4,3,ntet), dtype=np.complex128)
    face_error2 = np.zeros((4,3,ntet), dtype=np.complex128)
    

    # Compute Error estimate
    for itet in range(ntet):
        uinv = (1/ur[itet])*np.eye(3)
        ermat = er[itet]*np.eye(3)
        
        vertices = nodes[:,tets[:, itet]]
        
        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_field[:6, itet]]
        g_tri_ids = tris[:, tet_to_field[6:10, itet]-nedges]

        l_edge_ids = local_mapping(g_node_ids, g_edge_ids)
        l_tri_ids = local_mapping(g_node_ids, g_tri_ids)

        coords = centers[:,itet]
        Ef = Efield[tet_to_field[:,itet]]
        Rv1 = -compute_curl_curl(coords, vertices, Ef, l_edge_ids, l_tri_ids, uinv)
        Rv2 = k0**2*(ermat @ compute_field(coords, vertices, Ef, l_edge_ids, l_tri_ids))
        
        triids = tet_to_tri[:,itet]
        facecoords = tri_centers[:, triids]
        
        Volume = compute_volume(vertices[0,:], vertices[1,:], vertices[2,:])
        hks[itet] = Volume**(1/3)
        Rf = matmul(uinv, compute_curl(facecoords, vertices, Ef, l_edge_ids, l_tri_ids))
        tetc = centers[:,itet].flatten()
        
        max_elem_size[itet] = (Volume*12/np.sqrt(2))**(1/3)#(np.max(edge_lengths[tet_to_edge[:,itet]]) + np.min(edge_lengths[tet_to_edge[:,itet]]))/2
        
        for iface in range(4):
            i1, i2, i3 = tris[:, triids[iface]]
            normal = outward_normal(nodes[:,i1], nodes[:,i2], nodes[:,i3], tetc).astype(np.complex128)
            
            adj_tets = [int(tri_to_tet[j,triids[iface]]) for j in range(2)]
            adj_tets = [num for num in adj_tets if num not in (itet, -1234)]
            
            if len(adj_tets) == 0:
                continue
            area = areas[triids[iface]]
            
            hfs[iface,itet] = area**0.5
            
            itet_adj = adj_tets[0]
            iface_adj = np.argwhere(tet_to_tri[:,itet_adj]==triids[iface])[0][0]
            
            face_error2[iface_adj, :, itet_adj] = -area*cross_c(normal, uinv @ Rf[:, iface])
            face_error1[iface, :, itet] = area*cross_c(normal, uinv @ Rf[:,iface])
        

        error[itet] = np.linalg.norm(np.abs(Volume*(Rv1 + Rv2)))**2
    
    fdiff = np.abs(face_error1 - face_error2)
    fnorm = fdiff[:,0,:]**2 + fdiff[:,1,:]**2 + fdiff[:,2,:]**2
    ferror = np.sum(fnorm*hfs, axis=0)
    error = hks**2*error + 0.5*ferror

    return error, max_elem_size

def compute_error_estimate(field: MWField) -> np.ndarray:
    mesh = field.mesh

    nodes = mesh.nodes
    tris = mesh.tris
    tets = mesh.tets
    edges = mesh.edges
    centers = mesh.centers
    
    As = mesh.areas
    tet_to_edge = mesh.tet_to_edge
    tet_to_tri = mesh.tet_to_tri
    tri_centers = mesh.tri_centers
    tri_to_tet = mesh.tri_to_tet
    tet_to_field = field.basis.tet_to_field
    er = field._der
    ur = field._dur
    Ls = mesh.edge_lengths
    
    errors = []
    for key in field._fields.keys():
        excitation = field._fields[key]
        
        error, sizes = compute_error_single(nodes, tets, tris, edges,
                             centers, excitation, Ls, As, 
                             tet_to_edge, tet_to_tri, tri_centers,
                             tri_to_tet, tet_to_field, er, ur, field.k0)
        
        errors.append(error)
    
    error = np.max(np.array(errors), axis=0)
    return error, sizes