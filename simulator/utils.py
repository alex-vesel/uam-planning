import numpy as np
import scipy.spatial.distance as dist

def kms_to_ms(kms):
    return kms * 1000

def ms_to_kms(ms):
    return ms / 1000

def km_to_m(km):
    return km * 1000

def m_to_km(m):
    return m / 1000

def per_hr_to_per_s(per_hr):
    return per_hr / 3600

def per_s_to_per_hr(per_s):
    return per_s * 3600

def pairwise_distance(list_A, list_B):
    list_A_pts = np.array([[map_el.x, map_el.y] for map_el in list_A])
    list_B_pts = np.array([[map_el.x, map_el.y] for map_el in list_B])
    return dist.cdist(list_A_pts, list_B_pts)

def pairwise_heading(list_A, list_B):
    headings = np.zeros((len(list_A), len(list_B)))
    for i, map_el_A in enumerate(list_A):
        for j, map_el_B in enumerate(list_B):
            headings[i, j] = np.arctan2(map_el_B.y - map_el_A.y, map_el_B.x - map_el_A.x)
    return headings
    # import IPython; IPython.embed(); exit(0)
    list_A_pts = np.array([[map_el.x, map_el.y] for map_el in list_A])
    list_B_pts = np.array([[map_el.x, map_el.y] for map_el in list_B])
    costheta = 1 - dist.cdist(list_A_pts, list_B_pts, 'cosine')
    return np.arccos(costheta)