import numpy as np
from pyflann import *

def processGC(gc_map):

    # restric gc_map to limited cases
    gc_map_new = np.copy(gc_map)
    gc_labels = np.unique(gc_map)
    if 2 not in gc_labels:
        if 4 in gc_labels:
            gc_map_new[gc_map==4] = 2
            return gc_map_new
        elif 3 in gc_labels:
            gc_map_new[gc_map == 3] = 2
            return  gc_map_new
        else:
            return None

def decide_linelabel(lines, clusters, gc_map):
    '''GC map definition:
    # 1. background.
    # 2. frontal wall
    # 3. left wall
    # 4. right wall
    # 5. floor
    # 6. ceiling'''
    '''Output definition:
    linelabels: (9 in total)
    '23': between frontal wall (2) and left wall (3)
    '24': between frontal wall (2) and right wall (4)
    '25': between frontal wall (2) and floor (5)
    '26': between frontal wall (2) and ceiling (6)
    '35': between left wall (2) and floor (5)
    '36': between left wall (2) and ceiling (6)
    '45': between right wall (2) and floor (5)
    '46': between right wall (2) and ceiling (6)'''
    # build dataset for each gc label
    gc_labels = np.unique(gc_map)
    dataset = []
    for label in gc_labels:
        dataset.append(np.double(np.argwhere(gc_map == label)))

    flann = FLANN()
    linelabels = []

    line_list = [line_id for cluster in clusters for line_id in cluster]

    for line_id in line_list:
        # pay attention to the coordinate system change
        pt1 = np.array([lines[line_id][1], lines[line_id][0]])
        pt2 = np.array([lines[line_id][3], lines[line_id][2]])
        testset = np.array([pt1, pt2])

        dist_list = []
        for data in dataset:
            __, dists = flann.nn(data, testset, 1)
            dist_list.append(np.max(dists))

        linelabel1 = gc_labels[np.argsort(dist_list)[0]]
        linelabel2 = gc_labels[np.argsort(dist_list)[1]]

        if linelabel1 > linelabel2:
            linelabel1, linelabel2 = linelabel2, linelabel1
        linelabel = linelabel1 * 10 + linelabel2

        linelabels.append([line_id, linelabel])

    linelabels = np.array(linelabels)
    return linelabels

def pnts_gen(pt1, pt2, num_checks):
    pts = []
    for i in range(num_checks):
        pt = (i * (1./(num_checks-1))) * (pt2 - pt1) + pt1
        pts.append(pt)
    return pts

def line_filter(lines, clusters, mask):
    num_checks = 4
    new_clusters = [[] for i in range(len(clusters))]

    for cluster_id in range(len(clusters)):
        for line_id in clusters[cluster_id]:
            pt1 = np.array([lines[line_id][0], lines[line_id][1]])
            pt2 = np.array([lines[line_id][2], lines[line_id][3]])

            pnts = np.array(pnts_gen(pt1, pt2, num_checks), dtype=np.uint)

            ifpick = 1
            for pnt in pnts:
                if mask[pnt[1], pnt[0]] == 0:
                    ifpick = 0
                    break
            if ifpick == 1:
                new_clusters[cluster_id].append(line_id)

    return new_clusters