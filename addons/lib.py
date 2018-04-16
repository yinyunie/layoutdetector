import numpy as np
from pyflann import *
import sys
sys.path.append('/home/ynie1/Library/liblinear-2.20/python')
from liblinearutil import *

def gen_line_fromGC(gc_map, gc_label1, gc_label2):

    X0 = np.argwhere(gc_map == gc_label1)
    y0 = np.ones(len(X0))

    X1 = np.argwhere(gc_map == gc_label2)
    y1 = - np.ones(len(X1))

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    prob = problem(y, X)
    param = parameter('-s 3 -c 1 -B 1')
    model = train(prob, param)
    w, b = model.get_decfun()

    if w[0] != 0.:
        x1 = 0.
        x2 = gc_map.shape[1] - 1.
        y1 = -w[1] / w[0] * x1 - b / w[0]
        y2 = -w[1] / w[0] * x2 - b / w[0]

    else:
        x1 = -b / w[1]
        x2 = -b / w[1]
        y1 = 0.
        y2 = gc_map.shape[0] - 1.

    return [x1, y1, x2, y2]

def gen_line_clusters(vps, K, lines):
    # decide which vps this line belongs to
    line_clusters = [[] for i in range(3)]

    vp2D = [[] for i in range(3)]
    for i in xrange(3):
        vp2D[i] = np.array([vps[i][0] * K[0, 0] / vps[i][2] + K[0, 2], vps[i][1] * K[0, 0] / vps[i][2] + K[1, 2]])

    for i in range(len(lines)):

        x1 = lines[i][0]
        y1 = lines[i][1]
        x2 = lines[i][2]
        y2 = lines[i][3]

        pt1 = np.array([x1, y1])
        pt2 = np.array([x2, y2])
        ptm = (pt1 + pt2) / 2.

        vc = (pt1 - pt2) / (np.linalg.norm(pt1 - pt2))

        minAngle = 1000.
        bestIdx = None

        for j in range(3):
            vp2d_c = vp2D[j] - ptm
            vp2d_c = vp2d_c / np.linalg.norm(vp2d_c)

            dotValue = np.dot(vp2d_c, vc)

            if dotValue > 1.0:
                dotValue = 1.0
            if dotValue < -1.0:
                dotValue = -1.0

            angle = np.arccos(dotValue)
            angle = min(np.pi - angle, angle)

            if angle < minAngle:
                minAngle = angle
                bestIdx = j

        line_clusters[bestIdx].append(i)

    return line_clusters

def gen_lineproposals(lines, vps, K, mask_map, gc_map, clusters, line_labels):

    # decide which kinds of lines we should generate
    gc_labels = np.unique(gc_map)

    # initiate line info generated from gc
    lines_gc = []
    line_count = 0
    line_gc_labels = []

    # firsy we use logistic regression to generate essential lines
    if 5 in gc_labels:
        # generate line between 2 and 5
        nline = gen_line_fromGC(gc_map, 2, 5)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 25])
        line_count += 1

    if 6 in gc_labels:
        # generate line between 2 and 6
        nline = gen_line_fromGC(gc_map, 2, 6)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 26])
        line_count += 1

    if 3 in gc_labels:
        # generate line between 2 and 3
        nline = gen_line_fromGC(gc_map, 2, 3)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 23])
        line_count += 1

    if 4 in gc_labels:
        # generate line between 2 and 4
        nline = gen_line_fromGC(gc_map, 2, 4)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 24])
        line_count += 1

    if 3 in gc_labels and 5 in gc_labels:
        # generate line between 3 and 5
        nline = gen_line_fromGC(gc_map, 3, 5)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 35])
        line_count += 1

    if 3 in gc_labels and 6 in gc_labels:
        # generate line between 3 and 6
        nline = gen_line_fromGC(gc_map, 3, 6)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 36])
        line_count += 1

    if 4 in gc_labels and 5 in gc_labels:
        # generate line between 4 and 5
        nline = gen_line_fromGC(gc_map, 4, 5)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 45])
        line_count += 1

    if 4 in gc_labels and 6 in gc_labels:
        # generate line between 4 and 6
        nline = gen_line_fromGC(gc_map, 4, 6)

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 46])
        line_count += 1

    line_gc_clusters = gen_line_clusters(vps, K, lines_gc)

    line_gc_labels = np.array(line_gc_labels)

    return lines_gc, line_gc_labels, line_gc_clusters

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

def reshape_map(map, height):
    # downsampling for efficiency
    h, w = map.shape
    step = h / height
    hlist = range(0, h, step)
    wlist = range(0, w, step)
    map_new = np.zeros([len(hlist), len(wlist)], dtype=np.uint8)
    ix = 0
    for hid in hlist:
        iy = 0
        for wid in wlist:
            map_new[ix, iy] = map[hid, wid]
            iy = iy + 1
        ix = ix + 1

    return map_new

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
    # downsampling for efficiency
    height = 100
    size0 = gc_map.shape[0]
    gc_map = reshape_map(gc_map, height)
    im_scale = size0 / np.double(gc_map.shape[0])

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
        pt1 = np.array([lines[line_id][1], lines[line_id][0]])/im_scale
        pt2 = np.array([lines[line_id][3], lines[line_id][2]])/im_scale
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