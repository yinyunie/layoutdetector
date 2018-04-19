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
    param = parameter('-s 2 -c 1 -B 1')
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

def gen_line_clusters(vps2D, lines):
    # decide which vps this line belongs to
    line_clusters = [[] for i in range(3)]

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
            vp2d_c = vps2D[j] - ptm
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

def gen_lines_fromGC(gc_map, vps2D, ifscaleimg):

    if ifscaleimg:
        # downsampling for efficiency, comment it if not use
        height = 100
        size0 = gc_map.shape[0]
        gc_map = reshape_map(gc_map, height)
        im_scale = size0 / np.double(gc_map.shape[0])
    else:
        im_scale = 1.

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
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 25])
        line_count += 1

    if 6 in gc_labels:
        # generate line between 2 and 6
        nline = gen_line_fromGC(gc_map, 2, 6)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 26])
        line_count += 1

    if 3 in gc_labels:
        # generate line between 2 and 3
        nline = gen_line_fromGC(gc_map, 2, 3)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 23])
        line_count += 1

    if 4 in gc_labels:
        # generate line between 2 and 4
        nline = gen_line_fromGC(gc_map, 2, 4)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 24])
        line_count += 1

    if 3 in gc_labels and 5 in gc_labels:
        # generate line between 3 and 5
        nline = gen_line_fromGC(gc_map, 3, 5)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 35])
        line_count += 1

    if 3 in gc_labels and 6 in gc_labels:
        # generate line between 3 and 6
        nline = gen_line_fromGC(gc_map, 3, 6)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 36])
        line_count += 1

    if 4 in gc_labels and 5 in gc_labels:
        # generate line between 4 and 5
        nline = gen_line_fromGC(gc_map, 4, 5)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 45])
        line_count += 1

    if 4 in gc_labels and 6 in gc_labels:
        # generate line between 4 and 6
        nline = gen_line_fromGC(gc_map, 4, 6)
        nline = [corr * im_scale for corr in nline]

        lines_gc.append(nline)
        line_gc_labels.append([line_count, 46])
        line_count += 1

    line_gc_clusters = gen_line_clusters(vps2D, lines_gc)

    line_gc_labels = np.array(line_gc_labels)

    return lines_gc, line_gc_labels, line_gc_clusters

def ruleout(table_gclabel_vp, line_labels, clusters):

    new_line_labels = []
    new_clusters = [[] for i in range(3)]

    for line_id, gc_label in line_labels:

        vpid1 = [line_id in cluster for cluster in clusters].index(True)

        vpid2 = table_gclabel_vp[table_gclabel_vp[:, 0] == gc_label, 1][0]

        if vpid1 == vpid2:
            new_line_labels.append([line_id, gc_label])
            new_clusters[vpid1].append(line_id)

    new_line_labels = np.array(new_line_labels)

    return new_line_labels, new_clusters


def comb_to_set(lines_new, line_new_labels, line_new_clusters, lines, line_labels, clusters, table_gclabel_vp):

    # combines all line information to the a whole set

    # filter lines which have a wrong corresponding between gc_label and vp
    line_labels, clusters = ruleout(table_gclabel_vp, line_labels, clusters)

    # initialisation
    lines_set = []
    line_labels_set = []
    clusters_set = [[] for i in range(3)]
    id_mapping = []

    line_count = 0
    for line_id, line_label in line_labels:

        lines_set.append(lines[line_id])
        line_labels_set.append([line_count, line_label])
        id_mapping.append([line_id, line_count])
        line_count += 1

    line_labels_set = np.array(line_labels_set)
    id_mapping = np.array(id_mapping)

    num_lines = len(lines_set)
    lines_set = lines_set + lines_new

    line_new_labels[:, 0] = line_new_labels[:, 0] + num_lines

    line_labels_set = np.vstack([line_labels_set, line_new_labels])

    for cluster_id in range(len(clusters)):

        old_cluster_ids = [id_mapping[:, 1][id_mapping[:, 0] == clusters[cluster_id][i]][0] for i in range(len(clusters[cluster_id]))]

        new_cluster_ids = [id + num_lines for id in line_new_clusters[cluster_id]]

        clusters_set[cluster_id] = old_cluster_ids + new_cluster_ids

    return lines_set, line_labels_set, clusters_set

def infer_line(plabel1, pair_list2, lines_set, line_labels_set, clusters_set, vps2D, table_gclabel_vp, im_width, mask_map):

    if plabel1 > 2:
        label_to_gen = 2 * 10 + plabel1
    else:
        label_to_gen = plabel1 * 10 + 2

    vpid = table_gclabel_vp[table_gclabel_vp[:, 0] == label_to_gen, 1][0]

    vp = vps2D[vpid]

    new_lines_set = lines_set[:]
    new_line_labels_set = line_labels_set[:]
    new_clusters_set = clusters_set[:]
    count = len(lines_set)

    for plabel2 in pair_list2:

        if plabel1 > plabel2:
            label1 = plabel2 * 10 + plabel1
        else:
            label1 = plabel1 * 10 + plabel2

        if plabel2 > 2:
            label2 = 2 * 10 + plabel2
        else:
            label2 = plabel2 * 10 + 2

        lineID_set1 = new_line_labels_set[new_line_labels_set[:, 1] == label1, 0]
        lineID_set2 = new_line_labels_set[new_line_labels_set[:, 1] == label2, 0]

        for id1 in lineID_set1:
            for id2 in lineID_set2:
                # search the vp corresponding to the corner
                vpid1 = [id1 in cluster for cluster in new_clusters_set].index(True)
                vpid2 = [id2 in cluster for cluster in new_clusters_set].index(True)

                if vpid1 != table_gclabel_vp[table_gclabel_vp[:, 0] == label1, 1][0]:
                    continue
                if vpid2 != table_gclabel_vp[table_gclabel_vp[:, 0] == label2, 1][0]:
                    continue

                p1 = np.array([new_lines_set[id1][0], new_lines_set[id1][1], 1.0])
                p2 = np.array([new_lines_set[id1][2], new_lines_set[id1][3], 1.0])
                # line coefficient via p1 and p2
                line1 = np.cross(p1, p2)

                p1 = np.array([new_lines_set[id2][0], new_lines_set[id2][1], 1.0])
                p2 = np.array([new_lines_set[id2][2], new_lines_set[id2][3], 1.0])
                line2 = np.cross(p1, p2)

                corner = np.cross(line1, line2)
                corner = corner / corner[2]

                if mask_map[int(corner[1]), int(corner[0])] == 0:
                    continue

                if (corner[:2] - vp)[0] < 0:
                    lamb = - corner[0]/(corner[0]-vp[0])
                    endpt = corner[:2] + lamb * (corner[:2] - vp)

                else:
                    lamb = im_width - corner[0] / (corner[0] - vp[0])
                    endpt = corner[:2] + lamb * (corner[:2] - vp)

                newline = [endpt[0], endpt[1], corner[0], corner[1]]

                new_lines_set.append(newline)

                new_line_labels_set = np.append(new_line_labels_set, [[count, label_to_gen]], axis=0)

                new_clusters_set[vpid].append(count)

                count += 1

    return new_lines_set, new_line_labels_set, new_clusters_set

def infer_lines(lines_set, line_labels_set, clusters_set, vps2D, gc_labels, table_gclabel_vp, mask_map):

    __, im_width = mask_map.shape

    labels_to_gen = np.setdiff1d(gc_labels, [2])
    pair_list1 = np.intersect1d(labels_to_gen, [3, 4])
    pair_list2 = np.intersect1d(labels_to_gen, [5, 6])

    for label in labels_to_gen:

        if label in pair_list1:
            lines_set, line_labels_set, clusters_set = infer_line(label, pair_list2, lines_set, line_labels_set, clusters_set, vps2D, table_gclabel_vp, im_width, mask_map)
        else:
            lines_set, line_labels_set, clusters_set = infer_line(label, pair_list1, lines_set, line_labels_set, clusters_set, vps2D, table_gclabel_vp, im_width, mask_map)

    return lines_set, line_labels_set, clusters_set

def gen_lineproposals(lines, vps2D, gc_map, clusters, line_labels, mask_map):

    gc_labels = np.unique(gc_map)

    # generate lines from gc content
    ifscaleimg = True
    lines_gc, line_gc_labels, line_gc_clusters = gen_lines_fromGC(gc_map, vps2D, ifscaleimg)

    # correspond gc_label to vanishing point id
    table_gclabel_vp = []
    for line_id, label in line_gc_labels:
        vpid = [line_id in cluster for cluster in line_gc_clusters].index(True)
        table_gclabel_vp.append([label, vpid])

    table_gclabel_vp = np.array(table_gclabel_vp)

    # combine to lines set
    lines_set, line_labels_set, clusters_set = comb_to_set(lines_gc, line_gc_labels, line_gc_clusters, lines, line_labels, clusters, table_gclabel_vp)

    # generate lines from inference
    new_lines_set, new_line_labels_set, new_clusters_set = infer_lines(lines_set, line_labels_set, clusters_set, vps2D, gc_labels, table_gclabel_vp, mask_map)

    return new_lines_set, new_line_labels_set, new_clusters_set, table_gclabel_vp

def gen_proposals(lines_set, clusters_set, vps2D, vp2D, plabels, lineIDs_gclabel, edge_map):

    height, width = edge_map.shape

    label_set1 = np.intersect1d(plabels, [3, 4])
    label_set2 = np.intersect1d(plabels, [5, 6])

    proposals = []

    line_combs = np.stack(np.meshgrid(*lineIDs_gclabel), axis=len(plabels)).reshape(-1, len(plabels))

    for comb_id in xrange(len(line_combs)):

        new_lines = []
        endpt_dict = {}

        for plabel1 in label_set1:
            line1_id = line_combs[comb_id][plabels.index(plabel1)]
            endpt_dict[line1_id] = []

            p1 = np.array([lines_set[line1_id][0], lines_set[line1_id][1], 1.0])
            p2 = np.array([lines_set[line1_id][2], lines_set[line1_id][3], 1.0])
            line1 = np.cross(p1, p2)

            for plabel2 in label_set2:
                line2_id = line_combs[comb_id][plabels.index(plabel2)]
                endpt_dict[line2_id] = []

                p1 = np.array([lines_set[line2_id][0], lines_set[line2_id][1], 1.0])
                p2 = np.array([lines_set[line2_id][2], lines_set[line2_id][3], 1.0])
                line2 = np.cross(p1, p2)

                corner = np.cross(line1, line2)
                corner = corner / corner[2]

                if (corner[:2] - vp2D)[0] < 0:
                    lamb = - corner[0]/(corner[0]-vp2D[0])
                else:
                    lamb = width - corner[0] / (corner[0] - vp2D[0])

                endpt = corner[:2] + lamb * (corner[:2] - vp2D)
                newline = [endpt[0], endpt[1], corner[0], corner[1]]

                endpt_dict[line1_id].append(corner[:2])
                endpt_dict[line2_id].append(corner[:2])

                new_lines.append(newline)

        for line_id in line_combs[comb_id]:
            corners = endpt_dict[line_id]

            if len(corners) > 1:
                new_lines.append([coor for pnt in corners for coor in pnt])
            else:
                corner = corners[0]

                pt1 = np.array([lines_set[line_id][0], lines_set[line_id][1]])
                pt2 = np.array([lines_set[line_id][2], lines_set[line_id][3]])

                vpid = [line_id in cluster for cluster in clusters_set].index(True)
                vp_t = vps2D[vpid]

                if (corner - vp_t).dot(pt2 - pt1) > 0.:
                    if (pt2 - pt1)[0] > 0.:
                        lamb = (width - corner[0]) / (pt2[0] - pt1[0])
                    else:
                        lamb = - corner[0] / (pt2[0] - pt1[0])
                    endpnt = corner + lamb * (pt2 - pt1)
                else:
                    if (pt1 - pt2)[0] > 0.:
                        lamb = (width - corner[0]) / (pt1[0] - pt2[0])
                    else:
                        lamb = - corner[0] / (pt1[0] - pt2[0])
                    endpnt = corner + lamb * (pt1 - pt2)

                new_lines.append([endpnt[0], endpnt[1], corner[0], corner[1]])

        proposals.append(new_lines)

    return proposals

def gen_layoutproposals(lines_set, line_labels_set, clusters_set, table_gclabel_vp, vps2D, gc_labels, edge_map):

    # find out the third vp
    for gc_label, vp_id in table_gclabel_vp:
        if '2' not in str(gc_label):
            vp2D = vps2D[vp_id]
            break

    plabels = list(np.setdiff1d(gc_labels, [2]))

    lineIDs_gclabel = []

    for plabel in plabels:

        if 2 < plabel:
            label = 2 * 10 + plabel
        else:
            label = plabel * 10 + 2

        lineIDs_gclabel.append(line_labels_set[line_labels_set[:,1] == label, 0])

    proposals = gen_proposals(lines_set, clusters_set, vps2D, vp2D, plabels, lineIDs_gclabel, edge_map)

    return proposals

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