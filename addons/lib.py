import numpy as np
from pyflann import *
import sys
import cv2
sys.path.append('/home/ynie1/Library/liblinear-2.20/python')
from liblinearutil import *
from addons.predef import gc_def, gc_neighbours, neighbour_cluster_set

def get_mask(edge_map, threshold, dilate_size):
    # Binarise the edge map with the threhold value
    # Then dialte the result with dilate_size pixels.

    mask_map = (edge_map - np.min(edge_map))/(np.max(edge_map) - np.min(edge_map))

    mask_map[mask_map < threshold] = 0.
    mask_map[mask_map >= threshold] = 1.

    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    mask_map = cv2.dilate(mask_map, kernel, iterations=1)
    mask_map = np.uint8(mask_map)

    return mask_map

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

def gen_line_clusters(gc_labels, cnt, vps2D, line_gc_labels):
    # decide which vps this line belongs to
    line_clusters = [[] for i in range(3)]
    vps2D = np.array(vps2D)

    vps_ids = range(3)
    vps_imcnt = vps2D - cnt

    vp_cid1 = np.argmax(np.abs(vps_imcnt[:, 1]))

    vps_ids = np.setdiff1d(vps_ids, vp_cid1)
    vps_imcnt = vps_imcnt[vps_ids]

    if gc_def['right_wallID'] in gc_labels:
        vp_cid2 = vps_ids[np.argmin(np.linalg.norm(vps_imcnt, axis=1))]
    else:
        vp_cid2 = vps_ids[np.argmax(vps_imcnt[:,0])]

    vp_cid3 = np.setdiff1d(range(3), [vp_cid1, vp_cid2])[0]

    for line_id, gc_label in line_gc_labels:
        if gc_label in neighbour_cluster_set[0]:
            line_clusters[vp_cid1].append(line_id)
        if gc_label in neighbour_cluster_set[2]:
            line_clusters[vp_cid2].append(line_id)
        if gc_label in neighbour_cluster_set[1]:
            line_clusters[vp_cid3].append(line_id)

    return line_clusters

def gen_lines_fromGC(gc_map, vps2D, ifscaleimg):

    cnt = np.array([gc_map.shape[1], gc_map.shape[0]])/2.

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

    # firsy we use SVM to generate essential lines
    for plabel1, plabel2 in gc_neighbours:
        if (plabel1 in gc_labels) and (plabel2 in gc_labels):
            # generate line between plabel1 and plabel2
            nline = gen_line_fromGC(gc_map, plabel1, plabel2)
            nline = [corr * im_scale for corr in nline]

            lines_gc.append(nline)
            if plabel1 < plabel2:
                label = plabel1 * 10 + plabel2
            else:
                label = plabel2 * 10 + plabel1

            line_gc_labels.append([line_count, label])
            line_count += 1

    line_gc_labels = np.array(line_gc_labels)

    line_gc_clusters = gen_line_clusters(gc_labels, cnt, vps2D, line_gc_labels)

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


def comb_to_set(lines_new, line_new_labels, lines, line_labels, clusters, gc_labels, table_gclabel_vp):

    # combines all line information to the a whole set
    plabels = np.setdiff1d(gc_labels, [2])

    # initialisation
    lines_set = lines[:]
    line_labels_set = line_labels[:]
    clusters_set = clusters[:]
    line_count = len(lines)

    for plabel in plabels:
        if plabel > gc_def['frontal_wallID']:
            label1 = gc_def['frontal_wallID'] * 10 + plabel
        else:
            label1 = plabel * 10 + gc_def['frontal_wallID']

        if label1 not in line_labels[:,1]:
            line_id = line_new_labels[line_new_labels[:, 1] == label1, 0][0]

            lines_set.append(lines_new[line_id])
            line_labels_set = np.append(line_labels_set, [[line_count, label1]], axis=0)
            clusters_set[table_gclabel_vp[table_gclabel_vp[:, 0] == label1, 1][0]].append(line_count)
            line_count = line_count + 1

    return lines_set, line_labels_set, clusters_set

def infer_line(plabel1, pair_list2, lines_set, line_labels_set, clusters_set, vps2D, table_gclabel_vp, im_width, mask_map):

    if plabel1 > gc_def['frontal_wallID']:
        label_to_gen = gc_def['frontal_wallID'] * 10 + plabel1
    else:
        label_to_gen = plabel1 * 10 + gc_def['frontal_wallID']

    vpid = table_gclabel_vp[table_gclabel_vp[:, 0] == label_to_gen, 1][0]

    vp = vps2D[vpid]

    new_lines_set = []
    new_line_labels_set = []
    new_clusters_set = [[] for i in range(3)]
    count = 0

    for plabel2 in pair_list2:

        if plabel1 > plabel2:
            label1 = plabel2 * 10 + plabel1
        else:
            label1 = plabel1 * 10 + plabel2

        if plabel2 > gc_def['frontal_wallID']:
            label2 = gc_def['frontal_wallID'] * 10 + plabel2
        else:
            label2 = plabel2 * 10 + gc_def['frontal_wallID']

        lineID_set1 = line_labels_set[line_labels_set[:, 1] == label1, 0]
        lineID_set2 = line_labels_set[line_labels_set[:, 1] == label2, 0]

        for id1 in lineID_set1:
            for id2 in lineID_set2:
                # search the vp corresponding to the corner
                vpid1 = [id1 in cluster for cluster in clusters_set].index(True)
                vpid2 = [id2 in cluster for cluster in clusters_set].index(True)

                if vpid1 != table_gclabel_vp[table_gclabel_vp[:, 0] == label1, 1][0]:
                    continue
                if vpid2 != table_gclabel_vp[table_gclabel_vp[:, 0] == label2, 1][0]:
                    continue

                p1 = np.array([lines_set[id1][0], lines_set[id1][1], 1.0])
                p2 = np.array([lines_set[id1][2], lines_set[id1][3], 1.0])
                # line coefficient via p1 and p2
                line1 = np.cross(p1, p2)

                p1 = np.array([lines_set[id2][0], lines_set[id2][1], 1.0])
                p2 = np.array([lines_set[id2][2], lines_set[id2][3], 1.0])
                line2 = np.cross(p1, p2)

                corner = np.cross(line1, line2)
                corner = corner / corner[2]

                if mask_map[int(round(corner[1])), int(round(corner[0]))] == 0:
                    continue

                if (corner[:2] - vp)[0] < 0:
                    lamb = - corner[0]/(corner[0]-vp[0])
                    endpt = corner[:2] + lamb * (corner[:2] - vp)

                else:
                    lamb = im_width - corner[0] / (corner[0] - vp[0])
                    endpt = corner[:2] + lamb * (corner[:2] - vp)

                newline = [endpt[0], endpt[1], corner[0], corner[1]]

                new_lines_set.append(newline)

                new_line_labels_set.append([count, label_to_gen])

                new_clusters_set[vpid].append(count)

                count += 1

    new_line_labels_set = np.array(new_line_labels_set)

    return new_lines_set, new_line_labels_set, new_clusters_set

def infer_lines(lines_set, line_labels_set, clusters_set, vps2D, gc_labels, table_gclabel_vp, mask_map, ifinferExtralines):

    __, im_width = mask_map.shape

    labels_to_gen = np.setdiff1d(gc_labels, [gc_def['backgroundID'], gc_def['frontal_wallID']])
    pair_list1 = np.intersect1d(labels_to_gen, [gc_def['left_wallID'], gc_def['right_wallID']])
    pair_list2 = np.intersect1d(labels_to_gen, [gc_def['floor_ID'], gc_def['ceiling_ID']])

    if not ifinferExtralines:

        new_lines_set = []
        new_line_labels_set = []
        new_clusters_set = [[] for i in range(3)]

        for label in labels_to_gen:

            if label in pair_list1:
                lines, line_labels, clusters = infer_line(label, pair_list2, lines_set, line_labels_set, clusters_set,
                                                          vps2D, table_gclabel_vp, im_width, mask_map)
            else:
                lines, line_labels, clusters = infer_line(label, pair_list1, lines_set, line_labels_set, clusters_set,
                                                          vps2D, table_gclabel_vp, im_width, mask_map)

            if lines:
                new_lines_set, new_line_labels_set, new_clusters_set = unify(lines, line_labels, clusters,
                                                                             new_lines_set, new_line_labels_set,
                                                                             new_clusters_set)

        new_lines_set, new_line_labels_set, new_clusters_set = unify(new_lines_set, new_line_labels_set,
                                                                     new_clusters_set, lines_set, line_labels_set,
                                                                     clusters_set)
    else:

        new_lines_set = lines_set[:]
        new_line_labels_set = line_labels_set[:]
        new_clusters_set = clusters_set[:]

        for label in labels_to_gen:

            if label in pair_list1:
                lines, line_labels, clusters = infer_line(label, pair_list2, new_lines_set, new_line_labels_set,
                                                          new_clusters_set, vps2D, table_gclabel_vp, im_width, mask_map)
            else:
                lines, line_labels, clusters = infer_line(label, pair_list1, new_lines_set, new_line_labels_set,
                                                          new_clusters_set, vps2D, table_gclabel_vp, im_width, mask_map)
            if lines:
                new_lines_set, new_line_labels_set, new_clusters_set = unify(lines, line_labels, clusters,
                                                                             new_lines_set, new_line_labels_set,
                                                                             new_clusters_set)

    return new_lines_set, new_line_labels_set, new_clusters_set

def unify(lines, line_labels, clusters, new_lines_set, new_line_labels_set, new_clusters_set):

    count = len(new_lines_set)

    line_labels[:, 0] = line_labels[:, 0] + count

    new_lines_set = new_lines_set + lines

    if len(new_line_labels_set):
        new_line_labels_set = np.append(new_line_labels_set, line_labels, axis=0)
    else:
        new_line_labels_set = line_labels

    for cluster_id in range(len(clusters)):
        new_clusters_set[cluster_id] = new_clusters_set[cluster_id] + [line_id + count for line_id in clusters[cluster_id]]

    return new_lines_set, new_line_labels_set, new_clusters_set


def resort_lines(lines, line_labels, clusters, table_gclabel_vp):

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

    for cluster_id in range(len(clusters)):

        cluster_ids = [id_mapping[:, 1][id_mapping[:, 0] == clusters[cluster_id][i]][0] for i in
                       range(len(clusters[cluster_id]))]

        clusters_set[cluster_id] = cluster_ids

    return lines_set, line_labels_set, clusters_set

def gen_lineproposals(lines, vps2D, gc_map, clusters, line_labels, mask_map, ifinferLines, ifinferExtralines):
    # To generate line proposals from the masked lines on the edge map. The generated lines are with four gc labels on the
    # frontal wall

    # give the gc label list of the gc map
    gc_labels = np.unique(gc_map)

    # generate lines (line_gc) located between gc content pixels. e.g. For the line located between gc area frontal wall
    # (2) and left wall(3). The following function will get the line along with its gc label (line_gc_labels)(23).
    # line_gc_clusters: This a very important output, which gives the corresponding relationship between gc_label of lines
    # and the vanishing points.

    ifscaleimg = True # If scale the image for speed up efficiency
    lines_gc, line_gc_labels, line_gc_clusters = gen_lines_fromGC(gc_map, vps2D, ifscaleimg)

    # correspond gc_label to vanishing point id
    # build a lookup table about the corresponding vp id of each gc_label of line
    table_gclabel_vp = []
    for line_id, label in line_gc_labels:
        vpid = [line_id in cluster for cluster in line_gc_clusters].index(True)
        table_gclabel_vp.append([label, vpid])

    table_gclabel_vp = np.array(table_gclabel_vp)

    # a line with some gc label should also be consistent with
    # its vanishing point. For example, if a line is labeled as 23, it also should correspond to the same
    # vp with 23 (from the lookup table table_gclabel_vp),
    # otherwise, this line should be filtered out, as it can not be used in inferring new lines.

    # In resort_lines(), wrong lines are filtered out and the selected lines are numbered from 0.
    lines_set, line_labels_set, clusters_set = resort_lines(lines, line_labels, clusters, table_gclabel_vp)

    # Infer more lines from the filtered line segments.
    if ifinferLines:

        lines_set, line_labels_set, clusters_set = infer_lines(lines_set, line_labels_set, clusters_set, vps2D,
                                                               gc_labels, table_gclabel_vp, mask_map,
                                                               ifinferExtralines)
    # combine to the inferred line set with the original line set
    new_lines_set, new_line_labels_set, new_clusters_set = comb_to_set(lines_gc, line_gc_labels, lines_set, line_labels_set, clusters_set, gc_labels, table_gclabel_vp)

    return  new_lines_set, new_line_labels_set, new_clusters_set, table_gclabel_vp

def get_score(lines, edge_map):

    total_score = 0.
    total_num = 0

    step_len = 10.

    for line in lines:
        pt1 = np.array([line[2], line[3]])
        pt2 = np.array([line[0], line[1]])
        length = np.linalg.norm(pt1 - pt2)
        step_vec = (pt2 - pt1) / length

        c_len = 0
        pnts = []

        while c_len < length:

            pnt = pt1 + c_len * step_vec

            c_len += step_len

            if 0<=pnt[0]<=edge_map.shape[1]-1 and 0<=pnt[1]<=edge_map.shape[0]-1:

                pnts.append(pnt)

            else:
                break

        pnts = np.round(pnts)

        scores = [edge_map[int(hid), int(wid)] for wid, hid in pnts]

        score = sum(scores)
        num = len(scores)

        total_score += score
        total_num += num

    return total_score/total_num


def gen_proposals(lines_set, clusters_set, vps2D, vp2D, plabels, lineIDs_gclabel, edge_map):
    # lines_set: stores all line proposals.
    # clusters_set: stores the relations between line id and its corresponding vp id.
    # vps2D: vanishing points on 2D image.
    # vp2D: the vp that do nothing with the forming the frontal wall.
    # plabels: the four labels located around the frontal wall
    # lineIDs_gclabel: each subset contains line ids with the same gc label.
    # edge_map: the probability map of the room layout.

    __, width = edge_map.shape

    # left wall and right wall are parallel to each other in general, so as to the floor and the ceiling.
    # left wall(or right wall) must intersect with the floor and the ceiling, hence we divide the two sets as the following.
    label_set1 = np.intersect1d(plabels, [gc_def['left_wallID'], gc_def['right_wallID']])
    label_set2 = np.intersect1d(plabels, [gc_def['floor_ID'], gc_def['ceiling_ID']])

    proposals = []

    score_list = []

    line_combs = np.stack(np.meshgrid(*lineIDs_gclabel), axis=len(plabels)).reshape(-1, len(plabels))

    for comb_id in xrange(len(line_combs)):

        new_lines = []
        cornerpt_dict = {}

        # initialisation
        for plabel in line_combs[comb_id]:
            cornerpt_dict[plabel] = []

        for plabel1 in label_set1:
            line1_id = line_combs[comb_id][plabels.index(plabel1)]

            p1 = np.array([lines_set[line1_id][0], lines_set[line1_id][1], 1.0])
            p2 = np.array([lines_set[line1_id][2], lines_set[line1_id][3], 1.0])
            line1 = np.cross(p1, p2)

            for plabel2 in label_set2:
                line2_id = line_combs[comb_id][plabels.index(plabel2)]

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

                cornerpt_dict[line1_id].append(corner[:2])
                cornerpt_dict[line2_id].append(corner[:2])

                new_lines.append(newline)

        for line_id in line_combs[comb_id]:
            corners = cornerpt_dict[line_id]

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

        # get the score of each proposal.
        layout_score = get_score(new_lines, edge_map)

        score_list.append(layout_score)
        proposals.append(new_lines)

        if comb_id % 1000 == 0:
            print "Current %d th step; Score: %f.\n" % (comb_id, layout_score)

    return proposals, score_list

def gen_layoutproposals(lines_set, line_labels_set, clusters_set, table_gclabel_vp, vps2D, gc_labels, edge_map):
    # line_set: all line proposals.
    # line_labels_set: gc_labels correspond with lines in the line_set.
    # clusters_set: stores the relationships between the line_id and its corresponding vp_id.
    # table_gclabel_vp: stores the relations between gc_label of lines and the vp_id
    # vps2D: vp on 2D image.
    # gc_labels: existing gc content label of this case.
    # edge_map: the score map of the room layout.

    # find out the third vp
    for gc_label, vp_id in table_gclabel_vp:
        if str(gc_def['frontal_wallID']) not in str(gc_label):
            vp2D = vps2D[vp_id]
            break

    # the four labels are located around the frontal wall, hence their gc label of area should not contain the frontal
    # wall and background.
    plabels = list(np.setdiff1d(gc_labels, [gc_def['backgroundID'], gc_def['frontal_wallID']]))

    # found not the line ids who are located around the frontal wall.
    # In lineIDs_gclabel, line ids in whose subset have the same gc_label of line.
    lineIDs_gclabel = []

    for plabel in plabels:

        if gc_def['frontal_wallID'] < plabel:
            label = gc_def['frontal_wallID'] * 10 + plabel
        else:
            label = plabel * 10 + gc_def['frontal_wallID']

        lineIDs_gclabel.append(line_labels_set[line_labels_set[:,1] == label, 0])

    # generate layout proposals with lines in lineIDs_gclabel, and give their fitting score on the edge_map.
    proposals, score_list = gen_proposals(lines_set, clusters_set, vps2D, vp2D, plabels, lineIDs_gclabel, edge_map)

    return proposals, score_list

def processGC(gc_map):

    # restric gc_map to limited cases
    gc_map_new = np.copy(gc_map)
    gc_labels = np.unique(gc_map)
    if gc_def['frontal_wallID'] not in gc_labels:
        if gc_def['right_wallID'] in gc_labels:
            gc_map_new[gc_map==gc_def['right_wallID']] = gc_def['frontal_wallID']
            return gc_map_new
        elif gc_def['left_wallID'] in gc_labels:
            gc_map_new[gc_map == gc_def['left_wallID']] = gc_def['frontal_wallID']
            return  gc_map_new
        else:
            return None
    else:
        return gc_map_new

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
    # In this part, we define the gc label for each line (generally in pair form)
    # e.g. If the a line is located between the frontal wall (2) and the left wall (3),
    # then the label for this line is 23.
    # While, in the following process, a line with some label should also be consistent with
    # its vanishing point. For example, if a line is labeled as 23, it also should correspond to the same
    # vp with 23, otherwise, this line should be filtered out, as it can not be used in inferring new lines.

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

    # here, we used nearest neighbour method to judge its label.
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
    # Judge whether these lines are located in the mask == 1 area.

    # We use num_checks to uniformly sample points on lines to conduct the judgement.

    num_checks = 4
    new_clusters = [[] for i in range(len(clusters))]

    for cluster_id in range(len(clusters)):
        for line_id in clusters[cluster_id]:
            pt1 = np.array([lines[line_id][0], lines[line_id][1]])
            pt2 = np.array([lines[line_id][2], lines[line_id][3]])

            pnts = np.uint(np.round(pnts_gen(pt1, pt2, num_checks)))
            # pnts = np.array(pnts_gen(pt1, pt2, num_checks), dtype=np.uint)

            ifpick = 1
            for pnt in pnts:
                if mask[pnt[1], pnt[0]] == 0:
                    ifpick = 0
                    break
            if ifpick == 1:
                new_clusters[cluster_id].append(line_id)

    return new_clusters