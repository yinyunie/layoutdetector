import numpy as np

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