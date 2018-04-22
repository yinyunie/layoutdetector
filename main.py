if __name__ == '__main__':
    import argparse
    import cv2
    from addons import camera
    import scipy.io as sio
    import numpy as np
    from addons.lib import line_filter, decide_linelabel, processGC, gen_lineproposals, gen_layoutproposals, get_mask

    parser = argparse.ArgumentParser(description="Layout prediction, vanishing point detection and camera orientation decision from "
                                                 "a single image.")
    parser.add_argument("-f", "--file", dest="filename", type=str, metavar="FILE",
                        help="Give the address of image source")
    parser.add_argument("-l", "--layout", dest="layout", type=str, metavar="LAYOUT",
                        help="Give the address of coarse layout of the image.")
    parser.add_argument("-g", "--gc", dest="gc", type=str, metavar="GEOCONTENT",
                        help="Give the address of geometric content of the image.")
    args = parser.parse_args()

    # Read the source image
    try:
        image = cv2.imread(args.filename)
    except IOError:
        print 'Cannot open the image file, please verify the image address.'

    # get Camera intrinsic matrix and vanishing points.
    # mode = 1: estimate the vanishing points (vps) and vanishing directions using the estimated K.
    #           focal length = 1.2*max(cols,rows) of the image.
    # model = 2: use the LSE to re-estimate the camera parameters after the found vps.
    # model = 3: use the eigen value decomposition solution to re-estimate the camera parameters after the found vps.
    mode = 2
    K, vps, clusters, lines = camera.calibrate(image, mode, 1)

    # get the vps in 2D image plane.
    vps2D = [[] for i in range(3)]
    for i in xrange(3):
        vps2D[i] = np.array([vps[i][0] * K[0, 0] / vps[i][2] + K[0, 2], vps[i][1] * K[0, 0] / vps[i][2] + K[1, 2]])

    # read the coarse layout file.
    try:
        edge_map = sio.loadmat(args.layout)['edge_prob_map']
    except IOError:
        print 'Cannot open the layout file, please verify the address.'

    # get the binary mask from the edge map with the threshold and dilation step.
    # A higher threshold will largely improve the entire efficiency.
    mask_map = get_mask(edge_map, 0.1, 4)

    # use mask_map to filter out wrong line members
    new_clusters = line_filter(lines, clusters, mask_map)

    # draw filtered lines
    camera.drawClusters(image, lines, new_clusters, 'vps')

    # read geometric content of the image
    try:
        gc_map = sio.loadmat(args.gc)['GC_map']
        gc_map = processGC(gc_map)
    except IOError:
        print 'Cannot open the gc file, please verify the address.'

    # define the role of each line (between front wall and floor, e.t.c.)
    # definition: eight classes of lines are defined, but only four are used
    line_labels = decide_linelabel(lines, new_clusters, gc_map)

    camera.drawClusters(image, lines, new_clusters, 'gc', line_labels)

    # to generate line proposals for geometric-content(gc) labels placed around the frontal wall
    # Hence, only four kinds (divided by gc label) of line proposals are generated.
    # If check the ifinferLines tag, more line proposals will be inferred from the existing line segments
    # If check the ifinferExtralines, extra lines will be inferred from the existing inferred lines.
    ifinferLines = True
    ifinferExtralines = True
    lines_set, line_labels_set, clusters_set, table_gclabel_vp = gen_lineproposals(lines, vps2D, gc_map, new_clusters,
                                                                                   line_labels, mask_map, ifinferLines,
                                                                                   ifinferExtralines)
    camera.drawClusters(image, lines_set, clusters_set, 'vps')

    # generate layout proposals
    gc_labels = np.unique(gc_map)

    # Generate layout proposals from these line proposals.
    # A layout proposal is generated from line proposals with four different gc_labels (frontal-left wall,
    # frontal-right wall, frontal wall-ceiling and frontal wall-floor) with a vp.
    proposals, score_list = gen_layoutproposals(lines_set, line_labels_set, clusters_set, table_gclabel_vp, vps2D,
                                                gc_labels, edge_map)

    camera.draw_proposals(image, [proposals[i] for i in np.random.randint(0, len(proposals), 10)])

    camera.draw_proposals(image, [proposals[score_list.index(max(score_list))]])

    print 'Debug'
