if __name__ == '__main__':
    import argparse
    import cv2
    import sys
    from addons import camera
    import scipy.io as sio
    import numpy as np
    from addons.lib import line_filter, decide_linelabel, processGC, gen_lineproposals

    parser = argparse.ArgumentParser(description="Layout prediction, vanishing point detection and camera orientation decision from "
                                                 "a single image.")
    parser.add_argument("-f", "--file", dest="filename", type=str, metavar="FILE",
                        help="Give the address of image source")
    parser.add_argument("-l", "--layout", dest="layout", type=str, metavar="LAYOUT",
                        help="Give the address of coarse layout of the image.")
    parser.add_argument("-g", "--gc", dest="gc", type=str, metavar="GEOCONTENT",
                        help="Give the address of geometric content of the image.")
    args = parser.parse_args()

    # Read source image
    try:
        image = cv2.imread(args.filename)
    except IOError:
        print 'Cannot open the image file, please verify the image address.'

    # get Camera intrinsic matrix and vanishing points
    mode = 1
    K, vps, clusters, lines = camera.calibrate(image, mode, 1)

    # read coarse layout
    try:
        edge_map = sio.loadmat(args.layout)['edge_prob_map']
    except IOError:
        print 'Cannot open the layout file, please verify the address.'

    mask_map = (edge_map - np.min(edge_map))/(np.max(edge_map) - np.min(edge_map))

    mask_map[mask_map < 0.1] = 0.
    mask_map[mask_map >= 0.1] = 1.

    kernel = np.ones((4, 4), np.uint8)
    mask_map = cv2.dilate(mask_map, kernel, iterations=1)
    mask_map = np.uint8(mask_map)

    # use mask_map to filter out wrong line members
    new_clusters = line_filter(lines, clusters, mask_map)

    # draw filtered lines
    image1 = np.copy(image)
    camera.drawClusters(image1, lines, new_clusters, 'vps')
    cv2.imshow('', image1)
    cv2.waitKey(0)

    # next to generate line candidates
    # 1. define the role of each line (between front wall and floor, e.t.c.)
    # 2. using vanishing point to extend lines to ensure each corner point
    # 3. use each corner to generate occluded lines or undetected lines.
    # 4. use vanishing point to generate random lines within mask area.
    # 5. generate proposals

    # read geometric content of the image
    try:
        gc_map = sio.loadmat(args.gc)['GC_map']
        gc_map = processGC(gc_map)
    except IOError:
        print 'Cannot open the gc file, please verify the address.'

    # 1. define the role of each line (between front wall and floor, e.t.c.)
    # Definition: eight classes of lines are defined, but only four are used
    line_labels = decide_linelabel(lines, new_clusters, gc_map)

    image1 = np.copy(image)
    camera.drawClusters(image1, lines, new_clusters, 'gc', line_labels)
    cv2.imshow('', image1)
    cv2.waitKey(0)

    # generate line proposals
    lines_gc, line_gc_labels, line_gc_clusters = gen_lineproposals(lines, vps, K, mask_map, gc_map, new_clusters, line_labels)
    image1 = np.copy(image)
    camera.drawClusters(image1,lines_gc, line_gc_clusters, 'gc', line_gc_labels)
    cv2.imshow('', image1)
    cv2.waitKey(0)

    print 'Debug'









