if __name__ == '__main__':
    import argparse
    import cv2
    from addons import camera
    import scipy.io as sio
    import numpy as np
    from addons.lib import line_filter

    parser = argparse.ArgumentParser(description="Layout prediction, vanishing point detection and camera orientation decision from "
                                                 "a single image.")
    parser.add_argument("-f", "--file", dest="filename", type=str, metavar="FILE",
                        help="Give the address of image source")
    parser.add_argument("-l", "--layout", dest="layout", type=str, metavar="LAYOUT",
                        help="Give the address of coarse layout of the image.")
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
    camera.drawClusters(image1, lines, new_clusters)
    cv2.imshow('', image1)
    cv2.waitKey(0)


    print 'Debug'









