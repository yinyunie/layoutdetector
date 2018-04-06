if __name__ == '__main__':
    import argparse
    import cv2
    from addons import camera

    parser = argparse.ArgumentParser(description="Vanishing point detection script with camera intrinsic parameter decision.")
    parser.add_argument("-f", "--file", dest = "filename", type=str, metavar="FILE", help = "Give the address of image source")
    args = parser.parse_args()

    # Read source image
    try:
        image = cv2.imread(args.filename)
    except IOError:
        print 'Cannot open the image file, please verify the image address.'

    mode = 1
    K, R = camera.calibrate(image, mode, 1)