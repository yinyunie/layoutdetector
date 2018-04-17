import cv2
import numpy as np
from vpdetector import VPDetection
from pylsd.lsd import lsd
import random

def LineDetect(image, thLength):
    if image.shape[2] == 1:
        grayImage = image
    else:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imageLSD = np.copy(grayImage)

    # line segments, [pt1[0], pt1[1], pt2[0], pt2[1], width]
    linesLSD = lsd(imageLSD)
    del imageLSD

    # choose line segments whose length is less than thLength
    lineSegs = []
    for line in linesLSD:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        length = np.sqrt( ( x1 - x2 ) ** 2 + ( y1 - y2 ) ** 2 )
        if length > thLength:
            lineSegs.append([x1, y1, x2, y2])

    return lineSegs

def drawClusters(image, lines, clusters, colorPattern, linelabels = None):

    if colorPattern == 'vps':
        palette = [(255,0,0), (0,255,0), (0,0,255)]
    elif colorPattern == 'gc':
        palette = np.random.randint(0, 256, size=[8,3])
        labellist = [23, 24, 25, 26, 35, 36, 45, 46]

    colorID = 0
    for cluster_id in range(len(clusters)):
        for line_id in clusters[cluster_id]:
            pt1 = (np.int(lines[line_id][0]), np.int(lines[line_id][1]))
            pt2 = (np.int(lines[line_id][2]), np.int(lines[line_id][3]))
            if colorPattern == 'vps':
                cv2.line(image, pt1, pt2, palette[colorID], 2)
            elif colorPattern == 'gc':
                label = linelabels[linelabels[:,0]==line_id, 1]
                cid = labellist.index(label)
                cv2.line(image, pt1, pt2, palette[cid,:], 5)

        colorID += 1

    return image

def drawBox(image, vps, f, pp):
    vp2D = [[] for i in xrange(3)]
    for i in xrange(3):
        vp2D[i] = np.array([vps[i][0] * f / vps[i][2] + pp[0], vps[i][1] * f / vps[i][2] + pp[1]])

    space = 20
    width = image.shape[1]
    height = image.shape[0]

    upline = np.array([[i, 0] for i in range(0, width, space)])
    bottomline = np.array([[i, height - 1] for i in range(0, width, space)])
    leftline = np.array([[0, i] for i in range(0, height, space)])
    rightline = np.array([[width - 1, i] for i in range(0, height, space)])
    points = np.vstack([upline, bottomline, leftline, rightline])

    palatte = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(len(points)):
        pt1 = points[i][0], points[i][1]
        for k in xrange(3):
            pt2 = (np.int(vp2D[k][0]), np.int(vp2D[k][1]))
            cv2.line(image, pt1, pt2, palatte[k], 1)

    return image

def getCameraParas(lines, clusters):
    vp2D = [[] for i in range(3)]
    count = 0
    for cluster in clusters:
        lineMatrix = []
        Weights =  []
        for line_id in cluster:
            pt1 = np.array([lines[line_id][0], lines[line_id][1], 1.0])
            pt2 = np.array([lines[line_id][2], lines[line_id][3], 1.0])
            lineMatrix.append( np.cross(pt1, pt2) )
            Weights.append(np.linalg.norm(pt1 - pt2))

        lineMatrix = np.array(lineMatrix)
        Weights = np.array(Weights)
        Weights = np.diag(Weights/sum(Weights))

        # weighted MLS estimation
        A = lineMatrix[:, :2]
        y = -lineMatrix[:, 2]

        pt = np.linalg.inv(A.T.dot(Weights.T).dot(Weights).dot(A)).dot(A.T).dot(Weights.T).dot(Weights).dot(y)

        # pt = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)
        # # eigen value solution
        # eigenValues, eigenVecs = np.linalg.eig(lineMatrix.T.dot(lineMatrix))
        # pt_eigen = eigenVecs[:,np.argmin(eigenValues)]
        # pt_eigen = pt_eigen/pt_eigen[2]
        vp2D[count] = pt
        count = count + 1

    CoefMatrix = np.zeros([3, 4])
    count = 0
    for i in range(3):
        for j in range(i+1, 3):
            CoefMatrix[count][0] = vp2D[i][0] * vp2D[j][0] + vp2D[i][1] * vp2D[j][1]
            CoefMatrix[count][1] = vp2D[i][0] + vp2D[j][0]
            CoefMatrix[count][2] = vp2D[i][1] + vp2D[j][1]
            CoefMatrix[count][3] = 1.0
            count = count + 1
    eigenValues, eigenVecs = np.linalg.eig(CoefMatrix.T.dot(CoefMatrix))

    paras = eigenVecs[:, np.argmin(eigenValues)]

    SMatrix = np.array([[paras[0], 0., paras[1]], [0., paras[0], paras[2]], [paras[1], paras[2], paras[3]]])
    K_temp = np.linalg.inv(np.linalg.cholesky(SMatrix).T)
    K = K_temp / K_temp[2,2]

    return K

def calibrate(image, mode = 0, ifplot = 1):
    # Line segment detection
    thLength = 30.0 # threshold of the length of line segments

    # detect line segments from the source image
    lines = LineDetect( image, thLength)

    # Camera internal parameters
    pp = image.shape[1]/2., image.shape[0]/2. # principle point (in pixel)

    f = np.double(np.max(image.shape))# focal length (in pixel), a former guess

    noiseRatio = 0.5
    # VPDetection class
    detector = VPDetection(lines, pp, f, noiseRatio)
    vps, clusters = detector.run()

    # decide camera intrinsic parameters
    if mode == 0:
        K = [[f, 0., pp[0]], [0., f, pp[1]], [0., 0., 1.]]
    elif mode == 1:
        K = getCameraParas(lines, clusters)
        # Its ok to replace with the new camera intrinsic parameters to estimate the camera extrinsic matrix,
        # but the difference is minor.
        detector = VPDetection(lines, [K[0, 2], K[1, 2]], K[0, 0], noiseRatio)
        vps, clusters = detector.run()


    if ifplot:
        image1 = np.copy(image)
        drawClusters(image1, lines, clusters, 'vps')
        cv2.imshow("", image1)
        cv2.waitKey(0)

        image2 = np.copy(image)
        drawBox(image2, vps, K[0][0], [K[0][2], K[1][2]])
        cv2.imshow("", image2)
        cv2.waitKey(0)

    return K, vps, clusters, lines

