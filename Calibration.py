"""
Camera Calibration, including monocular, stereo and hand-eye.
Author: Haowen Wang
Reference: https://github.com/bvnayak/stereo_calibraion
Date: 2022/03/21
"""

import numpy as np
import cv2
import glob
import argparse
import yaml
import matplotlib.pyplot as plt
from tqdm import trange

class Calibration(object):

    def __init__(self, filepath):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        self.criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        self.objp = np.zeros((11*8, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 9.55  # distance between corners is 9.55mm

        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        # self.test_path = rebuildpath

    def show(self, img):
        cv2.namedWindow('img', 0)
        cv2.resizeWindow('img', 600, 500)
        cv2.imshow('img', img)
        cv2.waitKey(1000)
        cv2.destroyWindow('img')

    def monocular_calibrate(self, cal_path, display):
        images_left = glob.glob(cal_path + 'l*.bmp')
        images_right = glob.glob(cal_path + 'r*.bmp')
        images_left.sort()
        images_right.sort()
        self.img_left = images_left
        self.img_right = images_right

        for i in trange(len(images_left)):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (11, 8), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (11, 8), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (11, 8), corners_l, ret_l)
                if display:
                    self.show(img_l)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (11, 8), corners_r, ret_r)
                if display:
                    self.show(img_r)

            self.img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.img_shape, None, None)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims, criteria=self.criteria_stereo, flags=flags)

        self.R = R
        self.T = T
        self.M1 = M1
        self.d1 = d1
        self.M2 = M2
        self.d2 = d2

        camera_model = dict([('M1', np.asarray(M1).tolist()), 
                            ('M2', np.asarray(M2).tolist()),
                            ('dist1', np.asarray(d1).tolist()),
                            ('dist2', np.asarray(d2).tolist()),
                            ('r1', np.asarray(self.r1).tolist()),
                            ('r2', np.asarray(self.r2).tolist()),
                            ('t1', np.asarray(self.t1).tolist()),
                            ('t2', np.asarray(self.t2).tolist()),
                            ('R', np.asarray(R).tolist()),
                            ('T', np.asarray(T).tolist()),
                            ('E', np.asarray(E).tolist()),
                            ('F', np.asarray(F).tolist())])

        with open("./summary/calibration_matrix.yaml", "w") as f:
            yaml.dump(camera_model, f)

        cv2.destroyAllWindows()

        return camera_model

    def _read_gripper2base(self):
        t = []
        R = []
        with open('ur.txt', 'r') as f:
            for i in range(20):
                line = f.readline().strip('\n')
                data = np.asarray(line.split(' ')).astype(np.float32)
                t.append(data[:3])
                R.append(cv2.Rodrigues(data[3:])[0])  # vector to matrix
            t = np.asarray(t).reshape((20, 3, 1))
            R = np.asarray(R).reshape((20, 3, 3))

        return t, R

    def hand_eye_calibrate(self):
        self.t_gripper2base, self.R_gripper2base = self._read_gripper2base()
        self.R_cam2gripper_l, self.t_cam2gripper_l = cv2.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.r1, self.t1)
        self.R_cam2gripper_r, self.t_cam2gripper_r = cv2.calibrateHandEye(self.R_gripper2base, self.t_gripper2base, self.r2, self.t2)

    def undistortion(self, seed, display):
        if seed < 0 or seed > len(self.img_left):
            print("Input seed error! Choose another one.")
            return;
        instance_img = cv2.imread(self.img_right[seed])
        h = self.img_shape[0]
        w = self.img_shape[1]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.M1, self.d1, (w, h), 0, (w, h))
        result = cv2.undistort(instance_img, self.M1, self.d1, None, newcameramtx)
        if display:
            plt.subplot(121)
            plt.imshow(instance_img, cmap='gray')
            plt.title('Input Image')
            plt.subplot(122)
            plt.imshow(result, cmap='gray')
            plt.title('After Undistortion')
            plt.show()

    def getRectifyTransformation(self):
        h = self.img_shape[1]
        w = self.img_shape[0]
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.M1, self.d1, self.M2, self.d2, (w, h), self.R, self.T, alpha=0)
        map1x, map1y = cv2.initUndistortRectifyMap(self.M1, self.d1, R1, P1, (w, h), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(self.M2, self.d2, R2, P2, (w, h), cv2.CV_32FC1)

        return map1x, map1y, map2x, map2y, Q

    def rectify(self, test_path, display):
        image_test = glob.glob(test_path + '*.jpg')  # images waited to be rectified
        image1 = cv2.imread(image_test[0])
        image2 = cv2.imread(image_test[1])

        map1x, map1y, map2x, map2y, Q = self.getRectifyTransformation()
        rectified_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
        rectified_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

        self.image1 = rectified_img1
        self.image2 = rectified_img2

        if display:
            plt.subplot(121)
            plt.imshow(image1)
            plt.title('Left image (after rectify)')
            plt.subplot(122)
            plt.imshow(image2)
            plt.title('Right image (after rectify)')
            plt.show()

        if display:
            plt.subplot(121)
            plt.imshow(rectified_img1)
            plt.title('Left image (after rectify)')
            plt.subplot(122)
            plt.imshow(rectified_img2)
            plt.title('Right image (after rectify)')
            plt.show()

        return rectified_img1, rectified_img2

    def test_monocular(self):
        # reconstruct from 3D to 2D
        total_error_l = 0
        total_error_r = 0
        for i in range(len(self.objpoints)):
            imgpoints_l_re, _ = cv2.projectPoints(self.objpoints[i], self.r1[i], self.t1[i], self.M1, self.d1)
            imgpoints_r_re, _ = cv2.projectPoints(self.objpoints[i], self.r2[i], self.t2[i], self.M2, self.d2)
            error_l = cv2.norm(self.imgpoints_l[i], imgpoints_l_re, cv2.NORM_L2) / len(imgpoints_l_re)
            error_r = cv2.norm(self.imgpoints_r[i], imgpoints_r_re, cv2.NORM_L2) / len(imgpoints_r_re)
            total_error_l += error_l
            total_error_r += error_r
        print("---------- Monocular projection error ----------")
        print("total error left: ", total_error_l / len(self.objpoints))
        print("total error right: ", total_error_r / len(self.objpoints))

    def update(self, val=0):
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')
        blockSize = cv2.getTrackbarPos('blockSize','disp')
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        self.stereo.setNumDisparities(numDisparities)
        self.stereo.setBlockSize(blockSize)
        self.stereo.setPreFilterCap(preFilterCap)
        self.stereo.setUniquenessRatio(uniquenessRatio)
        self.stereo.setSpeckleRange(speckleRange)
        self.stereo.setSpeckleWindowSize(speckleWindowSize)
        self.stereo.setDisp12MaxDiff(disp12MaxDiff)
        self.stereo.setMinDisparity(minDisparity)

        disparity = self.stereo.compute(self.image1, self.image2)
        disparity = disparity.astype(np.float32)
        disparity = (disparity/16.0 - minDisparity)/numDisparities
        cv2.imshow("disp",disparity)

    def sgbm(self):
        cv2.namedWindow('disp')
        cv2.createTrackbar('numDisparities','disp',100,400,self.update)
        cv2.createTrackbar('blockSize','disp',1,50,self.update)
        cv2.createTrackbar('preFilterCap','disp',5,62,self.update)
        cv2.createTrackbar('uniquenessRatio','disp',0,40,self.update)
        cv2.createTrackbar('speckleRange','disp',0,30,self.update)
        cv2.createTrackbar('speckleWindowSize','disp',0,30,self.update)
        cv2.createTrackbar('disp12MaxDiff','disp',0,100,self.update)
        cv2.createTrackbar('minDisparity','disp',0,25,self.update)

        self.stereo = cv2.StereoSGBM_create()
        self.update()
        cv2.waitKey(0)

    def test_stereo(self):
        # project from 3D to 2D
        total_dist_h = 0
        total_dist_v = 0

        for seed in range(20):
            r1_matrix, _ = cv2.Rodrigues(self.r1[seed])
            r1 = np.asarray(r1_matrix)
            t1 = np.asarray(self.t1[seed])
            r2_matrix, _ = cv2.Rodrigues(self.r2[seed])
            r2 = np.asarray(r2_matrix)
            t2 = np.asarray(self.t2[seed])

            projection_matrix_l = np.matmul(self.M1, np.concatenate((r1, t1), axis=1))
            projection_matrix_r = np.matmul(self.M2, np.concatenate((r2, t2), axis=1))

            proj_points_l = np.asarray(self.imgpoints_l[seed]).reshape(-1, 2).T
            proj_points_r = np.asarray(self.imgpoints_r[seed]).reshape(-1, 2).T

            points = cv2.triangulatePoints(projection_matrix_l, projection_matrix_r, proj_points_l, proj_points_r)

            points_world = cv2.convertPointsFromHomogeneous(points.T)
            points = points_world.reshape(8, 11, 3)
            
            for i in range(8):
                for j in range(10):
                    dist_h = np.linalg.norm(points[i][j+1] - points[i][j])
                    total_dist_h += dist_h
            for i in range(7):
                for j in range(11):
                    dist_v = np.linalg.norm(points[i+1][j] - points[i][j])
                    total_dist_v += dist_v
           
        print("---------- Stereo projection error ----------")
        print("Ground truth distance: 9.55mm")
        print("Average distance between corner points (horizontal): ", total_dist_h / (80*20), 'mm')
        print("Average distance between corner points (vertical): ", total_dist_v / (77*20), 'mm')
        print("Error: ", 0.5*((total_dist_h/(80*20)-9.55)/9.55*100 + (total_dist_v / (77*20)-9.55)/9.55*100), "%")

    def _homo(self, M, t):
        M = np.array(M).reshape((3,3))
        t = np.array(t).reshape((3,1))
        new_m = np.concatenate((M, t), axis=1)
        h = np.array([[0, 0, 0, 1]])
        homo = np.concatenate((new_m, h), axis=0)
        return homo

    def test_hand_eye(self):
        # origin of chess board in robot base (only for left camera)
        points_base = []

        for i in range(20):
            origin = np.concatenate((self.objpoints[0][0], [1]), axis=0)  # in WCS

            M1 = self._homo(self.R_gripper2base[i], self.t_gripper2base[i])
            M2 = self._homo(self.R_cam2gripper_l, self.t_cam2gripper_l)
            M3 = self._homo(cv2.Rodrigues(self.r1[i])[0], self.t1[i])

            point = np.matmul(np.matmul(M1, np.matmul(M2, M3)), origin)
            points_base.append(point[:3])
        
        print("---------- Hand-eye recovery variation ----------")
        print("Origin of chessboard in robot base coordinate system: ")
        print(np.array(points_base))
        print("Variation: ", np.var(points_base, axis=0))


    def run(self, seed, display):
        # Monocular
        self.monocular_calibrate(self.cal_path, display)
        self.undistortion(seed, display)
        self.test_monocular()

        # Stereo
        self.stereo_calibrate(self.img_shape)
        self.test_stereo()

        # Hand-eye
        self.hand_eye_calibrate()
        self.test_hand_eye()

        # Rectify and rebuild
        rectifyed_img1, rectifyed_img2 = self.rectify(self.test_path, display)
        disp = self.sgbm()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', default='./pics/', help='String Filepath')
    parser.add_argument('--rebuildpath', default='./test/', help='String Filepath')
    parser.add_argument('--seed', default=12, help='Choose a view to undistort')
    parser.add_argument('--display', action='store_true', help='Show images')
    args = parser.parse_args()
    calibration = Calibration(args.filepath)
    calibration.run(seed=args.seed, display=args.display)
