# Camera Calibration Based on OpenCV
Created by Haowen Wang (王浩闻).

## Introduction
This repository combines monocular calibration, stereo calibration and hand-eye calibration in one **Calibration** Class, providing several evaluation indicators as well. Materials and equipments are provided by course *Machine Vision and Applications (ME6321-020-M01)* in Shanghai Jiao Tong University.

## Requirements
- cv2
- numpy
- tqdm (optional but strongly recommend)

## Usage
Run using the command line:

        python Calibration.py --filepath $PATH_TO_CHESSBOARD_PICS
    
If you want to draw chessboard corners and show figures after undistortion, you can use this:

        python Calibration.py --filepath $PATH_TO_CHESSBOARD_PICS --seed 1 --display

## Results
 The results below depend on the data I collected during the course. Due to copyright reasons, the data is temporarily unavailable.

 You can capture chessboard data with your own camera, and if you have multiple cameras, you can accomplish stereo calibration. Hand-eye calibration is based on our UR robot setup, you need to have transfer matrix from gripper to robot base.

        ---------- Monocular projection error ----------
        total error left:  0.008441095762112924
        total error right:  0.007967011256026382
        ---------- Stereo projection error ----------
        Ground truth distance: 9.55mm
        Average distance between corner points (horizontal):  9.576824239492417 mm
        Average distance between corner points (vertical):  9.58228165824692 mm
        Error:  0.30945496198605205 %
        ---------- Hand-eye recovery variation ----------
        Origin of chessboard in robot base coordinate system:
        [[-526.00269254   48.87568964   12.49978997]
        [-527.13695646   49.45469185    9.99154499]
        [-527.23901533   51.01683069   11.53577053]
        [-528.30095703   51.05902702   12.84296363]
        [-525.72501538   50.77681924   12.73126311]
        [-525.5319542    51.09835747   14.16331476]
        [-523.75720661   51.44166453   14.25730779]
        [-526.88239838   50.418217     12.14274419]
        [-526.59761667   50.62533409   13.72352599]
        [-522.50994762   51.48232041   14.20162578]
        [-525.37762073   51.74032733   12.47099556]
        [-527.76602174   50.24518746   11.83850533]
        [-524.95066462   50.24778366   10.8042137 ]
        [-523.30019807   50.58489379   12.57790372]
        [-526.33172315   50.1413322    12.59329086]
        [-523.92143203   50.49015221   11.61884993]
        [-524.41583159   50.05011489   11.65532372]
        [-523.57398073   51.02241665   12.57667433]
        [-525.56618665   50.29700728   12.42875851]
        [-525.84655598   50.35042758   12.686971  ]]
        Variation:  [2.3701965  0.44302978 1.13055601]

## Reference
stereo_calibration: https://github.com/bvnayak/stereo_calibration