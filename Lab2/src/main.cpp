#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// const char* CB_DIR = "../data/checkerboard_images/";
// const char* TEST_IMAGE_PATH = "../data/test_image.png";

// Checkerboard parameters
const int CB_ROWS = 5;
const int CB_COLS = 6;
const float EDGE_LEN = 0.11;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "USAGE: " << argv[0] << " CB_FOLDER_PATH TEST_IMG_PATH" << endl;
        cout << "CB_FOLDER_PATH: Path to the folder containing calibration images" << endl;
        cout << "TEST_IMG_PATH: Path to the image that will be corrected according with the computed parameters" << endl;

        return 1;
    }

    char* CB_DIR = argv[1];
    char* TEST_IMAGE_PATH = argv[2];

    vector<Mat> images;
    vector<filesystem::path> names;
    vector<vector<Point3f> > points3d;
    vector<vector<Point2f>> points2d;
    vector<double> imagesError;
    Size cb_size = Size(CB_ROWS, CB_COLS);

    // Init world coordinates
    vector<Point3f> worldCoords;
    for (int i=0; i<CB_COLS; i++) {
        for (int j=0; j<CB_ROWS; j++) {
            worldCoords.push_back(Point3f(j*EDGE_LEN, i*EDGE_LEN,0));
        }
    }

    // Load checkerboard images
    for (const auto & entry : fs::directory_iterator(CB_DIR)) {
        images.push_back(imread(entry.path()));
        names.push_back(entry.path());
    }

    // Detect checkerboard intersections per image using cv::findChessboardCorners
    vector<Point2f> currImgCorners;
    Mat gray;

    for (int i=0; i<images.size(); i++) {

        cvtColor(images[i], gray, COLOR_BGR2GRAY);
        bool found = findChessboardCorners(gray, cb_size, currImgCorners);
        cout << "Elaborated image " << i << " of " << images.size() << endl;

        if (found) {
            // refining pixel coordinates for given 2d points
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER , 30, 0.001);
            cornerSubPix(gray, currImgCorners, cb_size, cv::Size(-1,-1), criteria);
            
            points3d.push_back(worldCoords);
            points2d.push_back(currImgCorners);

        } else
            cout << "Corners not found for img: " << names[i] << endl;
    }

    // Calibrate camera with cv::calibrateCamera()
    cout << "Calibrating camera...\n";

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rotations;
    vector<Mat> translations;
    calibrateCamera(points3d, points2d, images[0].size(), cameraMatrix, distCoeffs, rotations, translations);

    // Print estimated intrinsic and distortion parameters

    cout << "\ncameraMatrix:\n" << cameraMatrix << endl;
    cout << "\n\ndistCoeffs:\n" << distCoeffs << endl;
    
/*
    vector<Point2f> projectedPoints;
    projectPoints(points3d[0], rotations[0], translations[0], cameraMatrix, distCoeffs, projectedPoints);
    cout << "\n\nProjected:  " << projectedPoints[0] << endl;
    cout << "\n\nPoint 2d:  " << points2d[0][0] << endl;
    cout << norm(points2d[0][0] - projectedPoints[0]) << endl;
*/

    // Compute mean reprojected error
    for (int i=0; i<images.size(); i++) {
        vector<Point2f> projectedPoints;
        projectPoints(points3d[i], rotations[i], translations[i], cameraMatrix, distCoeffs, projectedPoints);

        double currImageErr = 0;
        for(int j=0; j<projectedPoints.size(); j++)
            currImageErr += norm(points2d[i][j] - projectedPoints[j]);
        
        // save image's error - used later to show best and worst calibration image
        imagesError.push_back(currImageErr / projectedPoints.size());
    }
    
    // There are different errors I could show, I've chosen to print the average error so the user can add calibrating 
    // images to see how the final error is affected: for instance, adding a few "misleading" images would 
    // lead to an higher error 
    cout << "\nAvg re-projection error: " << sum(imagesError)[0] / images.size() << endl;

    // Print names of the images for which the calibration performs best and worst
    int bestIndex = 0;
    int worstIndex = 0;
    for (int i=0; i<images.size(); i++) {
        if (imagesError[i] < imagesError[bestIndex])
            bestIndex = i;
        if (imagesError[i] > imagesError[worstIndex])
            worstIndex = i;
    }

    cout << "Best calibration image name: " << names[bestIndex] << ", with avg error: " << imagesError[bestIndex] << endl;
    cout << "Worst calibration image name: " << names[worstIndex] << ", with avg error: " << imagesError[worstIndex] << endl;

    // Show best-worst calib images
    string bestImgWin = "Best calib img - err: " + to_string(imagesError[bestIndex]) + " - name: " + string(names[bestIndex]);
    string worstImgWin = "Worst calib img - err: " + to_string(imagesError[worstIndex]) + " - name: " + string(names[worstIndex]);
    namedWindow(bestImgWin, WINDOW_NORMAL);
    namedWindow(worstImgWin, WINDOW_NORMAL);
    imshow(bestImgWin, images[bestIndex]);
    imshow(worstImgWin, images[worstIndex]);

    waitKey(0);
    destroyAllWindows();

    // Undistort and rectify the test image acquired with the same camera, check: cv::initUndistortRectifyMap()
    Mat test_image = imread(TEST_IMAGE_PATH);
    Mat newCameraMatrix, map1x, map1y, result;

    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), newCameraMatrix, test_image.size(), CV_32FC1, map1x, map1y);
    remap(test_image, result, map1x, map1y, cv::INTER_CUBIC);
    undistort(test_image, result, cameraMatrix, distCoeffs);

    // Show result
    resize(test_image, test_image, Size(768,576));
    resize(result, result, Size(768, 576));

    hconcat(test_image, result, result);
    namedWindow("Correction result");
    imshow("Correction result", result);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
