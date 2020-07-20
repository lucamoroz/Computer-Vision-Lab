#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void updateCannyWindow(int _, void* data);
void updateHoughWindow(int _, void* data);

/**
 * NOTE: the default struct parameters have been set in order to solve the lab problem, that is the problem of
 * finding the two street-lines around the right road lane and the circle corresponding to the right street sign.
 * To execute the program with the default parameters just press a key to move to the next step.
 *
**/

struct CannyData {
    int minThreshold = 283;
    int ratio = 3;
    Mat src;
    Mat res;
    string targetWin;
};

struct HoughLinesData {
    int rhoAccumulator = 1;
    int thetaAccumulator = 3;
    int threshold = 120;
    int circleAccThreshold = 17;
    int circleMaxRadius = 30;
    Mat edgeImg;
    Mat src;
    vector<Vec2f> lines;
    string targetWin;
};

const char* IMG_PATH = "../data/input.png";

int main() {
    Mat src = imread(IMG_PATH, IMREAD_COLOR);

    Mat srcGray;
    Mat dst = Mat(src.size(), src.type());

    cvtColor(src, srcGray, COLOR_BGR2GRAY);

    CannyData cannyData;
    cannyData.targetWin = "Tune canny params";
    cannyData.src = srcGray;
    namedWindow(cannyData.targetWin, WINDOW_AUTOSIZE);
    createTrackbar("Minimum threshold: ", cannyData.targetWin, &cannyData.minThreshold, 401, updateCannyWindow,(void*)&cannyData);
    createTrackbar("Ratio: ", cannyData.targetWin, &cannyData.ratio, 50, updateCannyWindow,(void*)&cannyData);
    updateCannyWindow(0, (void*)&cannyData);

    waitKey(0);
    destroyAllWindows();

    HoughLinesData houghLinesData;
    houghLinesData.src = src;
    Canny(cannyData.src, houghLinesData.edgeImg, cannyData.minThreshold, cannyData.ratio*cannyData.minThreshold);
    houghLinesData.targetWin = "Tune Hough params";

    namedWindow(houghLinesData.targetWin, WINDOW_AUTOSIZE);
    createTrackbar("Line Rho accumulator: ", houghLinesData.targetWin, &houghLinesData.rhoAccumulator, 50, updateHoughWindow, (void*)&houghLinesData);
    createTrackbar("Line Theta accumulator: ", houghLinesData.targetWin, &houghLinesData.thetaAccumulator, 50, updateHoughWindow, (void*)&houghLinesData);
    createTrackbar("Line Threshold: ", houghLinesData.targetWin, &houghLinesData.threshold, 500, updateHoughWindow, (void*)&houghLinesData);
    createTrackbar("Circle acc threshold: ", houghLinesData.targetWin, &houghLinesData.circleAccThreshold, 500, updateHoughWindow, (void*)&houghLinesData);
    createTrackbar("Circle max radius: ", houghLinesData.targetWin, &houghLinesData.circleMaxRadius, 500, updateHoughWindow, (void*)&houghLinesData);

    updateHoughWindow(0, (void*)&houghLinesData);

    waitKey(0);

    return 0;
}

/**
 * Applies Canny Edge detector to data.src
 * showing the result.
 * @param _ unused
 * @param data must be CannyData
 */
void updateCannyWindow(int _, void* data) {
    CannyData cannyData = *((CannyData*) data);
    if (!cannyData.ratio)
        return;

    if (cannyData.minThreshold) {
        Canny(cannyData.src, cannyData.res, cannyData.minThreshold, cannyData.ratio*cannyData.minThreshold);
        imshow(cannyData.targetWin, cannyData.res);
    } else {
        imshow(cannyData.targetWin, cannyData.src);
    }
}

/**
 * Find lines and circles in data.edgeImg
 * @param _ unused
 * @param data must be of type HoughLinesData
 */
void updateHoughWindow(int _, void* data) {
    HoughLinesData houghData = *((HoughLinesData *) data);
    Mat imgWithLines = houghData.src.clone();

    if (houghData.rhoAccumulator < 1 || houghData.thetaAccumulator < 1 || houghData.threshold < 1) return;

    HoughLines(
            houghData.edgeImg,
            houghData.lines,
            houghData.rhoAccumulator,
            houghData.thetaAccumulator * CV_PI / 180,
            houghData.threshold
    );

    // show the two strongest lines found
    for (int i = 0; i < 2 && i < houghData.lines.size(); i++) {
        float rho = houghData.lines[i][0], theta = houghData.lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(imgWithLines, pt1, pt2, Scalar(0, 0, 255), 2);
    }

    // find circles
    vector<Vec3f> circles;

    HoughCircles(houghData.edgeImg, circles, HOUGH_GRADIENT, 1, houghData.edgeImg.rows/4, 100, houghData.circleAccThreshold, 0, houghData.circleMaxRadius);

    for(size_t i = 0; i < circles.size(); i++ ) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // draw the circle center
        circle(imgWithLines, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // draw the circle outline
        circle(imgWithLines, center, radius, Scalar(0,255,0), 3, 8, 0 );
    }
    imshow(houghData.targetWin, imgWithLines);

}