#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "panoramic_image.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Mat equalize(Mat input) {
    vector<Mat> channelsHSV(3);
    Mat hsv;
    cvtColor(input, hsv, COLOR_BGR2HSV);
    split(hsv, channelsHSV);

    equalizeHist(channelsHSV[1], channelsHSV[1]);
    equalizeHist(channelsHSV[2], channelsHSV[2]);
    merge(channelsHSV, hsv);

    cvtColor(hsv, hsv, COLOR_HSV2BGR);

    return hsv;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        // argv[0] is the executable name
        cout << "USAGE: $" << argv[0] << " PANORAMIC_FOLDER_PATH CAMERA_FOV MATCH_FILTER_RATIO" << endl;
        cout << "PANORAMIC_FOLDER_PATH: path to the lab image" << endl;
        cout << "CAMERA_FOV: field of view of the camera used to take the pictures inside PANORAMIC_FOLDER_PATH" << endl;
        cout << "MATCH_FILTER_RATIO: used to discard pair of matches with distance > match_filter_ratio * min_pair_distance" << endl;

        return 1;
    }

    string img_folder_path(argv[1]);
    double fov = atof(argv[2]);
    float match_filter_ratio = atof(argv[3]);

    Mat panoramic = PanoramicImage(img_folder_path, fov)
        .findKeypoints()
        .findMatches()
        .refineAndComputeTranslations(match_filter_ratio)
        .composePanoramicImage();

    panoramic = equalize(panoramic);

    namedWindow("Panoramic", WINDOW_NORMAL);
    imshow("Panoramic", panoramic);

    waitKey(0);
    destroyAllWindows();

    return 0;
}


