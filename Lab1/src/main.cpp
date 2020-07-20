#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "mouse_callback.h"

using namespace cv;
using namespace std;

int main() {
    const String winName = "Image";
    Mat image;

    image =imread("../data/robocup.jpg");
    if(image.empty()) {
        printf("../data/robocup.jpg NOT FOUND!");
        return 1;
    }
    resize(image, image, Size(image.cols/2, image.rows/2));
    namedWindow(winName, WINDOW_AUTOSIZE);
    imshow(winName, image);
    setMouseCallback(winName, mouseCallback, (void*) &image);

    waitKey(0);
    return 0;
}
