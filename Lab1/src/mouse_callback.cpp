#include <opencv2/opencv.hpp>

#define RECT_Y_LEN 9
#define RECT_X_LEN 9
#define THRESHOLD 80
#define AREA_LEN 20

using namespace cv;
using namespace std;

void mouseCallback(int event, int x, int y, int flags, void *data) {
    if (event != EVENT_LBUTTONDOWN)
        return;

    //cout << "x: " << x << "   y: " << y << endl;

    //Mat image_copy = (*(Mat*) data).clone();
    Mat image_copy = *(Mat*) data;

    Rect rect(x, y, RECT_X_LEN, RECT_Y_LEN);
    Scalar avg = mean(image_copy(rect));
    cout << "Mean: " << avg << endl;

    for (int row = x-AREA_LEN; row <= x+AREA_LEN; row++) {
        for (int col = y-AREA_LEN; col <= y+AREA_LEN; col++) {

            if (
                    abs(image_copy.at<Vec3b>(col, row)[0] - 35) < THRESHOLD
                    && abs(image_copy.at<Vec3b>(col, row)[1] - 155) < THRESHOLD
                    && abs(image_copy.at<Vec3b>(col, row)[2] - 205) < THRESHOLD
                    ) {

                cout << "Setting pixels" << endl;
                image_copy.at<Vec3b>(col, row)[0] = 92;
                image_copy.at<Vec3b>(col, row)[1] = 37;
                image_copy.at<Vec3b>(col, row)[2] = 201;
            }
        }
    }

    imshow("Image", image_copy);

}