#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

struct MedianFilterData {
    int kernelSize = 1;
    Mat src;
    string targetWin;
};

struct GaussianFilterData {
    int kernelSize = 1;
    int sigma = 1;
    Mat src;
    string targetWin;
};

struct BilateralFilterData {
    int sigmaRange = 1;
    int sigmaSpace = 1;
    Mat src;
    string targetWin;
};

void showHistogram(std::vector<cv::Mat>& hists);
void partOne();
void partTwo();

void updateMFWindow(int _, void* data);
void updateGFWindow(int _, void* data);
void updateBFWindow(int _, void* data);

// The lab is composed of two functions, each function is divided into phases showing different results.
// To move to the next phase press any key.

char* IMG_PATH;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        cout << "USAGE: " << argv[0] << " IMG_PATH" << endl;
        cout << "IMG_PATH: path to the lab image" << endl;

        return 1;
    }

    IMG_PATH = argv[1];

    partOne();
    partTwo();

    waitKey(0);
    destroyAllWindows();
    return 0;
}

// Histogram Equalization
void partOne() {
    Mat img = imread(IMG_PATH, IMREAD_COLOR);

    namedWindow("Before - RGB", WINDOW_NORMAL);
    imshow("Before - RGB", img);

    // Print BGR histograms
    Mat channels[3];
    split(img, channels);
    vector<Mat> histograms(3);

    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    for (int i=0; i<3; i++)
        calcHist( &channels[i], 1, nullptr, Mat(), histograms[i], 1, &histSize, &histRange, true, false );

    showHistogram(histograms);

    waitKey(0);
    destroyAllWindows();

    // Equalize histograms (BGR) and show results
    for (int i=0; i<3; i++) {
        equalizeHist(channels[i], channels[i]);
        calcHist( &channels[i], 1, nullptr, Mat(), histograms[i], 1, &histSize, &histRange, true, false );
    }
    showHistogram(histograms);

    // Show equalized image
    namedWindow("Equalized RGB", WINDOW_NORMAL);
    merge(vector<Mat>({ channels[0], channels[1], channels[2] }), img);
    imshow("Equalized RGB", img);

    waitKey(0);
    destroyAllWindows();

    // Repeat with HSV color space for each channel
    Mat hsv = imread(IMG_PATH);
    namedWindow("Before - HSV", WINDOW_NORMAL);
    imshow("Before - HSV", hsv);

    vector<Mat> channelsHSV(3);
    cvtColor(hsv, hsv, COLOR_BGR2HSV);
    split(hsv, channelsHSV);

    // Equalize one channel and merge for each channel
    for (int i=0; i<3; i++) {
        Mat tmpCopy = channelsHSV[i].clone();
        equalizeHist(channelsHSV[i], channelsHSV[i]);
        merge(channelsHSV, hsv);

        cvtColor(hsv, hsv, COLOR_HSV2BGR);

        string winName = "Equalized channel " + to_string(i);
        namedWindow(winName, WINDOW_NORMAL);
        imshow(winName, hsv);

        // restore previous
        channelsHSV[i] = tmpCopy;
    }

    waitKey(0);
    destroyAllWindows();
}

// Image Filtering
void partTwo() {
    Mat img = imread(IMG_PATH);

    // Median Filtere
    string medianWinName("Median Filter");
    MedianFilterData MFdata;
    MFdata.src = img.clone();
    MFdata.targetWin = medianWinName;

    namedWindow(medianWinName, WINDOW_NORMAL);
    createTrackbar("MF Sernel Size", medianWinName, &MFdata.kernelSize, 50, updateMFWindow, (void*) &MFdata);
    imshow(medianWinName, MFdata.src);

    waitKey(0);
    destroyAllWindows();

    //Gaussian Blur Filter
    string gaussianWinName("Gaussian Filter");
    GaussianFilterData GFdata;
    GFdata.src = img.clone();
    GFdata.targetWin = gaussianWinName;
    
    namedWindow(gaussianWinName, WINDOW_NORMAL);
    createTrackbar("GF Kernel Size", gaussianWinName, &GFdata.kernelSize, 100, updateGFWindow, (void*) &GFdata);
    createTrackbar("GF Sigma", gaussianWinName, &GFdata.sigma, 100, updateGFWindow, (void*) &GFdata);

    imshow(gaussianWinName, GFdata.src);

    waitKey(0);
    destroyAllWindows();

    // Bilateral Filter
    string bilateralWinName("Bilateral Filter");
    BilateralFilterData BFdata;
    BFdata.src = img.clone();
    BFdata.targetWin = bilateralWinName;

    namedWindow(bilateralWinName, WINDOW_NORMAL);
    createTrackbar("BF  Sigma Range", bilateralWinName, &BFdata.sigmaRange, 100, updateBFWindow, (void*) &BFdata);
    createTrackbar("BF  Sigma Space", bilateralWinName, &BFdata.sigmaSpace, 7, updateBFWindow, (void*) &BFdata);

    imshow(BFdata.targetWin, BFdata.src);

    waitKey(0);
    destroyAllWindows();
}

void updateMFWindow(int _, void* data) {
    MedianFilterData MFdata = *((MedianFilterData*) data);
    if (MFdata.kernelSize%2 != 1) return; // ignore odd kernel size
    Mat result;
    medianBlur(MFdata.src, result, MFdata.kernelSize);
    imshow(MFdata.targetWin, result);
}

void updateGFWindow(int _, void* data) {
    GaussianFilterData GFdata = *((GaussianFilterData*) data);
    if (GFdata.kernelSize%2 != 1) return; // ignore odd kernel size
    Mat result;
    GaussianBlur(GFdata.src, result, Size(GFdata.kernelSize, GFdata.kernelSize), GFdata.sigma, GFdata.sigma);
    imshow(GFdata.targetWin, result);
}

void updateBFWindow(int _, void* data) {
    BilateralFilterData BFdata = *((BilateralFilterData*) data);
    Mat result;
    int kernelSize = 6 * BFdata.sigmaSpace;
    bilateralFilter(BFdata.src, result, kernelSize, BFdata.sigmaRange, BFdata.sigmaSpace);
    imshow(BFdata.targetWin, result);
}

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
void showHistogram(std::vector<cv::Mat>& hists)
{
    // Min/Max computation
    double hmax[3] = {0,0,0};
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    std::string wname[3] = { "blue", "green", "red" };
    cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                             cv::Scalar(0,0,255) };

    std::vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
        {
            cv::line(
                    canvas[i],
                    cv::Point(j, rows),
                    cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
                    hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
                    1, 8, 0
            );
        }

        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}