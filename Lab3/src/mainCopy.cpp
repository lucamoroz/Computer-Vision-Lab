#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

const char* IMG_PATH = "../data/lena.png";

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

void onMFKernelSizeChange(int kSize, void* data);
void updateMFWindow(MedianFilterData data);

void onGFKernelSizeChange(int kSize, void* data);
void onGFSigmaChange(int sigma, void* data);
void updateGFWindow(GaussianFilterData data);

void onBFSigmaRangeChange(int sigmaRange, void* data);
void onBFSigmaSpaceChange(int sigmaSpace, void* data);
void updateBFWindow(BilateralFilterData data);

// The lab is divided into two functions, each function is divided into phases showing different results.
// To move to the next phase press any key.
int main() {

    // partOne();
    partTwo();
    waitKey(0);
    destroyAllWindows();
    return 0;
}

// Histogram Equalization
void partOne() {
    Mat img = imread(IMG_PATH, IMREAD_COLOR);

    namedWindow("Before");
    imshow("Before", img);

    // Print BGR histograms
    Mat channels[3];
    split(img, channels);
    vector<Mat> histograms(3);

    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    for (int i=0; i<3; i++) {
        Mat histogram;
        calcHist( &channels[i], 1, 0, Mat(), histogram, 1, &histSize, &histRange, true, false );
        histograms[i] = histogram;
    }

    showHistogram(histograms);
    waitKey(0);
    destroyAllWindows();

    // Equalize histograms (BGR) and show results
    for (int i=0; i<3; i++) {
        equalizeHist(channels[i], channels[i]);
        Mat histogram;
        calcHist( &channels[i], 1, 0, Mat(), histogram, 1, &histSize, &histRange, true, false );
        histograms[i] = histogram;
    }
    showHistogram(histograms);

    // Show equalized image
    namedWindow("Equalized");
    vector<Mat> _tmp{channels[0], channels[1], channels[2]};
    merge(_tmp, img);
    imshow("Equalized", img);

    waitKey(0);
    destroyAllWindows();

    // Repeat with HSV color space for each channel
    Mat hsv = imread(IMG_PATH);
    namedWindow("Before", WINDOW_NORMAL);
    imshow("Before", hsv);

    vector<Mat> channelsHSV(3);
    cvtColor(hsv, hsv, COLOR_BGR2HSV);
    split(hsv, channelsHSV);

    for (int i=0; i<3; i++) {
        Mat tmpCopy = channelsHSV[i].clone();
        equalizeHist(channelsHSV[i], channelsHSV[i]);
        merge(channelsHSV, hsv);

        cvtColor(hsv, hsv, COLOR_HSV2BGR);

        char winName[40];
        sprintf(winName, "Equalized channel %d", i);
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

    // Median Filter
    string medianWinName("Median Filter");
    MedianFilterData MFdata;
    MFdata.src = img.clone();
    MFdata.targetWin = medianWinName;

    namedWindow(medianWinName);
    createTrackbar("MF Sernel Size", medianWinName, &MFdata.kernelSize, 50, onMFKernelSizeChange, (void*) &MFdata);
    imshow(medianWinName, MFdata.src);

    waitKey(0);
    destroyAllWindows();

    //Gaussian Blur Filter
    string gaussianWinName("Gaussian Filter");
    GaussianFilterData GFdata;
    GFdata.src = img.clone();
    GFdata.targetWin = gaussianWinName;
    
    namedWindow(gaussianWinName);
    createTrackbar("GF Kernel Size", gaussianWinName, &GFdata.kernelSize, 100, onGFKernelSizeChange, (void*) &GFdata);
    createTrackbar("GF Sigma", gaussianWinName, &GFdata.sigma, 100, onGFSigmaChange, (void*) &GFdata);

    imshow(gaussianWinName, GFdata.src);

    waitKey(0);
    destroyAllWindows();

    // Bilateral Filter
    string bilateralWinName("Bilateral Filter");
    BilateralFilterData BFdata;
    BFdata.src = img.clone();
    BFdata.targetWin = bilateralWinName;

    namedWindow(bilateralWinName);
    createTrackbar("BF  Sigma Range", bilateralWinName, &BFdata.sigmaRange, 100, onBFSigmaRangeChange, (void*) &BFdata);
    createTrackbar("BF  Sigma Space", bilateralWinName, &BFdata.sigmaSpace, 7, onBFSigmaSpaceChange, (void*) &BFdata);

    imshow(BFdata.targetWin, BFdata.src);

    waitKey(0);
    destroyAllWindows();
}

void onMFKernelSizeChange(int kSize, void* data) {
    if (kSize%2 != 1) return; // ignore odd kernel size
    MedianFilterData MFdata = *((MedianFilterData*) data);
    MFdata.kernelSize = kSize;
    updateMFWindow(MFdata);
}

void updateMFWindow(MedianFilterData data) {
    Mat result;
    medianBlur(data.src, result, data.kernelSize);
    imshow(data.targetWin, result);
}

void onGFKernelSizeChange(int kSize, void* data) {
    if (kSize%2 != 1) return; // ignore odd kernel size
    GaussianFilterData GFdata = *((GaussianFilterData*) data);
    GFdata.kernelSize = kSize;
    updateGFWindow(GFdata);
}

void onGFSigmaChange(int sigma, void* data) {
    GaussianFilterData GFdata = *((GaussianFilterData*) data);
    GFdata.sigma = sigma;
    updateGFWindow(GFdata);
}

void updateGFWindow(GaussianFilterData data) {
    Mat result;
    GaussianBlur(data.src, result, Size(data.kernelSize, data.kernelSize), data.sigma, data.sigma);
    imshow(data.targetWin, result);
}

void onBFSigmaRangeChange(int sigmaRange, void* data) {
    BilateralFilterData BFdata = *((BilateralFilterData*) data);
    BFdata.sigmaRange = sigmaRange;
    updateBFWindow(BFdata);
}

void onBFSigmaSpaceChange(int sigmaSpace, void* data) {
    BilateralFilterData BFdata = *((BilateralFilterData*) data);
    BFdata.sigmaSpace = sigmaSpace;
    updateBFWindow(BFdata);
}

void updateBFWindow(BilateralFilterData data) {
    Mat result;
    int kernelSize = 6 * data.sigmaSpace;
    // if (kernelSize % 2 == 0)

    bilateralFilter(data.src, result, kernelSize, data.sigmaRange, data.sigmaSpace);
    imshow(data.targetWin, result);
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