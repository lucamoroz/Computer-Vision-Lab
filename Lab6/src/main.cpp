#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


/**
 Global variables
 */
vector<Mat> obj_images;
vector<vector<KeyPoint>> obj_keypoints;
vector<Mat> obj_descriptors;
vector<KeyPoint> video_keypoints;
Mat video_descriptor;
vector<vector<DMatch>> matches;
vector<Mat> H;
vector<vector<Point2f>> h_src(4);
vector<vector<Point2f>> h_dst(4);


void loadImages(string images_folder)
{
    vector<String> img_names;
    //to save in the vector all the images' path
    glob(images_folder, img_names);
    
    for (int i=0;i<img_names.size();i++)
    {
        obj_images.push_back(imread(img_names[i]));
    }
}

vector<KeyPoint> extractKeypoints(Mat img)
{
    vector<KeyPoint> img_keypoints;
    
    Ptr<SIFT> detector = SIFT::create();
    detector->detect(img, img_keypoints);
    return img_keypoints;
}

Mat extractDescriptor(Mat img, vector<KeyPoint> img_keypoints)
{
    Mat img_descriptor;
    
    Ptr<SIFT> detector = SIFT::create();
    detector->compute(img, img_keypoints, img_descriptor);
    return img_descriptor;
}

vector<DMatch> findMatches(Mat obj_descr, Mat video_descr)
{
    BFMatcher matcher = BFMatcher();
    vector<DMatch> curr_match;
    matcher.match(obj_descr, video_descr, curr_match);
    
    return curr_match;

}


int main(int argc, char* argv[]) {
    
    string data_folder(argv[1]);
    string video_path = data_folder + "/video.mov";
    string images_folder = data_folder + "/objects/*.png";
    
    //load images using the path
    loadImages(images_folder);
    //extract features
    for (int i = 0; i < obj_images.size(); i++)
    {
        obj_keypoints.push_back(extractKeypoints(obj_images[i]));
        obj_descriptors.push_back(extractDescriptor(obj_images[i], obj_keypoints[i]));
    }
    
    VideoCapture cap(video_path); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    int current_frame = 0;
    
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        
        
        if(current_frame == 0)
        {
            video_keypoints = extractKeypoints(frame);
            video_descriptor = extractDescriptor(frame, video_keypoints);
            
            for (int i = 0; i < obj_images.size(); i++)
            {
                matches.push_back(findMatches(obj_descriptors[i], video_descriptor));
            
                // Further refine matches through RANSAC
                
                vector<uint8_t> mask;

                for (int j = 0 ; j < matches[i].size(); j++) {
                    h_src[i].push_back(obj_keypoints[i][matches[i][j].queryIdx].pt);
                    h_dst[i].push_back(video_keypoints[matches[i][j].trainIdx].pt);
                }
                H.push_back(findHomography(h_src[i], h_dst[i], RANSAC, 3, mask));
               //adjust destination points considered into the destination(video frame)
                vector<Point2f> temp;
                for (int j = 0; j < h_src[i].size(); j++) {
                    if (!mask[j]) continue;
                    temp.push_back(h_dst[i][j]);
                }
                h_dst[i] = temp;
            }


        }
        else
        {
                //*******your implementation********
        }
        imshow("Video", frame);
        if(waitKey(30) >= 0) break;
        
        current_frame++;
    }
  

    return 0;
}
