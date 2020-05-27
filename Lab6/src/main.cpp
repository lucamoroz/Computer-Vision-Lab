#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

/**
 Global variables
 */
vector<Mat> obj_images;




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






int main(int argc, char* argv[]) {
    
    string data_folder(argv[1]);
    string video_path = data_folder + "/video.mov";
    string images_folder = data_folder + "/objects/*.png";
    
    //load images using the path
    loadImages(images_folder);
    
    
    
    VideoCapture cap(video_path); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        imshow("Video", frame);
        if(waitKey(30) >= 0) break;
    }
  

    return 0;
}
