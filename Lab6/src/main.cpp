#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// CONFIG
Size GAUSSIAN_BLUR_SIZE(3,3);
float GAUSSIAN_SIGMA = 1;
float RANSAC_REPROJECT_ERROR = 3;
Size LK_WIN_SIZE(17,17);
int LK_MAX_PYR_LEV = 2;

/**
 Global variables
 */
vector<Scalar> colors;
vector<Mat> obj_images;
vector<vector<KeyPoint>> obj_keypoints;
vector<Mat> obj_descriptors;
vector<vector<Point2f>> h_dst(4);


void loadImages(string images_folder) {
    vector<String> img_names;
    //to save in the vector all the images' path
    glob(images_folder, img_names);
    
    for (int i=0;i<img_names.size();i++)
    {
        obj_images.push_back(imread(img_names[i]));
    }
}

vector<KeyPoint> extractKeypoints(Mat img) {
    vector<KeyPoint> img_keypoints;
    
    Ptr<SIFT> detector = SIFT::create();
    detector->detect(img, img_keypoints);
    return img_keypoints;
}

Mat extractDescriptor(Mat img, vector<KeyPoint> img_keypoints) {
    Mat img_descriptor;
    
    Ptr<SIFT> detector = SIFT::create();
    detector->compute(img, img_keypoints, img_descriptor);
    return img_descriptor;
}

vector<DMatch> findMatches(Mat obj_descr, Mat video_descr) {
    BFMatcher matcher = BFMatcher();
    vector<DMatch> curr_match;
    matcher.match(obj_descr, video_descr, curr_match);
    
    return curr_match;
}

void colorKeypoints(Mat frame, vector<Point2f> keypoints, Scalar color) {
    for (int i = 0; i < keypoints.size(); i++)
    {
        circle(frame, keypoints[i], 3, color);
    }
}

void printRectangle(vector<Point2f> outline_points, Mat src_img, Scalar color) {
    line(src_img, outline_points[0], outline_points[1], color, 2);
    line(src_img, outline_points[1], outline_points[3], color, 2);
    line(src_img, outline_points[3], outline_points[2], color, 2);
    line(src_img, outline_points[2], outline_points[0], color, 2);
}

/**
 * @param frame starting frame on which keypoints will be detected
 * @param obj_outline_points will hold lists of 4 points wrapping each object to track
 * @param features_obj will hold a list of keypoints for each object
 */
void initTracking(Mat frame, vector<vector<Point2f>> &obj_outline_points, vector<vector<Point2f>> &features_obj) {
    vector<vector<Point2f>> h_src(4);
    vector<KeyPoint> video_keypoints;
    Mat video_descriptor;
    vector<vector<DMatch>> matches;

    video_keypoints = extractKeypoints(frame);
    video_descriptor = extractDescriptor(frame, video_keypoints);

    for (int i = 0; i < obj_images.size(); i++) {
            matches.push_back(findMatches(obj_descriptors[i], video_descriptor));
            
            vector<uint8_t> mask;

            for (int j = 0 ; j < matches[i].size(); j++)
            {
                h_src[i].push_back(obj_keypoints[i][matches[i][j].queryIdx].pt);
                h_dst[i].push_back(video_keypoints[matches[i][j].trainIdx].pt);
            }
            Mat H = findHomography(h_src[i], h_dst[i], RANSAC, RANSAC_REPROJECT_ERROR, mask);
            
            //adjust destination points considered into the frame
            vector<Point2f> temp;
            for (int j = 0; j < h_src[i].size(); j++)
            {
                if (!mask[j]) continue;
                temp.push_back(h_dst[i][j]);
            }
            features_obj.push_back(temp);

            vector<Point2f> outline_points;
            outline_points.push_back(Point2f(0,0));
            outline_points.push_back(Point2f(obj_images[i].cols, 0));
            outline_points.push_back(Point2f(0,obj_images[i].rows));
            outline_points.push_back(Point2f(obj_images[i].cols, obj_images[i].rows));

            perspectiveTransform(outline_points, outline_points, H);
            obj_outline_points.push_back(outline_points);
    }
}

/**
 * @return false if tracking failed, true otherwise
 */
bool updateKeypointsAndOutline(Mat frame1, Mat frame2, vector<Point2f> &pts_to_track, vector<Point2f> &outline_pts) {
    if (pts_to_track.empty()) {
        return false;
    }

    vector<uchar> status;
    vector<float> err;
    vector<Point2f> keypoints_destination;

    calcOpticalFlowPyrLK(frame1, frame2, pts_to_track, keypoints_destination, status, err, LK_WIN_SIZE, LK_MAX_PYR_LEV);

    vector<uint8_t> mask;
    Mat H = findHomography(pts_to_track, keypoints_destination, RANSAC, RANSAC_REPROJECT_ERROR, mask);

    // Case homography not found
    if (H.empty())
        return false;

    // Move rectangle & keypoints according with the homography
    perspectiveTransform(pts_to_track, pts_to_track, H);
    perspectiveTransform(outline_pts, outline_pts, H);

    // Filter outliers and update keypoints to track
    vector<Point2f> temp;
    for (int j = 0; j < keypoints_destination.size(); j++) {
        if (mask[j] || status[j] == 1)
            temp.push_back(keypoints_destination[j]);
        else
            cout << "one keypoint discarded" << endl;
    }
    pts_to_track = temp;
    return true;
}


int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "USAGE: $" << argv[0] << " VIDEO_PATH OBJECTS_PATH" << endl;
        cout << "VIDEO_PATH: path to the video on which objects will be detected." << endl;
        cout << "OBJECTS_PATH: path to the folder containing the objects to detect on the video." << endl;
        return 1;
    }

    string video_path(argv[1]);
    string obj_images_folder(argv[2]);

    //load images using the path
    loadImages(obj_images_folder);

    // assign a random color to each object
    RNG rng;
    for (int i = 0; i < obj_images.size(); i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }

    // Blur objects to track
    for (int i = 0; i < obj_images.size(); i++) {
        GaussianBlur(obj_images[i], obj_images[i], GAUSSIAN_BLUR_SIZE, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA);
    }

    //extract features
    for (int i = 0; i < obj_images.size(); i++)
    {
        obj_keypoints.push_back(extractKeypoints(obj_images[i]));
        obj_descriptors.push_back(extractDescriptor(obj_images[i], obj_keypoints[i]));
    }
    
    VideoCapture cap(video_path); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    bool first_frame = true;
    vector<bool> tracking_failed;
    vector<vector<Point2f>> outline_points_obj; // 4 points defining a rectangle for each obj to track
    vector<vector<Point2f>> pts_to_track_obj; // keypoints for each obj to track
    Mat curr_frame, prev_frame, curr_gray, prev_gray;

    VideoWriter video;

    while (true)
    {
        cap >> curr_frame; // get a new frame from camera
        if (curr_frame.empty())
            break;

        // Work on gray and blurred frame
        cvtColor(curr_frame, curr_gray, COLOR_BGR2GRAY);
        GaussianBlur(curr_gray, curr_gray, GAUSSIAN_BLUR_SIZE, GAUSSIAN_SIGMA, GAUSSIAN_SIGMA);

        if(first_frame) {
            initTracking(curr_frame, outline_points_obj, pts_to_track_obj);
            tracking_failed = vector<bool>(pts_to_track_obj.size(), false);
            video = VideoWriter("result.avi", VideoWriter::fourcc('M','J','P','G'), 15, Size(curr_frame.cols,curr_frame.rows));
            first_frame = false;
        } else {
            for (int i = 0; i < pts_to_track_obj.size(); i++) {

                if (tracking_failed[i])
                    continue;

                bool success = updateKeypointsAndOutline(prev_gray, curr_gray, pts_to_track_obj[i], outline_points_obj[i]);

                if (!success) {
                    tracking_failed[i] = true;
                    cout << "Tracking failed for obj: " << i << endl;
                    continue;
                }
            }
        }

        prev_frame = curr_frame.clone();
        prev_gray = curr_gray.clone();

        // print rectangles and keypoints
        for (int j = 0; j < outline_points_obj.size(); j++) {
            if (!tracking_failed[j]) {
                printRectangle(outline_points_obj[j], curr_frame, colors[j]);
                // colorKeypoints(curr_frame, pts_to_track_obj[j], Scalar(0, 255, 255));
            }
        }
        video.write(curr_frame);
    }
  
    video.release();
    cap.release();

    return 0;
}
