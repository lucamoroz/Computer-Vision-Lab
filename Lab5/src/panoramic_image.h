#ifndef LAB5_PANORAMIC_IMAGE_H
#define LAB5_PANORAMIC_IMAGE_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <panoramic_utils.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


class PanoramicImage {
    Ptr<SIFT> extractor = SIFT::create();
    // Enable crossCheck for consistency
    BFMatcher matcher = BFMatcher(NORM_L2, true);

public:
    vector<Mat> images;
    vector<vector<KeyPoint>> keypoints;
    vector<vector<DMatch>> matches;
    vector<int> x_translations;


    PanoramicImage(string images_folder_path, int FOV) {
        images = vector<Mat>();

        vector<String> images_path;
        glob(images_folder_path + "/*.*",images_path);

        for (const auto& path : images_path) {
            Mat projected_img = PanoramicUtils::cylindricalProj(imread(path), FOV / 2);
            images.push_back(projected_img);
        }
    }

    /**
     * Extract features for each image.
     * Result will be available at PanoramicImage.keypoints,
     * where keypoints[i] will contains the features found on image i.
    */
    PanoramicImage& findKeypoints() {
        keypoints = vector<vector<KeyPoint>>();
        Mat gray;

        vector<KeyPoint> kp;

        for (auto & image : images) {
            extractor->detect(image, kp);
            keypoints.push_back(kp);

            /*
            // DEBUG - show keypoints
            Mat temp;
            drawKeypoints(image, kp, temp);

            namedWindow("Keypoints", WINDOW_NORMAL);
            imshow("Keypoints", temp);

            waitKey(0);
            destroyAllWindows();
             */
        }

        return *this;
    }

    /**
     * Extract matches for each pair of consecutive images.
     * Result will be available at PanoramicImage.matches,
     * where matches[i] will contains the matches from image i to image i+1.
    */
    PanoramicImage& findMatches() {
        matches = vector<vector<DMatch>>();

        vector<DMatch> curr_matches;
        vector<KeyPoint> kp1, kp2;
        Mat ds1, ds2;

        for (int i = 0; i < keypoints.size() - 1; i++) {
            kp1 = keypoints[i];
            kp2 = keypoints[i+1];

            // Get descriptors
            extractor->compute(images[i],kp1, ds1);
            extractor->compute(images[i+1],kp2, ds2);

            matcher.match(ds1, ds2, curr_matches);
            matches.push_back(curr_matches);

            /*
            // DEBUG - show matches
            Mat matchImg;
            drawMatches(images[i], kp1, images[i+1], kp2, matches, matchImg);
            namedWindow("matches");
            imshow("matches", matchImg);
            waitKey(0);
            destroyAllWindows();
             */

        }

        return *this;
    }


    /**
     * Compute translations from iamge i to image i+1.
     * Result will be available at PanoramicImage.x_translations.
     * Note that outliers will be discarded through RANSAC.
     * @param match_filter_ratio Considering the minimum distance between a pair of matches, if the distance
     * of a match is greater than match_filter_ratio * min_pair_distance then the match is excluded.
     */
    PanoramicImage& refineAndComputeTranslations(float match_filter_ratio) {
        x_translations = vector<int>();

        for (int i = 0; i < matches.size(); i++) {

            // Find min match distance and use it to refine matches
            float min_match_dist = INFINITY;
            for (const auto& match : matches[i]) {
                if (match.distance < min_match_dist)
                    min_match_dist = match.distance;
            }

            // Remove matches that don't satisfy the criteria
            auto criteria =[&](DMatch m) { return m.distance > min_match_dist * match_filter_ratio; };
            matches[i].erase(remove_if(matches[i].begin(), matches[i].end(), criteria), matches[i].end());

            /*
            // DEBUG - show matches
            Mat matchImg;
            drawMatches(images[i], keypoints[i], images[i+1], keypoints[i+1], matches[i], matchImg);
            namedWindow("matches");
            imshow("matches", matchImg);
            waitKey(0);
            destroyAllWindows();
            */

            // Further refine matches through RANSAC
            vector<Point2f> h_src;
            vector<Point2f> h_dst;
            vector<uint8_t> mask;

            for (const auto &match : matches[i]) {
                h_src.push_back(keypoints[i][match.queryIdx].pt);
                h_dst.push_back(keypoints[i+1][match.trainIdx].pt);
            }

            findHomography(h_src, h_dst, RANSAC, 3, mask);

            // Estimate x, y translation with the inliers found with the RANSAC method
            float cum_dx = 0;
            int n_inliers = 0;
            for (int j = 0; j < h_src.size(); j++) {
                if (!mask[j]) continue;
                n_inliers++;
                cum_dx += h_dst[j].x - h_src[j].x;
                // cout << "from: " << h_src[j] << "  to: " << h_dst[j] << endl;
            }

            // Append avg x, y translation
            x_translations.push_back(cum_dx / n_inliers);
        }

        return *this;
    }

    /**
     * @return Composed image
     */
    Mat composePanoramicImage() {
        int img_rows = images[0].rows;
        int img_cols = images[0].cols;

        // Prepare panoramic image
        int panoramic_width = images[0].cols;
        for (auto &dx : x_translations)
            panoramic_width += abs(dx);

        Mat panoramic = Mat(img_rows, panoramic_width, images[0].type());

        // Init with the first image, then concatenate portions of consecutive images
        images[0].copyTo(panoramic(Rect(0, 0, img_cols, img_rows)));

        // Start composing from right if the translations are positive - ie pictures taken with counterclockwise direction
        int curr_x = x_translations[0] > 0 ? panoramic_width - img_cols : img_cols;

        for (int i = 0; i < x_translations.size(); i++) {
            // Portion of image to copy depends on the direction on which pictures are taken
            Rect curr_img_roi;
            if (x_translations[i] < 0) // If clockwise direction: right portion of the current image, otherwise left portion
                curr_img_roi = Rect(img_cols + x_translations[i], 0, -x_translations[i], img_rows);
            else
                curr_img_roi = Rect(0, 0, x_translations[i], img_rows);

            Rect panoramic_roi = Rect(curr_x, 0, abs(x_translations[i]), panoramic.rows);

            images[i+1](curr_img_roi).copyTo(panoramic(panoramic_roi));

            curr_x -= x_translations[i];
        }

        return panoramic;
    }

};


#endif //LAB5_PANORAMIC_IMAGE_H
