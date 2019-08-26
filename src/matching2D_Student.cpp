#include <numeric>
#include "matching2D.hpp"

using namespace std;

using cv::ORB;
using cv::BRISK;
using cv::AKAZE;
using cv::FastFeatureDetector;
using cv::xfeatures2d::SIFT;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;


// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING; //NORM_L2 is also aviable.....
        matcher = cv::BFMatcher::create(normType, crossCheck);
        //matcher.match(descSource, descRef, matches, Mat() );
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);        
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector< std::vector<cv::DMatch> > knn_matches;
        matcher->knnMatch( descSource, descRef, knn_matches, 2 );         
    }

    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
    double t = (double)cv::getTickCount();
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = BRISK::create(threshold, octaves, patternScale);
        // perform feature description
        //double t = (double)cv::getTickCount();    
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        
    }
    //  ---------------------------assignments-------------------------------------------------------
    else if(descriptorType.compare("FAST") == 0)
    {
        //... FAST
        int threshold = 30;
        bool nonMaxSuppression = true;
        extractor = FastFeatureDetector::create(threshold, nonMaxSuppression);
        // perform feature description
        //double t = (double)cv::getTickCount(); 
        extractor->detect(img, keypoints);       
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        //  AKAZE
        cv::Mat mask;
        extractor = AKAZE::create();
        // perform feature description
        //double t = (double)cv::getTickCount();
        extractor->detectAndCompute(img, mask, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        // ORB
        cv::Mat mask;
        extractor = ORB::create();
        // perform feature description
        //double t = (double)cv::getTickCount();
        extractor->detectAndCompute(img,mask,keypoints,descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create(true,true,22,4);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        //SIFT
        cv::Mat mask;
        int nfeatures=0;
        int nOctaveLayers=3;
        double contrastThreshold=0.04;
        double edgeThreshold=10;
        double sigma=1.6;
        extractor = SIFT::create(nfeatures,nOctaveLayers,contrastThreshold,edgeThreshold,sigma);
        // perform feature description
        //double t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }

    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permis sible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}