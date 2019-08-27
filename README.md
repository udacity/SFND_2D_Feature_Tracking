# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.




## Mid-Term Report


### DataBuffer Optimization


2 frames buffer is requested.

```
// push image into data frame buffer
unsigned dataBufferSize = 2; <--  the number we chose.
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);
if( dataBuffer.size() > dataBufferSize )
{
	dataBuffer.erase(dataBuffer.begin()); //delet first of element
}
```



### Keypoints
Keypoints detectors like FAST, BRISK, ORB, AKAZE, and SIFT are created.
Detected keypoints within a bounding box ( in the center area of the images ) is requrested others are removed. 

```
bool bFocusOnVehicle = true;//true;
if(bFocusOnVehicle)
{
    // draw rect on image
	vector<cv::KeyPoint> filtered; 
	for(int i=0; i< keypoints.size(); i++)
    {                
		if(  ( keypoints[i].pt.x > topLeft_x)   &&  (keypoints[i].pt.x < botRight_x) ){
			if ( (keypoints[i].pt.y > topLeft_y)  &&  (keypoints[i].pt.y < botRight_y) ){
            	filtered.push_back(keypoints[i]);
            }
        }
    }
    keypoints = filtered;   
	cout<<"Size of keypoints in the bounding box = "<<keypoints.size()<<endl;
}
```

with ORB keypoints detection and AKAZE descriptor type, I did not know why the keypoints are not filtered, all detected keypoints are still there.

### Descriptors
MP4. to MP6. were finished completely.

### Performance
MP7. All keypoints has been counted accordingly.
MP8. Descriptors and matching keypoints are extracted.
MP9. All combinations extractor and matcher are tested. Running time also counted. The top 3 combinations are recored on spreadsheet.txt file.
