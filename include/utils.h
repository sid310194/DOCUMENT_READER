#ifndef _UTIL_
#define _UTIL_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv; 

namespace Utils
{
  float checkOverlapping(cv::Rect r1, cv::Rect r2);

  Mat normalizeImg(Mat img);

  Mat thresholdImg(Mat img);

  Mat invThresholdImg(Mat img);

  vector<int> histogram(Mat img);

  vector<cv::Point> extractLines(vector<int> hist);

  vector<cv::Rect> rejectSegments(vector<cv::Rect> words, 
                                  float thresh = 0.75);

  vector<cv::Rect> joinSegments(vector<cv::Rect> words,
                                int mode = 1);

  cv::dnn::Net loadModel(string modelFile);

  char predictChar(Mat img, cv::dnn::Net network);

  float findSlantAngle(Mat img);
};

#endif
