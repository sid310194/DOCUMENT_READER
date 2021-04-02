#include "utils.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

//#define DEBUG

using namespace std;
using namespace cv; 

struct comparex
{
  bool operator () (cv::Rect a1, cv::Rect a2) 
  {
    if (a1.x < a2.x)
      return true;
    if (a1.x == a2.x)
      if (a1.y < a2.y)
        return true;
    return false;
  }
};

char mapToChar(int classId)
{
  char ch;
  int actual;

  if (classId < 10) // 0-9
    actual = classId + 48;
  else if (classId < 36) // A-Z
    actual = classId + 65 - 10;
  else // a-z
    actual = classId + 97 - 36;

  ch = actual;

  return ch;
}

namespace Utils
{
  cv::dnn::Net loadModel(string modelFile)
  {
    cv::dnn::Net network = cv::dnn::readNet(modelFile); //, "" , "Torch");

    return network;
  }

  void softMax(Mat &img)
  {
    int rows = img.rows;
    int cols = img.cols;

    float sum = 0.0;

    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        float val = exp(img.at<float>(i, j));
        sum += val;
        img.at<float>(i, j) = val;
      }
    }

    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
        img.at<float>(i, j) /= sum;
    }

  }


  char predictChar(Mat img, cv::dnn::Net network)
  {
    char ch;
    cv::Mat blob;
    
    Mat kernel;
    kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 3));

    Mat srcImg;
    Mat image(cv::Size(64, 64), CV_8UC1, cv::Scalar(255, 255, 255));

#if 0
    cv::resize(img, srcImg, cv::Size(44,44));
    Mat tmpImg(image(cv::Rect(10,10,44,44)));
    srcImg.copyTo(tmpImg);
#else
    int newWidth, newHeight;
    if (img.cols > img.rows)
    {
      newWidth = 44;
      newHeight = newWidth * img.rows / (float)img.cols;
    }
    else
    {
      newHeight = 44;
      newWidth = newHeight * img.cols / (float)img.rows;
    }

    cv::resize(img, srcImg, cv::Size(newWidth, newHeight));
    Mat tmpImg(image(cv::Rect(32 - newWidth/2, 32 - newHeight/2 , newWidth, newHeight)));
    srcImg.copyTo(tmpImg);
#endif

#ifdef DEBUG1
    imshow("img", srcImg);
    imshow("image", image);
#endif

    Mat dilatedImg;
    cv::erode(image, dilatedImg, kernel);
#ifdef DEBUG1
    imshow("dilatedImg", dilatedImg);
#endif

    cv::dnn::blobFromImage(dilatedImg, blob, 1.0/255.0, cv::Size(64, 64));//, mean, swapRB, false);
    network.setInput(blob);
    cv::Mat prob = network.forward();
    softMax(prob);

    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    // if the conf is low
    if (confidence < 0.8)
    {
      ch = mapToChar(classId);
      //cout << "classId :: " << classId << " conf :: " << confidence <<  endl;
      //cout << "char :: " << ch << endl;
      //cout << " Again ---" << endl;
      Mat newBlob;
      cv::dnn::blobFromImage(image, newBlob, 1.0/255.0, cv::Size(64, 64));//, mean, swapRB, false);
      network.setInput(newBlob);
      prob = network.forward();
      softMax(prob);

      minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
      classId = classIdPoint.x;
    }

    ch = mapToChar(classId);
    //cout << "classId :: " << classId << " conf :: " << confidence <<  endl;
    //cout << "char :: " << ch << endl;
    //imshow("dilatedImg", dilatedImg);
    //waitKey(0);
    return ch;
  }


  float checkOverlapping(cv::Rect r1, cv::Rect r2)
  {
    float area1 = r1.width * r1.height;

    Rect intersection(0,0,0,0);
    intersection = r1 & r2;
    //cout << "int :: " << intersection.x << " " << intersection.y << 
    //  " " << intersection.width << " " << intersection.height << endl;

    float area = intersection.width * intersection.height;

    return area/area1;
  }


  float findSlantAngle(Mat img)
  {
    int x1 = 0;
    int y1 = 0;
    int x2 = img.cols - 1 ;
    int y2 = img.rows - 1 ;

    Mat kernel;
    kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 3));
    Mat erodeImg;
    cv::dilate(img, erodeImg, kernel);

    vector<cv::Point> pointVec;
    pointVec.reserve((x2-x1) * (y2-y1));

    for (int y = y1; y < y2; y++)
    {
      for (int x = x1; x < x2; x++)
      {
        if (erodeImg.at<uchar>(y, x) == 0)
        {
          pointVec.push_back(cv::Point(x, y));
        }
      }
    }

    cv::RotatedRect r;
    r = cv::minAreaRect(pointVec);
    //cout << "Angle: : " << r.angle << endl;
    cv::Rect r1 = cv::boundingRect(pointVec);
    cv::rectangle(erodeImg, r1, cv::Scalar(0,0,0), 2); 

    float slope;

    if (r.angle < -45)
      slope = r.angle + 90;
    else
      slope = r.angle;

    return slope;
  }


  Mat normalizeImg(Mat img)
  {
    Mat result;
    cv::normalize(img, result, 0, 255, NORM_MINMAX);

#ifdef DEBUG
    imshow("normalized image", result);
    waitKey(0);
#endif

    return result;
  }

  Mat thresholdImg(Mat img)
  {
    Mat result;

    cv::threshold(img, result, 180, 255, THRESH_BINARY);
    //cv::threshold(img, result, 125, 255, THRESH_OTSU);

#ifdef DEBUG
    imshow("thresh image", result);
    waitKey(0);
#endif

    return result;
  }

  Mat invThresholdImg(Mat img)
  {
    Mat result;

    cv::threshold(img, result, 180, 255, THRESH_BINARY_INV);
    //cv::threshold(img, result, 125, 255, THRESH_OTSU);

#ifdef DEBUG
    imshow("inv thresh image", result);
    waitKey(0);
#endif

    return result;
  }

  vector<int> histogram(Mat img)
  {
    vector<int> hist(img.rows);

    for (int i = 0; i < img.rows; i++)
    {
      hist[i] = 0;
      for (int j = 0; j < img.cols; j++)
      {
        if (img.at<int>(i,j) > 125)
          hist[i]++;
      }
    }

#if 0
    for (int i = 0; i < img.rows; i++)
      cout << hist[i] << " ";
    cout << endl;
    cout << img.rows << " " << img.cols << " " << hist.size() << endl; 
#endif
    return hist;
  }

  vector<cv::Point> extractLines(vector<int> hist)
  {
    vector<cv::Point> lines;

    int rows = hist.size();

    int start = -1;
    int end = -1;
    bool findStart = true;

    for (int i = 0; i < rows; i++)
    {
      if (findStart)
      {
        if (hist[i] > 20)
        {
          start = i;
          findStart = false;
        }
      }
      else
      {
        if (hist[i] <=15)
        {
          end = i;
          findStart = true;
          cv::Point pt(start, end);
          lines.push_back(pt);
        }
      }
    }

    return lines;
  }

  vector<cv::Rect> rejectSegments(vector<cv::Rect> words, 
                                  float thresh)
  {
    sort(words.begin(), words.end(), comparex());
    int totalWords = words.size();
    vector<int> status(totalWords, 1);

    for (int i = 0; i < totalWords; i++)
    {
      //cout << " i : " << words[i].x << " " << words[i].y << " " <<
      //  words[i].width << " " << words[i].height << endl;

      if (words[i].width * words[i].height <= 5 || 
          words[i].width < 3 || words[i].height < 3)
      {
        status[i] = 0;
      }

      if (status[i] == 0)
        continue;

      for (int j = 0; j < totalWords; j++)
      {
        //cout << " j : " << words[j].x << " " << words[j].y << " " <<
        //  words[j].width << " " << words[j].height << endl;
        if (status[j] == 0 || i == j)
          continue;

        if (checkOverlapping(words[i], words[j]) > thresh ||
            checkOverlapping(words[j], words[i]) > thresh)
        {
          //cout << "Rejecting" << endl;
          cv::Rect r = words[i] | words[j];
          words[j] = r;
          status[i] = 0;
          break;
        }

      }

    }

    vector<cv::Rect> result;

    for (int i = 0; i < totalWords; i++)
    {
      if (status[i] == 1)
        result.push_back(words[i]);
    }

    sort(result.begin(), result.end(), comparex());
    return result;

  }


  vector<cv::Rect> joinSegments(vector<cv::Rect> words, int mode)
  {
    vector<cv::Rect> result;

    sort(words.begin(), words.end(), comparex());
    int totalWords = words.size();
    vector<int> status(totalWords, 1);

    for (int i = 0; i < totalWords; i++)
    {
      if (status[i] == 0)
        continue;

      for (int j = 0; j < totalWords; j++)
      {
        if (status[j] == 0 || i == j)
          continue;

        if (mode == 1)
        {
          if (words[i].x <= words[j].x &&
              words[j].x <= words[i].x + words[i].width)
          {
            cv::Rect r = words[i] | words[j];
            words[i] = r;
            status[j] = 0;
          }
          else if (abs(words[i].x - (words[j].x + words[j].width)) < 3 ||
                   abs(words[j].x - (words[i].x + words[i].width)) < 3)
          {
            cv::Rect r = words[i] | words[j];
            words[i] = r;
            status[j] = 0;
          }
        }
        else
        {
          if (words[i].x <= words[j].x &&
              words[j].x + words[j].width <= words[i].x + words[i].width)
          {
            cv::Rect r = words[i] | words[j];
            words[i] = r;
            status[j] = 0;
          }
        }

      }

    }

    for (int i = 0; i < totalWords; i++)
    {
      if (words[i].width < 10 && words[i].height < 10)
        continue;

      if (status[i] == 1)
        result.push_back(words[i]);
    }

    sort(result.begin(), result.end(), comparex());

    return result;
  }

};
