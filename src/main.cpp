#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/types_c.h"
#include "utils.h"

using namespace std;
using namespace cv; 

struct comparey
{
  bool operator () (cv::Rect a1, cv::Rect a2) 
  {
    if (a1.y < a2.y)
      return true;
    if (a1.y == a2.y)
      if (a1.x < a2.x)
        return true;
    return false;
  }
};


Mat  rotateImg(Mat src, double angle)
{
  // get rotation matrix for rotating the image around its center in pixel coordinates
  cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle, center not relevant
  cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
  rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

  cv::Mat dst;
  cv::warpAffine(src, dst, rot, bbox.size());

  return dst;
}

char digitToChar(char ch)
{
  if (ch == '0')
    return 'o';

  if (ch == '2')
    return 'z';

  if (ch == '1')
    return 'l';

  if (ch == '4')
    return 'g';
}

string postProcess(string str)
{
  int charCnt = 0;
  int digitCnt = 0;
  string output = str;

  for (int i = 0; i < str.length(); i++)
  {
    if (str[i] > 64)
      charCnt++;
    else
      digitCnt++;

  }

  if (charCnt >= digitCnt)
  {
    for (int i = 0; i < str.length(); i++)
    {
      if (str[i] < 64)
        output[i] = digitToChar(str[i]);
    }
  }

  return output;
}

int main(int argc, char** argv)
{

  if (argc < 4) // arguments should not be less then 2
  {   
    cout << " ERROR in arguments" << endl;
    cout << " Run : ./binary <image-name> <model-file> <output-path>" << endl;
    return -1; 
  }   

  cv::dnn::Net OCRNetwork = Utils::loadModel(argv[2]);

  // load the color image 
  Mat colorImg = imread(argv[1], 1); 

  // convert the image to grayscale
  Mat grayImg;
  cvtColor(colorImg, grayImg, CV_BGR2GRAY);

  namedWindow("original", 0);
  resizeWindow("original", 500,500);
  imshow("original", colorImg);
  namedWindow("gray", 0);
  resizeWindow("gray", 500,500);
  imshow("gray", grayImg);
  waitKey(0);

  /**
   * Algorithm steps:
   * 1. Threshold the image & binarize it 
   *    (also keep the inverse thresholded image for contour finding)
   * 2. Histogram on the y-projrction to find lines
   * 3. Dilate the images of lines and extract the words
   * 4. from each word extract the characters
   * 5. classify each character using CNN
   * 6. concatenate the char to form word
   * 7. concatenate the words to form line
   * 8. concatenate the lines to form the document text
   */

  Mat normImg = Utils::normalizeImg(grayImg);
  Mat threshImg = Utils::thresholdImg(normImg);
  Mat invThreshImg = Utils::invThresholdImg(normImg);

  float slope = Utils::findSlantAngle(threshImg);

  if (slope > -5.0 && slope < 5.0)
    slope = 0.0;

  Mat corrColorImg = rotateImg(colorImg, slope);
  Mat corrGrayImg = rotateImg(grayImg, slope);
  Mat corrInvImg = rotateImg(invThreshImg, slope);
  Mat corrThreshImg;
  bitwise_not(corrInvImg, corrThreshImg);

  imwrite("thresh.jpg", corrInvImg);
  namedWindow("Pre Processed Image", 0);
  resizeWindow("Pre Processed Image", 500, 500);
  imshow("Pre Processed Image", corrInvImg);
  waitKey(0);

  vector<int> hist;
  vector<cv::Point> lines;
  Mat linesImg = corrColorImg.clone();
  vector<Rect> linesROI;
  int totalLines;

#if 0
  hist = Utils::histogram(corrInvImg);
  lines = Utils::extractLines(hist);

  // Show the extracted line
  totalLines = lines.size();

  for (int i = 0; i < totalLines; i++)
  {
    Rect r(0, lines[i].x, corrColorImg.cols, lines[i].y - lines[i].x);

    if (r.y - 4 >= 0)
    {
      r.y -= 4;
      r.height += 4;
    }
    if (r.y + r.height + 4 < corrColorImg.rows)
    {
      r.height += 4;
    }


    if (i > 0)
    {
      if (Utils::checkOverlapping(linesROI[i-1], r) > 0.0 ||
          Utils::checkOverlapping(r, linesROI[i-1]) > 0.0)
      {
        linesROI[i-1] = linesROI[i-1] | r;
      }
      else
        linesROI.push_back(r);
    }
    else
      linesROI.push_back(r);

  }

#else
  Mat kernel1;

  kernel1 = cv::getStructuringElement(MORPH_RECT, cv::Size(corrColorImg.cols/10, 1));
  Mat tmpLines = corrInvImg.clone();
  Mat dilLines;
  cv::dilate(tmpLines, dilLines, kernel1);

  vector< vector<cv::Point> > linesContours;
  cv::findContours(dilLines, linesContours, RETR_LIST, CHAIN_APPROX_SIMPLE);
  vector<cv::Rect> tmpLineSegs;

  for (int j = 0; j < linesContours.size(); j++)
  {
    Rect x = cv::boundingRect(linesContours[j]);

    if (x.x -2 >= 0)
    {
      x.x -= 2;
      x.width += 2;
    }
    if (x.y -2 > 0)
    {
      x.y -= 2;
      x.height += 2;
    }
    if (x.x + x.width + 2 <= corrColorImg.cols)
      x.width += 2;
    if (x.y + x.height + 2 <= corrColorImg.rows)
      x.height +=2;

    if (x.width < 10 || x.height < 10)
      continue;

    tmpLineSegs.push_back(x);
  }

  linesROI = Utils::rejectSegments(tmpLineSegs, 0.8);



#endif

  sort(linesROI.begin(), linesROI.end(), comparey());

  totalLines = linesROI.size();
  //cout << "Total Lines :: " << totalLines << endl;

  for (int i = 0; i < totalLines; i++)
    cv::rectangle(linesImg, linesROI[i], cv::Scalar(0,0,255));

  // for each line extract words 
  vector< vector<Rect> > words;
  Mat wordImg = corrColorImg.clone();

  for (int i = 0; i < totalLines; i++)
  {
    Mat tmpImg(corrInvImg, linesROI[i]);
    Mat lineImg = tmpImg.clone();

    // dilating the line image in the horizontal direction
    Mat dilatedImg;

    Mat kernel;

    if (corrColorImg.cols > 1700)
      kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(9, 1));
    else if (corrColorImg.cols > 1300)
      kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(7, 1));
    else if (corrColorImg.cols > 1000)
      kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(5, 1));
    else
      kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 1));

    cv::dilate(lineImg, dilatedImg, kernel);

    vector< vector<cv::Point> > contours;
    cv::findContours(dilatedImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    int totalContours = contours.size();
    Mat tmpColorImg(corrColorImg, linesROI[i]);
    //cout << "Total words :: " << totalContours << endl;
    vector<cv::Rect> tmpWords;

    for (int j = 0; j < totalContours; j++)
    {
      Rect x = cv::boundingRect(contours[j]);

      if (x.x > 0)
      {
        x.x -= 1;
        x.width += 1;
      }
      if (x.y > 0)
      {
        x.y -= 1;
        x.height += 1;
      }
      if (x.x + x.width + 1 <= tmpColorImg.cols)
        x.width += 1;
      if (x.y + x.height + 1 <= tmpColorImg.rows)
        x.height +=1;

      tmpWords.push_back(x);
    }

    vector<cv::Rect> finalWords = Utils::joinSegments(Utils::rejectSegments(tmpWords, 0.8));
    words.push_back(finalWords);

    //cout << "Final words :: " << finalWords.size() << endl;

    for (int j = 0; j < finalWords.size(); j++)
    {
      cv::Rect wordRect = finalWords[j];
      wordRect.x += linesROI[i].x; 
      wordRect.y += linesROI[i].y; 
      cv::rectangle(wordImg, wordRect, cv::Scalar(255,0,0));
    }

    imshow("line", lineImg);
    //imshow("kernel", kernel);
    imshow("dilated img", dilatedImg);
    namedWindow("word", 0);
    resizeWindow("word", 500,500);
    imshow("word", wordImg);
    waitKey(0);
  }  

  // Exteaction of the characters 
  Mat characterImg = corrThreshImg.clone();
  Mat charTmpImg = corrColorImg.clone();

  cout << "\n\n Digitized Output: \n" << endl; 
  for (int i = 0; i < totalLines; i++)
  {
    Mat tmpImg(corrInvImg, linesROI[i]);
    Mat lineImg = tmpImg.clone();

    int totalWords = words[i].size();
    string lineTxt;

    for(int j = 0; j < totalWords; j++)
    {
      string wordTxt;
      Mat wordImg(lineImg, words[i][j]);
      vector< vector<cv::Point> > contours;
      cv::findContours(wordImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

      Mat tmpColorImg(corrColorImg, linesROI[i]);
      Mat tmpWordImg(tmpColorImg, words[i][j]);
      int totalContours = contours.size();
      vector<cv::Rect> characters;
      //cout << "Total Chars :: " << totalContours << endl;

      for (int k = 0; k < totalContours; k++)
      {
        Rect x = cv::boundingRect(contours[k]);
        if (x.x > 0)
        {
          x.x -= 1;
          x.width += 1;
        }
        if (x.y > 0)
        {
          x.y -= 1;
          x.height += 1;
        }
        if (x.x + x.width + 1 <= tmpColorImg.cols)
          x.width += 1;
        if (x.y + x.height + 1 <= tmpColorImg.rows)
          x.height +=1;

        characters.push_back(x);
      }

      vector<cv::Rect> finalChars = Utils::joinSegments(Utils::rejectSegments(characters), 2);
      //vector<cv::Rect> finalChars = Utils::rejectSegments(characters, 0.5);
      //cout << "Total final Chars :: " << finalChars.size() << endl;

      for (int k = 0; k < finalChars.size(); k++)
      {
        cv::Rect charSeg = finalChars[k];
        charSeg.x += (linesROI[i].x + words[i][j].x);
        charSeg.y += (linesROI[i].y + words[i][j].y);

        if (charSeg.x < 0)
        {
          charSeg.width += charSeg.x;
          charSeg.x = 0;
        }
        if (charSeg.y < 0)
        {
          charSeg.height += charSeg.y;
          charSeg.y = 0;
        }
        if (charSeg.x + charSeg.width > corrInvImg.cols)
          charSeg.width = corrInvImg.cols - charSeg.x;
        if (charSeg.y + charSeg.height > corrInvImg.rows)
          charSeg.height = corrInvImg.rows - charSeg.y;

        Mat charImg(characterImg, charSeg);
        //Mat charImg(corrGrayImg, charSeg);
        char ch = Utils::predictChar(charImg, OCRNetwork);
        wordTxt.push_back(ch);
        cv::rectangle(charTmpImg, charSeg, cv::Scalar(255,0,0));
      }

      wordTxt = postProcess(wordTxt);
      lineTxt += " " + wordTxt;

    }

    cout << lineTxt << endl;
  }

  cout << "\n\n";

  namedWindow("lines", 0);
  resizeWindow("lines", 500,500);
  imshow("lines", linesImg);
  namedWindow("words", 0);
  resizeWindow("words", 500,500);
  imshow("words", wordImg);
  namedWindow("char", 0);
  resizeWindow("char", 500,500);
  imshow("char", charTmpImg);
  waitKey(0);

  char fileNameLine[64];
  char fileNameWord[64];
  char fileNameChar[64];

  sprintf(fileNameLine, "%s_lines.jpg", argv[3]);
  sprintf(fileNameWord, "%s_words.jpg", argv[3]);
  sprintf(fileNameChar, "%s_chars.jpg", argv[3]);

  imwrite(fileNameLine, linesImg);
  imwrite(fileNameWord, wordImg);
  imwrite(fileNameChar, charTmpImg);

  return 0;
}
