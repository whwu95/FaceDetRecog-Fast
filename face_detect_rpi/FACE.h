#ifndef FACE_FACE_H
#define FACE_FACE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
class FACE {

public:

    FACE();
    FACE(const std::string &dir);
    ~FACE();

    void detect(const cv::Mat& img, std::vector<cv::Mat>& cropFace);


    std::vector<cv::Rect> bounding_box_;
    
};


#endif //FACE_FACE_H
