//#define CPU_ONLY

#ifndef FACE_FACE_H
#define FACE_FACE_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
//#include <io.h>
//#include <atltime.h>

using namespace cv;
using namespace caffe;
class FACE {

public:

    FACE();
    FACE(const std::string &dir);
    FACE(const std::vector<std::string> model_file, const std::vector<std::string> trained_file);
    ~FACE();

    void detection_TEST(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
    void detect(const cv::Mat& img, std::vector<cv::Mat>& cropFace);
    std::vector<float> Recognize(const cv::Mat& img);
    void Preprocess(const cv::Mat &img);
    void Preprocessing(const cv::Mat &img);
    void STEP1_Net();
    void STEP2_Net();
    void STEP3_Net();
    void detect_net(int i);

    void NMS_Min(std::vector<cv::Rect>& bounding_box);
    void NMS_Union(std::vector<cv::Rect>& bounding_box, float threshold);

    void Predict(const cv::Mat& img, int i);
    void Predict(const std::vector<cv::Mat> imgs, int i);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
    void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);
    void extractFeature(const cv::Mat& cropFace, std::vector<float>& feature);
    void calculateDistance(std::vector<float> output1, std::vector<float> output2,float& l2);
    
    float IoU(cv::Rect rect1, cv::Rect rect2);
    float IoM(cv::Rect rect1, cv::Rect rect2);
    void resize_img();
    void GenerateBoxs(cv::Mat img);
    void BoxRegress(std::vector<cv::Rect>& bounding_box, std::vector<cv::Rect> regression_box , int step);
    void Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h);
    cv::Mat crop(cv::Mat img, cv::Rect& rect);
    void Bbox2Square(std::vector<cv::Rect>& bounding_box);

    void img_show(cv::Mat img, std::string name);
    void img_show_T(cv::Mat img, std::string name);

    void interpolateCubic(float x, float* coeffs);
    cv::Mat crop_face(const cv::Mat& image, cv::Point2f key_pt[5], cv::Point2f base_pt[5]);

    //param for four net
    std::vector<boost::shared_ptr<Net<float> > > nets_;
    std::vector<cv::Size> input_geometry_;
    int num_channels_;

    //variable for the image
    cv::Mat img_;
    std::vector<cv::Mat> img_resized_;
    std::vector<double> scale_;
    
    //variable for the output of the neural network
    std::vector<cv::Rect> regression_box_;
    std::vector<float> regression_box_temp_;
    std::vector<cv::Rect> bounding_box_;
    std::vector<float> confidence_;
    std::vector<float> confidence_temp_;
    std::vector<std::vector<cv::Point> > alignment_;
    std::vector<float> alignment_temp_;
    
    //paramter for the threshold
    int minSize_ = 100;
    float factor_ = 0.709;
    float threshold_[3] = {0.6, 0.7, 0.7};
    float threshold_NMS_[2]={0.5, 0.7};


};


#endif //FACE_FACE_H
