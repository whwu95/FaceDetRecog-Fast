#pragma once
//#include "basic_header.h"
#include "facesdk.h"

typedef struct {
	int x1, y1, x2, y2;
	float score;  /**< the detection score */
} CvRectScore;
typedef struct {
	int x1, y1, x2, y2;
	float score;  /**< the detection score */
	float reg_x1, reg_y1, reg_x2, reg_y2;
} CvRectScoreReg;
typedef struct {
	int x1, y1, x2, y2;
	float score;  /**< the detection score */
	float reg_x1, reg_y1, reg_x2, reg_y2;
	cv::Point2f point[5];
} CvRectScoreRegPoint;

typedef struct {
	float x1, y1, x2, y2;
	float score;  /**< the detection score */
	cv::Point2f point[5];
}CvRectScorePoint;

class OS_API face_detector
{
public:
	face_detector();
	~face_detector();

public:
	void system_init();
	vector<CvRectScorePoint> face_detect_main(cv::Mat& image);
	void system_release();	

private:
	Net<float> * net1, *net2, *net3;
	float * scales;
	int factor_count;

	int NonMaximalSuppression(CvRectScoreReg* boxes_orig, int num, float overlap = -1);
	int NonMaximalSuppression(CvRectScoreRegPoint* boxes_orig, int num, float overlap = -1);
	int NonMaximalSuppression2(CvRectScoreRegPoint* boxes_orig, int num, float overlap = -1);
};

