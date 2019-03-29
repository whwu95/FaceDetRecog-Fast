
#include "FACE.h"
#include "havon_ffd.h"
#include <sys/time.h>

FACE::FACE(){}

//FACE::FACE(const std::string &dir) {}

FACE::~FACE(){}

void FACE::detect(const cv::Mat& img, std::vector<cv::Mat>& cropFace)
{

     timeval start, end; 
     unsigned  long t;


     std::vector<cv::Rect> rectangles;
     cv::Mat src = img;
     IplImage *gray = NULL;  
     const int rects_num = 10;
     struct square_rect rects[10];
     struct havon_xffd *ffd = havon_xffd_create(128, 128*4);

     IplImage tmp = img;
     IplImage *frame = cvCloneImage(&tmp);
     if (!gray) {
       gray = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);
     }

     cvCvtColor(frame, gray, CV_RGB2GRAY); 
 
     uint32_t num_saved = 0;
     gettimeofday(&start,NULL); 
     havon_xffd_detect(ffd, (const uint8_t *)gray->imageData, gray->width, gray->height, gray->widthStep, rects, rects_num, &num_saved);
     gettimeofday(&end,NULL); 
     t=1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
     cout<<"time is "<<t<<endl;
     cout<<"detect face num is "<<num_saved<<endl;

     for (uint32_t idx = 0; idx < num_saved; ++idx) {       
      bounding_box_.push_back(cv::Rect(rects[idx].cx-rects[idx].size/2,rects[idx].cy - rects[idx].size/2,rects[idx].size, rects[idx].size) );
      if (rects[idx].score > 5.0)
     {
		//cvCircle(frame, cvPoint(rects[idx].cx, rects[idx].cy), rects[idx].size/2, CV_RGB(255,0,0), 4,8,0);
      Mat crop = src(bounding_box_[idx]);
      //string name = "face_" + to_string(idx+1) + ".jpg";
      //cv::imwrite(name, crop);    
      cropFace.push_back(crop);
		rectangle(src, cv::Rect(bounding_box_[idx].x, bounding_box_[idx].y, bounding_box_[idx].height, bounding_box_[idx].width), cv::Scalar(0, 0, 255), 3);
	 }  

   }
     imshow("detect",src);
     //cvShowImage("xffd", frame);
     //cvWaitKey(0);
     cvReleaseImage(&gray);
     cvReleaseImage(&frame);
}




