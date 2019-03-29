#include <iostream>
#include "FACE.h"
#include "opencv2/opencv.hpp"
#include <string.h>


using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    VideoCapture cap(1);//PC-1.RPI-0
	Mat image;
	Mat edges;
    vector<Mat> cropFace;

    if(!cap.isOpened()){
		cout<<"Don't open camera"<<endl;
		return -1;	
	}
    
    while(1)
   	{
		cap>>image;
		//cvtColor(image,edges,CV_BGR2GRAY);
		//imshow("camera",image);
		FACE face;
    	face.detect(image, cropFace);
 		waitKey(30);
   	}

	return 0;
}
