#include <iostream>
#include "FACE.h"
#include "opencv2/opencv.hpp"
#include <string.h>
#include <fstream>
using namespace std;
using namespace cv;

void GetName(const  std::string& s, std::string &n)
{
	std::size_t found = s.find_last_of("/");
	std::string tmp_str = s.substr(0, found);
	found = tmp_str.find_last_of("/");
	n = tmp_str.substr(found + 1);
}

int main(int argc, char** argv)  {

	string pt[4] = {
		"../model/step1.prototxt",
		"../model/step2.prototxt",
		"../model/step3.prototxt",
		"../model/face.prototxt"
	};

	string cm[4] = {
		"../model/step1.caffemodel",
		"../model/step2.caffemodel",
		"../model/step3.caffemodel",
		"../model/face.caffemodel"
	};

	vector<string> model_file(pt, pt + 4);
	vector<string> trained_file(cm, cm + 4);
	FACE face(model_file, trained_file);

	string img_list_dir = argv[1];//image list
	string feature_dir = argv[3];//feature
	string name_dir = argv[4];//name

	string num = argv[2];
	int img_num = atoi(num.c_str());

	ifstream img_list_file(img_list_dir.c_str());
	string img_path, name;

	for (int n = 0; n<img_num; n++)
	{
			img_list_file >> img_path;
			cout << n << endl;

			Mat img = cv::imread(img_path, 1);
			if (img.empty())
		{
				// write feature to txt
				std::ofstream feature_file(feature_dir.c_str(), ios::in | ios::app);
				feature_file << 0;
				feature_file << endl;
				feature_file.close();
		}
		
			else
		{
			vector<Mat> cropFace;
			vector<float> feature;
			face.detect(img, cropFace);
			if (cropFace.size() > 0)
			{
				    
				    int ind = 0;
				    int point_num = 0;

				    for(int i = 0; i < face.bounding_box_.size(); i++)
				    {
				    	int area1 = face.bounding_box_[i].height * face.bounding_box_[i].width;
				    	for (int j = i +1; j < face.bounding_box_.size(); j++)
				    	{
				    		cv::Mat temp;
				    		int area2 = face.bounding_box_[j].height * face.bounding_box_[j].width;
				    		if (area2 > area1) {
				    			temp = cropFace[j];
				    			cropFace[j] = cropFace[i];
				    			cropFace[i]= temp;
				    		}
				    	}
					}
					
					

	                for ( int k = 0; k < 96; k++ ) 
	                {
				    	for ( int p = 0; p < 112; p++) 
				    	{
				    		uchar t = 127.5;
				    		uchar c0 = cropFace[0].at<cv::Vec3b>(k, p)[0];
				    		uchar c1 = cropFace[0].at<cv::Vec3b>(k, p)[1];
				    		uchar c2 = cropFace[0].at<cv::Vec3b>(k, p)[2];
				    		if (c0 == t && c1 == t && c2 ==t) 
				    		++point_num;
				    	}
				    }

				     if (point_num > 400 && face.bounding_box_.size() == 1)
				     {
				     	ind = 0;
				     }
                                      if (point_num > 400 && face.bounding_box_.size() > 1)
				     {
				     	ind = 1;
				     }


					face.extractFeature(cropFace[ind], feature);
					// write feature to txt
					std::ofstream feature_file(feature_dir.c_str(), ios::in | ios::app);
					std::ostream_iterator<float> feature_iterator(feature_file, " ");
					std::copy(feature.begin(), feature.end(), feature_iterator);
					feature_file << endl;
					feature_file.close();
				
			}

			else
			{
					// write feature to txt
					std::ofstream feature_file(feature_dir.c_str(), ios::in | ios::app);
					feature_file << 0;
					feature_file << endl;
					feature_file.close();
			}
		}

			GetName(img_path, name);
			// write name to txt
			std::ofstream name_file(name_dir.c_str(), ios::in | ios::app);
			name_file << name << endl;
			name_file.close();
	
	}
	
	img_list_file.close();
	return 0;
}


