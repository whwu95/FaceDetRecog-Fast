#include <iostream>
#include "FACE.h"
#include "opencv2/opencv.hpp"
#include <string.h>
#include <fstream>
using namespace std;
using namespace cv;

vector<float> vector_read_feature(string feat)
{
	vector<float> feature;
	const char *d = " ";
	char *feat_gallery = (char*)feat.data();
	char *p;
	p = strtok(feat_gallery, d);
	while (p)
	{
		feature.push_back(atof(p));
		p = strtok(NULL, d);
	}
	return feature;
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

	string db_feature_dir = argv[1];
	string test_feature_dir = argv[2];
 string num = argv[3];
	int img_num = atoi(num.c_str());
 	
	string result_dir = argv[4];

	vector<vector<float> > all_db_feature, all_test_feature;
	vector<float > db_feat, test_feat;
	std::ifstream db_feature_file, test_feature_file;

	int detect_error_num = 0;

	db_feature_file.open(db_feature_dir.c_str());
	string db_feature_str;
	while (getline(db_feature_file, db_feature_str))
	{
		db_feat = vector_read_feature(db_feature_str);
		all_db_feature.push_back(db_feat);
		if (db_feat.size() != 256)
			detect_error_num++;
	}
	db_feature_file.close();

	test_feature_file.open(test_feature_dir.c_str());
	string test_feature_str;
	while (getline(test_feature_file, test_feature_str))
	{
		test_feat = vector_read_feature(test_feature_str);
		all_test_feature.push_back(test_feat);
		if (test_feat.size() != 256)
			detect_error_num++;
	}
	test_feature_file.close();

	//
	int db_img_num = all_db_feature.size();
	int test_img_num = all_test_feature.size();

	int pass_num1[201] = { 0 };
	int pass_num2[201] = { 0 };
	int cmp_num = 0;
	
	float distance;

	for (int i = 0; i < img_num; i++)
	{
		if (all_db_feature.at(i).size() == 256 && all_test_feature.at(i).size() == 256)
		{
			face.calculateDistance(all_db_feature.at(i), all_test_feature.at(i), distance);
			float threshold_distance = 0.7;
			for (int k = 0; k <= 200; k++){
				threshold_distance += 0.005;
				if (distance < threshold_distance)
					pass_num1[k]++;
			}
		}
	}

	for (int i = img_num; i < img_num*2; i++)
	{
		if (all_db_feature.at(i).size() == 256 && all_test_feature.at(i).size() == 256)
		{
			face.calculateDistance(all_db_feature.at(i), all_test_feature.at(i), distance);
			float threshold_distance = 0.7;
			for (int k = 0; k <= 200; k++){
				threshold_distance += 0.005;
				if (distance >= threshold_distance)
					pass_num2[k]++;
			}
		}
	}
 float best_threshold = 0.7,best_accu = -1;
	std::ofstream result_file(result_dir.c_str(), ios::in | ios::app);
	result_file << "db_img_num=" << db_img_num << " ";
	result_file << "test_img_num=" << test_img_num << " ";
	result_file << "detect_error_num=" << detect_error_num << " ";
	result_file << "cmp_num=" << (img_num*2 - detect_error_num) << endl;
	float threshold_distance = 0.7;
	for (int k = 0; k <= 200; k++){
		threshold_distance += 0.005;
		float accu = ((float)(pass_num1[k] + pass_num2[k])) / (img_num*2 - detect_error_num);
		result_file << k << " threshold_distance=" << threshold_distance << " ";
		result_file << "pass_num1=" << pass_num1[k] << " ";
		result_file << "pass_num2=" << pass_num2[k] << " ";
		result_file << "accu=" << accu << endl;
   if(accu>best_accu)
   {
   best_accu = accu;
   best_threshold = threshold_distance;
   }
	}
 result_file << "best_threshold=" << best_threshold << " ";
		result_file << "best_accu=" << best_accu << endl;
	result_file.close();

	//}

	return 0;

}
