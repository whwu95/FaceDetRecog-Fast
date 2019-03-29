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

	vector<vector<float> > all_db_feature;
	vector<float > db_feat;
	std::ifstream db_feature_file;

	db_feature_file.open(db_feature_dir.c_str());
	string db_feature_str;
	int detect_error_num = 0;
	while (getline(db_feature_file, db_feature_str))
	{
		db_feat = vector_read_feature(db_feature_str);
		all_db_feature.push_back(db_feat);
		if (db_feat.size() != 256)
			detect_error_num++;
	}
	db_feature_file.close();


	string db_name_dir = argv[2];

	vector<string> all_db_name;
	std::ifstream db_name_file;

	db_name_file.open(db_name_dir.c_str());
	string db_name_str;
	while (getline(db_name_file, db_name_str))
		all_db_name.push_back(db_name_str);
	db_name_file.close();

	
	int db_img_num = all_db_feature.size();
	float distance;
	//string result_dir = "recog_lfw_result.txt";
	string result_dir = argv[3];


	//for (int k = 0; k <= 100; k++)
	//{
	//threshold_distance += k*0.005;
	int pass_num[101] = { 0 };
	int cmp_num = 0;
	
	for (int i = 0; i < db_img_num; i++)
	{
		cout << "i= " << i <<  endl;
		if (all_db_feature.at(i).size() != 256)
			continue;

		float min_distance = 1000;
		int min_idx = -1;
		for (int j =0; j < db_img_num; j++)
		{
			if (i==j || all_db_feature.at(j).size() != 256)
				continue;
			face.calculateDistance(all_db_feature.at(i), all_db_feature.at(j), distance);

			if (distance < min_distance)
			{
				min_distance = distance;
				min_idx = j;
			}
		}
   
   
   
		if (0 == strcmp(all_db_name.at(i).c_str(), all_db_name.at(min_idx).c_str()))//same person
		{
			float threshold_distance = 0.7;
			for (int k = 0; k <= 100; k++){
				threshold_distance += 0.005;
				if (min_distance < threshold_distance)
					pass_num[k]++;
			}
		}
		else//diff
		{
			float threshold_distance = 0.7;
			for (int k = 0; k <= 100; k++){
				threshold_distance += 0.005;
				if (min_distance >= threshold_distance)
					pass_num[k]++;
			}
		}
		
	}

  float best_threshold = 0.7,best_rate = -1;
	std::ofstream result_file(result_dir.c_str(), ios::in | ios::app);
	result_file << "db_img_num=" << db_img_num << " ";
	result_file << "detect_error_num=" << detect_error_num << " ";
	result_file << "cmp_num=" << (db_img_num - detect_error_num )<< endl;
	float threshold_distance = 0.7;
	for (int k = 0; k <= 100; k++){
		threshold_distance += 0.005;
		float rate = (float)pass_num[k] / (db_img_num - detect_error_num);
		result_file << k << " threshold_distance=" << threshold_distance << " ";
		result_file << "pass_num=" << pass_num[k] << " ";
		result_file << "rate=" << rate << endl;
   if(rate>best_rate)
   {
   best_rate = rate;
   best_threshold = threshold_distance;
   }
	}
 result_file << "best_threshold=" << best_threshold << " ";
		result_file << "best_rate=" << best_rate << endl;
	result_file.close();

	//}

	return 0;

}
