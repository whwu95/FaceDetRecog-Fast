#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
//#include <opencv2/core/types.hpp>
#include <vector>


//#include <string>
//#include <cstring>
#include <cstdlib>
#include <cmath>
#include <io.h>
#include <atltime.h>

using namespace cv;
using namespace std;

void interpolateCubic(float x, float* coeffs)
{
	const float A = -0.5f;

	coeffs[0] = ((A*(x + 1) - 5 * A)*(x + 1) + 8 * A)*(x + 1) - 4 * A;
	coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
	coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
	coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

cv::Mat crop_face(cv::Mat& image, cv::Point2f key_pt[5], cv::Point2f base_pt[4])//crop_img中关键点crop_point
{
	//计算image中关键点key_point
	float crop_size[2] = { 112, 96 };

	//r= X\U 计算第一种变换，[u v] = [x y 1]*Tinv, [x y] = [u v 1]*T
	cv::Mat U = (cv::Mat_<float>(10, 1) << key_pt[0].x, key_pt[1].x,
		key_pt[2].x, key_pt[3].x, key_pt[4].x,
		key_pt[0].y, key_pt[1].y,
		key_pt[2].y, key_pt[3].y, key_pt[4].y);
	cv::Mat X = (cv::Mat_<float>(10, 4) << base_pt[0].x, base_pt[0].y, 1, 0,
		base_pt[1].x, base_pt[1].y, 1, 0,
		base_pt[2].x, base_pt[2].y, 1, 0,
		base_pt[3].x, base_pt[3].y, 1, 0,
		base_pt[4].x, base_pt[4].y, 1, 0,
		base_pt[0].y, -base_pt[0].x, 0, 1,
		base_pt[1].y, -base_pt[1].x, 0, 1,
		base_pt[2].y, -base_pt[2].x, 0, 1,
		base_pt[3].y, -base_pt[3].x, 0, 1,
		base_pt[4].y, -base_pt[4].x, 0, 1);
	cv::Mat r = X.inv(cv::DECOMP_SVD)*U;
	float sc = r.at<float>(0, 0);
	float ss = r.at<float>(0, 1);
	float tx = r.at<float>(0, 2);
	float ty = r.at<float>(0, 3);
	cv::Mat Tinv = (cv::Mat_<float>(3, 3) << sc, -ss, 0,
		ss, sc, 0,
		tx, ty, 1);
	cv::Mat T = Tinv.inv(cv::DECOMP_SVD);

	//rR = XR\U,计算第二种变换，[x y] = [u v 1]*T
	cv::Mat XR = (cv::Mat_<float>(8, 4) << -base_pt[0].x, base_pt[0].y, 1, 0,
		-base_pt[1].x, base_pt[1].y, 1, 0,
		-base_pt[2].x, base_pt[2].y, 1, 0,
		-base_pt[3].x, base_pt[3].y, 1, 0,
		-base_pt[4].x, base_pt[4].y, 1, 0,
		base_pt[0].y, base_pt[0].x, 0, 1,
		base_pt[1].y, base_pt[1].x, 0, 1,
		base_pt[2].y, base_pt[2].x, 0, 1,
		base_pt[3].y, base_pt[3].x, 0, 1,
		base_pt[4].y, base_pt[4].x, 0, 1);
	cv::Mat rR = XR.inv(cv::DECOMP_SVD)*U;
	float scR = rR.at<float>(0, 0);
	float ssR = rR.at<float>(0, 1);
	float txR = rR.at<float>(0, 2);
	float tyR = rR.at<float>(0, 3);
	cv::Mat TinvR = (cv::Mat_<float>(3, 3) << scR, -ssR, 0,
		ssR, scR, 0,
		txR, tyR, 1);
	cv::Mat TR = TinvR.inv(cv::DECOMP_SVD);
	cv::Mat TreflectY = (cv::Mat_<float>(3, 3) << -1, 0, 0,
		0, 1, 0,
		0, 0, 1);
	cv::Mat T2 = TR * TreflectY;
	cv::Mat Tinv2 = T2.inv();

	//使用[x y] = [ u v 1]*T(or T2)进行验证，选取最接近的一种变换
	cv::Mat uv = (cv::Mat_<float>(5, 3) << key_pt[0].x, key_pt[0].y, 1,
		key_pt[1].x, key_pt[1].y, 1,
		key_pt[2].x, key_pt[2].y, 1,
		key_pt[3].x, key_pt[3].y, 1,
		key_pt[4].x, key_pt[4].y, 1);
	cv::Mat xy1 = uv*T;
	cv::Mat xy2 = uv*T2;
	cv::Mat xy = (cv::Mat_<float>(5, 2) << base_pt[0].x, base_pt[0].y,
		base_pt[1].x, base_pt[1].y,
		base_pt[2].x, base_pt[2].y,
		base_pt[3].x, base_pt[3].y,
		base_pt[4].x, base_pt[4].y);
	float norm1 = norm(xy1(cv::Rect(0, 0, 2, 5)) - xy);
	float norm2 = norm(xy2(cv::Rect(0, 0, 2, 5)) - xy);
	cv::Mat trans;
	if (norm1 <= norm2)
		trans = Tinv(cv::Rect(0, 0, 2, 3));
	else
		trans = Tinv2(cv::Rect(0, 0, 2, 3));

	/*//验证[u v] = [x y 1]*Tinv
	cout << uv << endl;
	xy1 = (Mat_<float>(4, 3) << crop_point[0].x, crop_point[0].y,1,
	crop_point[1].x, crop_point[1].y,1,
	crop_point[2].x, crop_point[2].y,1,
	crop_point[3].x, crop_point[3].y,1);
	Mat uv1 = xy1*Tinv;
	cout << uv1 << endl;
	Mat uv2 = xy1*Tinv2;
	cout << uv2 << endl;*/

	cv::Mat crop_img = cv::Mat::zeros(crop_size[0], crop_size[1], CV_8UC3);
	sc = trans.at<float>(0, 0);
	ss = trans.at<float>(1, 0);
	tx = trans.at<float>(2, 0);
	ty = trans.at<float>(2, 1);
	for (int i = 0; i < crop_size[0]; i++)//y
		for (int j = 0; j < crop_size[1]; j++)//x
		{
		/*int x = (int)(sc*(j+1) + ss*(i+1) + tx)-1;
		int y = (int)(-ss*(j + 1) + sc*(i + 1) + ty)-1;*/
		float x = sc*j + ss*i + tx;//由目的坐标求得原来的坐标
		float y = -ss*j + sc*i + ty;
		int xx = int(x);
		int yy = int(y);//得到4*4个像素中左上角的坐标

		float coeffs_x[4], coeffs_y[4];
		interpolateCubic(x - xx, coeffs_x);
		interpolateCubic(y - yy, coeffs_y);

		xx = xx - 1;
		yy = yy - 1;//得到4*4个像素中左上角的坐标
		float b[3] = { 0, 0, 0 };
		for (int r = 0; r < 4; r++)//a与coeffs_y相乘，得到最终值b
		{
			float a[3] = { 0, 0, 0 };
			for (int c = 0; c < 4; c++)//依次取得每一行的四个像素点，与coeffs_x相乘，得到每一行的值a
			{
				uchar p[3] = { 127.5, 127.5, 127.5 };
				if ((xx + c)>-1 && (yy + r) > -1 && (xx + c) < image.cols && (yy + r) < image.rows)
				{
					p[0] = image.at<cv::Vec3b>((yy + r), (xx + c))[0];
					p[1] = image.at<cv::Vec3b>((yy + r), (xx + c))[1];
					p[2] = image.at<cv::Vec3b>((yy + r), (xx + c))[2];
				}
				a[0] += coeffs_x[c] * p[0];
				a[1] += coeffs_x[c] * p[1];
				a[2] += coeffs_x[c] * p[2];
			}
			b[0] += coeffs_y[r] * a[0];
			b[1] += coeffs_y[r] * a[1];
			b[2] += coeffs_y[r] * a[2];
		}
		crop_img.at<cv::Vec3b>(i, j)[0] = b[0];
		crop_img.at<cv::Vec3b>(i, j)[1] = b[1];
		crop_img.at<cv::Vec3b>(i, j)[2] = b[2];
		}
	imwrite("crop.jpg", crop_img);
	return crop_img;
}

int main()
{
	Point2f base_pt2[5];
	base_pt2[0] = cv::Point2f(30.2946, 51.6963);
	base_pt2[1] = cv::Point2f(65.5318, 51.5014);
	base_pt2[2] = cv::Point2f(48.0252, 71.7366);
	base_pt2[3] = cv::Point2f(33.5493, 92.3665);
	base_pt2[4] = cv::Point2f(62.7299, 92.2041);

	Point2f key_pt[5];
	key_pt[0] = cv::Point2f(162.5656, 160.4765);
	key_pt[1] = cv::Point2f(225.1543, 159.3038);
	key_pt[2] = cv::Point2f(201.0826, 196.9603);
	key_pt[3] = cv::Point2f(170.5590, 230.4894);
	key_pt[4] = cv::Point2f(225.0684, 227.9168);


	Mat img = imread("Aaliyah_04.jpg");
	cv::Mat crop_img = crop_face(img, key_pt, base_pt2);
	

	return 0;
}