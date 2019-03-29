#include "face_detect.h"
//#include "txt_io.h"

//#define my_max(a,b) (a)>(b)?(a):(b)
//#define my_min(a,b) (a)<(b)?(a):(b)

face_detector::face_detector()
{

}
face_detector::~face_detector()
{

}

void face_detector::system_init()
{
	//net init
	Caffe::set_mode(Caffe::CPU);
	string protofile;
	string modelfile;
	protofile = "../models/face_model1.prototxt";
	net1 = new Net<float>(protofile, TEST);
	modelfile = "../models/face_model1.caffemodel";
	net1->CopyTrainedLayersFrom(modelfile);

	protofile = "../models/face_model2.prototxt";
	net2 = new Net<float>(protofile, TEST);
	modelfile = "../models/face_model2.caffemodel";
	net2->CopyTrainedLayersFrom(modelfile);

	protofile = "../models/face_model3.prototxt";
	net3 = new Net<float>(protofile, TEST);
	modelfile = "../models/face_model3.caffemodel";
	net3->CopyTrainedLayersFrom(modelfile);
}
void face_detector::system_release()
{
	delete net1;
	delete net2;
	delete net3;

}
vector<CvRectScorePoint> face_detector::face_detect_main(cv::Mat& frame0)
{
	// detect set
	float l1 = 0.6;
	float l2 = 0.7;
	float l3 = 0.5;
	float n11 = 0.5;
	float n12 = 0.7;
	float n21 = 0.7;
	float n31 = 0.3;
	float n32 = 0.6;
	float factor = 0.709;
	float minface = 20;

	float m = 12.0f / minface;
	float minl = float(min(frame0.rows, frame0.cols) )* m;

	factor_count = 0;
	while (minl >= 12)
	{
		minl *= factor;
		factor_count++;
	}
	if (factor_count > 0)//防止循环中发生重新分配，可以先进行while，后对scales进行赋值scales.reserve()
	{
		float scale = m;
		scales = new float[factor_count];
		for (int i = 0; i < factor_count; i++)
		{
			scales[i] = scale;
			scale *= factor;
		}
	}

	cv::Mat frame1;
	cv::flip(frame0, frame1, 1);
	cv::Mat frame = frame1.t();//左右翻转+转置，还原matlab中数据类型
	cv::Mat frame_RGB;//opencv读入的图像是BGR存储顺序，matlab读入图像是RGB存储顺序
	cvtColor(frame, frame_RGB, CV_BGR2RGB);
	cv::Mat frame_f;//浮点型
	frame_RGB.convertTo(frame_f, CV_32FC3);

	//dl face detect
	vector<CvRectScoreReg> total_boxes_reg;
	vector<CvRectScore> total_boxes1;
	cv::Mat im_data1_pre = frame_f.clone();
	for (int j = 0; j < factor_count; j++)//将不同尺寸的frame送入net1
	{
		//准备输入
		float scale = scales[j];
		int hs = frame.rows * scale;//float强制转int，直接舍去小数点后数值
		int ws = frame.cols * scale;

		cv::Mat im_data1 = cv::Mat::zeros(cv::Size(ws, hs), CV_32FC3);
		//resize(frame_f, im_data1, Size(ws, hs));
		cv::resize(im_data1_pre, im_data1, cv::Size(ws, hs));
		im_data1_pre.release();
		im_data1_pre = im_data1.clone();

		//准备把数据存入
		Blob<float>* input_layer = net1->input_blobs()[0];
		input_layer->Reshape(1, 3, hs, ws);
		net1->Reshape();
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		vector<cv::Mat> in_data1;
		for (int i = 0; i < 3; ++i) {
			cv::Mat channel(hs, ws, CV_32FC1, input_data);
			in_data1.push_back(channel);
			input_data += width * height;
		}
		cv::split(im_data1, in_data1);
		for (int dim = 0; dim < 3; dim++)
		{
			in_data1[dim] = (in_data1[dim] - 127.5f) / 128.0f;
		}
			
		//push image into caffe blobs
		input_layer = net1->input_blobs()[0];
		input_data = input_layer->mutable_cpu_data();
		CHECK(reinterpret_cast<float*>(in_data1.at(0).data)
			== net1->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";

		net1->ForwardPrefilled();

		Blob<float>* output1_blobs_1 = net1->output_blobs()[0];//h*w*c,c=4对应位移
		Blob<float>* output1_blobs_2 = net1->output_blobs()[1];//h*w*c,c=2对应得分，c：2是人脸的的人

		vector<CvRectScoreReg> boundingbox1;

		float stride = 2.0f;
		float cellsize = 12.0f;

		int out2_H = output1_blobs_2->height();
		int out2_W = output1_blobs_2->width();
		for (int x = 0;x < out2_H; x++)
		{
			for (int y = 0; y < out2_W; y++)//此处y对应matlab中y-1，x对应matlab中x-1
			{
				float score = 0;
				score = output1_blobs_2->data_at(0, 1, x, y);
				if (score > l1)//认定为人脸
				{
					CvRectScoreReg tmp;
					tmp.x1 = (stride*y + 1) / scale - 1;
					tmp.y1 = (stride*x + 1) / scale - 1;
					tmp.x2 = (stride*y + cellsize) / scale - 1;
					tmp.y2 = (stride*x + cellsize) / scale - 1;
					tmp.score = score;
					tmp.reg_x1 = output1_blobs_1->data_at(0, 1, x, y);
					tmp.reg_y1 = output1_blobs_1->data_at(0, 0, x, y);
					tmp.reg_x2 = output1_blobs_1->data_at(0, 3, x, y);
					tmp.reg_y2 = output1_blobs_1->data_at(0, 2, x, y);
					boundingbox1.push_back(tmp);//人脸regs 位移
				}
			}
		}//end 由net1 的out 生成boundingbox
		//非极大值抑制
		CvRectScoreReg * boxes = new CvRectScoreReg[boundingbox1.size()];
		for (int i = 0; i < boundingbox1.size(); i++)
			boxes[i] = boundingbox1[i];
		int pick = NonMaximalSuppression(boxes, boundingbox1.size(), n11);
		for (int i = 0; i < pick; i++)
			total_boxes_reg.push_back(boxes[i]);
		delete[]boxes;
	}
	delete[]scales;

	int numbox = total_boxes_reg.size();
	if (numbox != 0)
	{
		//非极大值抑制
		CvRectScoreReg * boxes = new CvRectScoreReg[numbox];
		for (int i = 0; i < numbox; i++)
			boxes[i] = total_boxes_reg[i];//类型转换
		int pick = NonMaximalSuppression(boxes, numbox, n12);
		for (int i = 0; i < pick; i++)
		{
			CvRectScoreReg tmp = boxes[i];//由rect和reg得到最后的total_boxes1位置
			int w2 = tmp.x2 - tmp.x1;
			int h2 = tmp.y2 - tmp.y1;
			float x1 = float(tmp.x1) + tmp.reg_x1*float(w2);
			float y1 = float(tmp.y1) + tmp.reg_y1*float(h2);
			float x2 = float(tmp.x2) + tmp.reg_x2*float(w2);
			float y2 = float(tmp.y2) + tmp.reg_y2*float(h2);
			float maxline = max(x2-x1, y2-y1);//长方形boxes变正方形boxes
			x1 = x1 + (x2 - x1)*0.5 - maxline*0.5;
			y1 = y1 + (y2 - y1)*0.5 - maxline*0.5;
			x2 = x1 + maxline;
			y2 = y1 + maxline;

			CvRectScore tmp_total_box;
			tmp_total_box.x1 = (int)x1;
			tmp_total_box.y1 = (int)y1;
			tmp_total_box.x2 = (int)x2;
			tmp_total_box.y2 = (int)y2;
			tmp_total_box.score = tmp.score;
	
			total_boxes1.push_back(tmp_total_box);
		}
		delete[]boxes;
	}

	//开始net2
	vector<cv::Mat> im_data2;
	for (int n = 0; n < total_boxes1.size(); n++)
	{
		//超出边缘部分，添0处理
		CvRectScore o_box = total_boxes1[n];
		cv::Mat d_img = cv::Mat::zeros(o_box.y2 - o_box.y1 + 1, o_box.x2 - o_box.x1+1, CV_32FC3);

		for (int i = 0; i < d_img.rows; i++)//行
		{
			int y = o_box.y1 + i;
			for (int j = 0; j < d_img.cols; j++)
			{
				int x = o_box.x1 + j;
				if ((x >= 0) && (x < frame.cols) && (y >= 0) && (y < frame.rows))
				{
					d_img.at<cv::Vec3f>(i, j)[0] = frame_f.at<cv::Vec3f>(y, x)[0];
					d_img.at<cv::Vec3f>(i, j)[1] = frame_f.at<cv::Vec3f>(y, x)[1];
					d_img.at<cv::Vec3f>(i, j)[2] = frame_f.at<cv::Vec3f>(y, x)[2];
				}
			}
		}
		cv::Mat im_data = cv::Mat::zeros(cv::Size(24, 24), CV_32FC3);
		cv::resize(d_img, im_data, cv::Size(24, 24));
		im_data2.push_back(im_data);

	}
	//送入网络
	vector<CvRectScore> total_boxes2;
	vector<CvRectScoreReg> boundingbox2;
	for (int n = 0; n < im_data2.size(); n++)
	{
		Blob<float>* input_layer = net2->input_blobs()[0];
		input_layer->Reshape(1, 3, 24, 24);
		net2->Reshape();
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		vector<cv::Mat> in_data2;
		for (int i = 0; i < 3; ++i) {
			cv::Mat channel(24, 24, CV_32FC1, input_data);
			in_data2.push_back(channel);
			input_data += width * height;
		}
		cv::split(im_data2[n], in_data2);
		for (int dim = 0; dim < 3; dim++)
			in_data2[dim] = (in_data2[dim] - 127.5f) / 128.0f;
		CHECK(reinterpret_cast<float*>(in_data2.at(0).data)
			== net2->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";

		net2->ForwardPrefilled();

		//net2输出
		Blob<float>* output2_blobs_1 = net2->output_blobs()[0];//nchw-1*2*1*1
		Blob<float>* output2_blobs_2 = net2->output_blobs()[1];//nchw-1*2*1*1
		//处理net2的输出
		float score = output2_blobs_2->data_at(0, 1, 0, 0);
		if (score > l2)//认定为人脸
		{
			CvRectScoreReg tmp;
			tmp.x1 = total_boxes1[n].x1;
			tmp.y1 = total_boxes1[n].y1;
			tmp.x2 = total_boxes1[n].x2;
			tmp.y2 = total_boxes1[n].y2;
			tmp.score = score;
		
			tmp.reg_x1 = output2_blobs_1->data_at(0, 1, 0, 0);
			tmp.reg_y1 = output2_blobs_1->data_at(0, 0, 0, 0);
			tmp.reg_x2 = output2_blobs_1->data_at(0, 3, 0, 0);
			tmp.reg_y2 = output2_blobs_1->data_at(0, 2, 0, 0);
			boundingbox2.push_back(tmp);//人脸regs 位移
		}
	}

	numbox = boundingbox2.size();
	if (numbox != 0)
	{
		//非极大值抑制
		CvRectScoreReg * boxes = new CvRectScoreReg[numbox];
		for (int i = 0; i < numbox; i++)
			boxes[i] = boundingbox2[i];//类型转换
		int pick = NonMaximalSuppression(boxes, numbox, n21);
		for (int i = 0; i < pick; i++)
		{
			CvRectScoreReg tmp = boxes[i];//由rect和reg得到最后的total_boxes1位置
			int w = tmp.x2 - tmp.x1+1;
			int h = tmp.y2 - tmp.y1+1;
			float x1 = float(tmp.x1) + tmp.reg_x1*float(w);
			float y1 = float(tmp.y1) + tmp.reg_y1*float(h);
			float x2 = float(tmp.x2) + tmp.reg_x2*float(w);
			float y2 = float(tmp.y2) + tmp.reg_y2*float(h);
			float maxline = max(x2 - x1, y2 - y1);//长方形boxes变正方形boxes
			x1 = x1 + (x2 - x1)*0.5 - maxline*0.5;
			y1 = y1 + (y2 - y1)*0.5 - maxline*0.5;
			x2 = x1 + maxline;
			y2 = y1 + maxline;

			CvRectScore tmp_total_box;
			tmp_total_box.x1 = (int)x1;
			tmp_total_box.y1 = (int)y1;
			tmp_total_box.x2 = (int)x2;
			tmp_total_box.y2 = (int)y2;
			tmp_total_box.score = tmp.score;
			total_boxes2.push_back(tmp_total_box);
		}
		delete[]boxes;
	}
	
	//开始net3
	vector<cv::Mat> im_data3;
	for (int n = 0; n < total_boxes2.size(); n++)
	{
		//超出边缘部分，添0处理
		CvRectScore o_box = total_boxes2[n];
		cv::Mat d_img = cv::Mat::zeros(o_box.y2 - o_box.y1 + 1, o_box.x2 - o_box.x1 + 1, CV_32FC3);

		for (int i = 0; i < d_img.rows; i++)//行
		{
			int y = o_box.y1 + i;
			for (int j = 0; j < d_img.cols; j++)
			{
				int x = o_box.x1 + j;
				if ((x >= 0) && (x < frame.cols) && (y >= 0) && (y < frame.rows))
				{
					d_img.at<cv::Vec3f>(i, j)[0] = frame_f.at<cv::Vec3f>(y, x)[0];
					d_img.at<cv::Vec3f>(i, j)[1] = frame_f.at<cv::Vec3f>(y, x)[1];
					d_img.at<cv::Vec3f>(i, j)[2] = frame_f.at<cv::Vec3f>(y, x)[2];
				}
			}
		}
		cv::Mat im_data = cv::Mat::zeros(cv::Size(48, 48), CV_32FC3);
		cv::resize(d_img, im_data, cv::Size(48, 48));
		im_data3.push_back(im_data);
	}
	//送入网络
	vector<CvRectScorePoint> total_boxes3;
	vector<CvRectScoreRegPoint> boundingbox3;
	for (int n = 0; n < im_data3.size(); n++)
	{
		Blob<float>* input_layer = net3->input_blobs()[0];
		input_layer->Reshape(1, 3, 48, 48);
		net3->Reshape();
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		vector<cv::Mat> in_data3;
		for (int i = 0; i < 3; ++i) {
			cv::Mat channel(48, 48, CV_32FC1, input_data);
			in_data3.push_back(channel);
			input_data += width * height;
		}
		cv::split(im_data3[n], in_data3);
		for (int dim = 0; dim < 3; dim++)
			in_data3[dim] = (in_data3[dim] - 127.5f) / 128.0f;
		CHECK(reinterpret_cast<float*>(in_data3.at(0).data)
			== net3->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";

		net3->ForwardPrefilled();
		//net3输出
		Blob<float>* output3_blobs_1 = net3->output_blobs()[0];//nchw-1*4*1*1
		Blob<float>* output3_blobs_2 = net3->output_blobs()[1]; //nchw - 1 * 10 * 1 * 1
		Blob<float>* output3_blobs_3 = net3->output_blobs()[2];//nchw-1*2*1*1

		float score = output3_blobs_3->data_at(0, 1, 0, 0);
		if (score >= l3)//认定为人脸
		{
			CvRectScoreRegPoint tmp;
			tmp.x1 = total_boxes2[n].x1;
			tmp.y1 = total_boxes2[n].y1;
			tmp.x2 = total_boxes2[n].x2;
			tmp.y2 = total_boxes2[n].y2;
			tmp.score = score;

			tmp.reg_x1 = output3_blobs_1->data_at(0, 1, 0, 0);
			tmp.reg_y1 = output3_blobs_1->data_at(0, 0, 0, 0);
			tmp.reg_x2 = output3_blobs_1->data_at(0, 3, 0, 0);
			tmp.reg_y2 = output3_blobs_1->data_at(0, 2, 0, 0);

			cv::Point2f point[5];
			for (int j = 0; j < 5; j++)
			{
				point[j].y = output3_blobs_2->data_at(0, j, 0, 0);
				point[j].x = output3_blobs_2->data_at(0, j + 5, 0, 0);
			}
			for (int j = 0; j < 5; j++)
			{
				tmp.point[j].x = tmp.x1 + point[j].x*(tmp.x2 - tmp.x1 + 1);
				tmp.point[j].y = tmp.y1 + point[j].y*(tmp.y2 - tmp.y1 + 1);
			}

			boundingbox3.push_back(tmp);//人脸regs 位移
		}
	}//end 由net3 的out 生成total_boxes3
	//对total_boxes3进行非最大值抑制
	numbox = boundingbox3.size();
	if (numbox != 0)
	{
		//非极大值抑制
		CvRectScoreRegPoint * boxes = new CvRectScoreRegPoint[numbox];
		for (int i = 0; i < numbox; i++)
			boxes[i] = boundingbox3[i];//类型转换
		int pick = NonMaximalSuppression(boxes, numbox, n31);
		//非极大值抑制2
		CvRectScoreRegPoint * boxes2 = new CvRectScoreRegPoint[pick];
		for (int i = 0; i < pick; i++)
			boxes2[i] = boxes[i];//类型转换
		delete[]boxes;
		int pick2 = NonMaximalSuppression2(boxes2, pick, n32);
		for (int i = 0; i < pick2; i++)
		{
			CvRectScoreRegPoint tmp = boxes2[i];//由rect和reg得到最后的total_boxes1位置
			int w = tmp.x2 - tmp.x1 + 1;
			int h = tmp.y2 - tmp.y1 + 1;
			float x1 = float(tmp.x1) + tmp.reg_x1*float(w);
			float y1 = float(tmp.y1) + tmp.reg_y1*float(h);
			float x2 = float(tmp.x2) + tmp.reg_x2*float(w);
			float y2 = float(tmp.y2) + tmp.reg_y2*float(h);

			CvRectScorePoint tmp_total_box;
			tmp_total_box.x1 = x1;
			tmp_total_box.y1 = y1;
			tmp_total_box.x2 = x2;
			tmp_total_box.y2 = y2;
			tmp_total_box.score = tmp.score;
			for (int j = 0; j < 5; j++)
			{
				tmp_total_box.point[j].x = tmp.point[j].x;
				tmp_total_box.point[j].y = tmp.point[j].y;
			}
			total_boxes3.push_back(tmp_total_box);
		}
		delete[]boxes2;
	}

	vector<CvRectScorePoint> result;
	for (int i = 0; i < total_boxes3.size(); i++)//输出坐标进行转置和翻转
	{
		CvRectScorePoint tmp;
		tmp.x1 = frame0.cols - total_boxes3[i].y2;
		tmp.y1 = total_boxes3[i].x1;
		tmp.x2 = frame0.cols - total_boxes3[i].y1;
		tmp.y2 = total_boxes3[i].x2;
		tmp.score = total_boxes3[i].score;

		tmp.point[0].x = frame0.cols - total_boxes3[i].point[1].y;
		tmp.point[0].y = total_boxes3[i].point[1].x;

		tmp.point[1].x = frame0.cols - total_boxes3[i].point[0].y;
		tmp.point[1].y = total_boxes3[i].point[0].x;

		tmp.point[2].x = frame0.cols - total_boxes3[i].point[2].y;
		tmp.point[2].y = total_boxes3[i].point[2].x;

		tmp.point[3].x = frame0.cols - total_boxes3[i].point[4].y;
		tmp.point[3].y = total_boxes3[i].point[4].x;

		tmp.point[4].x = frame0.cols - total_boxes3[i].point[3].y;
		tmp.point[4].y = total_boxes3[i].point[3].x;

		result.push_back(tmp);
	}

	return result;
}

int face_detector::NonMaximalSuppression(CvRectScoreReg* boxes_orig, int num, float overlap) {

	CvRectScoreReg *boxes = boxes_orig;
	int numNMS = 0;
	int numGood = num;

	if (overlap >= 0) {
		while ((numGood - numNMS)>0)  {
			int best = -1;
			float bestS = -10000000;
			for (int i = numNMS; i < numGood; i++) {
				if (boxes[i].score > bestS) {
					bestS = boxes[i].score;
					best = i;
				}
			}

			CvRectScoreReg b = boxes[best];
			CvRectScoreReg tmp = boxes[numNMS];
			boxes[numNMS] = boxes[best];
			boxes[best] = tmp;
			numNMS++;

			int A1 = (b.x2 - b.x1+1)*(b.y2 - b.y1 +1);
			int A2, inter;
			float overlay_area;
			int numPick = 0;
			int x1, x2, y1, y2;
			for (int i = numNMS; i < numGood; i++) {
				x1 = max(b.x1, boxes[i].x1);
				y1 = max(b.y1, boxes[i].y1);
				x2 = min(b.x2, boxes[i].x2);
				y2 = min(b.y2, boxes[i].y2);
				A2 = (boxes[i].x2 - boxes[i].x1 + 1)*(boxes[i].y2 - boxes[i].y1+1);
				int width = max(0, (x2 - x1+1));
				int height = max(0, (y2 - y1+1));
				inter = width*height;
				overlay_area = float(inter) / (float)(A1 + A2 - inter);
				if (overlay_area <= overlap) {
					boxes[numNMS + numPick] = boxes[i];
					numPick++;
				}
			}
			numGood = numNMS + numPick;
		}
	}
	return numNMS;
}
int face_detector::NonMaximalSuppression(CvRectScoreRegPoint* boxes_orig, int num, float overlap) {

	CvRectScoreRegPoint *boxes = boxes_orig;
	int numNMS = 0;
	int numGood = num;

	if (overlap >= 0) {
		while ((numGood - numNMS)>0)  {
			int best = -1;
			float bestS = -10000000;
			for (int i = numNMS; i < numGood; i++) {
				if (boxes[i].score > bestS) {
					bestS = boxes[i].score;
					best = i;
				}
			}
			
			CvRectScoreRegPoint b = boxes[best];
			CvRectScoreRegPoint tmp = boxes[numNMS];
			boxes[numNMS] = boxes[best];
			boxes[best] = tmp;
			numNMS++;

			int A1 = (b.x2 - b.x1 + 1)*(b.y2 - b.y1 + 1);
			int A2, inter;
			float overlay_area;
			int numPick = 0;
			int x1, x2, y1, y2;
			for (int i = numNMS; i < numGood; i++) {
				x1 = max(b.x1, boxes[i].x1);
				y1 = max(b.y1, boxes[i].y1);
				x2 = min(b.x2, boxes[i].x2);
				y2 = min(b.y2, boxes[i].y2);
				A2 = (boxes[i].x2 - boxes[i].x1 + 1)*(boxes[i].y2 - boxes[i].y1 + 1);
				int width = max(0, (x2 - x1 + 1));
				int height = max(0, (y2 - y1 + 1));
				inter = width*height;
				overlay_area = float(inter) / (float)(A1 + A2 - inter);
				if (overlay_area <= overlap) {
					boxes[numNMS + numPick] = boxes[i];
					numPick++;
				}
			}
			numGood = numNMS + numPick;
		}
	}
	return numNMS;
}
int face_detector::NonMaximalSuppression2(CvRectScoreRegPoint* boxes_orig, int num, float overlap) {

	CvRectScoreRegPoint *boxes = boxes_orig;
	int numNMS = 0;
	int numGood = num;

	if (overlap >= 0) {
		while ((numGood - numNMS)>0)  {
			int best = -1;
			float bestS = -10000000;
			for (int i = numNMS; i < numGood; i++) {
				if (boxes[i].score > bestS) {
					bestS = boxes[i].score;
					best = i;
				}
			}
			
			CvRectScoreRegPoint b = boxes[best];
			CvRectScoreRegPoint tmp = boxes[numNMS];
			boxes[numNMS] = boxes[best];
			boxes[best] = tmp;
			numNMS++;

			int A1 = (b.x2 - b.x1 + 1)*(b.y2 - b.y1 + 1);
			int A2, inter;
			float overlay_area;
			int numPick = 0;
			int x1, x2, y1, y2;
			for (int i = numNMS; i < numGood; i++) {
				x1 = max(b.x1, boxes[i].x1);
				y1 = max(b.y1, boxes[i].y1);
				x2 = min(b.x2, boxes[i].x2);
				y2 = min(b.y2, boxes[i].y2);
				A2 = (boxes[i].x2 - boxes[i].x1 + 1)*(boxes[i].y2 - boxes[i].y1 + 1);
				int width = max(0, (x2 - x1 + 1));
				int height = max(0, (y2 - y1 + 1));
				inter = width*height;
				overlay_area = float(inter) / float(min(A1, A2));
				if (overlay_area <= overlap) {
					boxes[numNMS + numPick] = boxes[i];
					numPick++;
				}
			}
			numGood = numNMS + numPick;
		}
	}
	return numNMS;
}