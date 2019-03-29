#include "global.h"
#include "FACE.h"
#include "havon_ffd.h"
#include <time.h>

FACE::FACE(){}

FACE::FACE(const std::string &dir) {}

FACE::FACE(const std::vector<std::string> model_file, const std::vector<std::string> trained_file)
{
    #ifdef CPU_ONLY
        Caffe::set_mode(Caffe::CPU);
    #else
        Caffe::set_mode(Caffe::GPU);
    #endif

    for(int i = 0; i < model_file.size(); i++)
    {
        boost::shared_ptr<Net<float> > net;

        cv::Size input_geometry;
        int num_channel;

        net.reset(new Net<float>(model_file[i], TEST));
        net->CopyTrainedLayersFrom(trained_file[i]);

        Blob<float>* input_layer = net->input_blobs()[0];
        num_channel = input_layer->channels();
        input_geometry = cv::Size(input_layer->width(), input_layer->height());

        nets_.push_back(net);
        input_geometry_.push_back(input_geometry);
        if(i == 0)
            num_channels_ = num_channel;
        else if(num_channels_ != num_channel)
            std::cout << "Error: The number channels of the nets are different!" << std::endl;
    }
}


FACE::~FACE(){}

void FACE::detection_TEST(const cv::Mat& img, std::vector<cv::Rect>& rectangles)
{
    Preprocess(img);
    STEP1_Net();
    img_show_T(img, "STEP1-Net");
    NMS_Union(bounding_box_,threshold_NMS_[1]);
    img_show_T(img, "STEP1-Net_nms");
    STEP2_Net();
    img_show_T(img, "STEP2-Net");
    NMS_Union(bounding_box_,threshold_NMS_[1]);
    img_show_T(img, "STEP2-Net_nms");
    STEP3_Net();
    img_show_T(img, "STEP3-Net");
    NMS_Min(bounding_box_);
    img_show_T(img, "STEP3-Net_nms");
   
}


void FACE::extractFeature(const cv::Mat& cropFace, std::vector<float>& feature)
{ 
     timeval starti, endi; 
     unsigned  long t; 
     gettimeofday(&starti, NULL); 

   for(int i = 0; i < bounding_box_.size(); i++){
    	 if (bounding_box_[i].height * bounding_box_[i].width < bounding_box_[i+1].height * bounding_box_[i+1].width){
    	 	bounding_box_[i] = bounding_box_[i+1];
    	 }

    }
   
   Preprocessing(cropFace);
   cv::Mat image=img_;

   image.convertTo(image, CV_32FC3, 0.0078125,-127.5*0.0078125);

   boost::shared_ptr<Net<float> > net = nets_[3];
   Blob<float>* input_layer = net->input_blobs()[0];
   input_layer->Reshape(1, num_channels_,
                         image.rows, image.cols);
	  	    
   net->Reshape();
   std::vector<cv::Mat> input_channels;
   WrapInputLayer(image, &input_channels, 3);

   net->Forward();
   Blob<float>* output_layer = net->output_blobs()[0];
   const float* begin = output_layer->cpu_data();
   const float* end = begin + output_layer->channels();
   feature = std::vector<float>(begin, end);

    gettimeofday(&endi, NULL); 
    t=1000000*(endi.tv_sec - starti.tv_sec) + (endi.tv_usec - starti.tv_usec);
    cout<<"time is "<<t<<endl;
}

void FACE::Preprocess(const cv::Mat &img)
{
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else       
        sample = img;
    
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);

     
    cv::cvtColor(sample_float,sample_float,cv::COLOR_BGR2RGB);
    
    sample_float = sample_float.t();
    img_ = sample_float;
}

void FACE::Preprocessing(const cv::Mat &img)
{
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else       
        sample = img;
    
    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);

    img_ = sample_float;
}


void FACE::STEP1_Net()
{
    resize_img();

    for(int j = 0; j < img_resized_.size(); j++)
    {
        cv::Mat img = img_resized_[j];
        Predict(img, 0);
        GenerateBoxs(img);
        NMS_Union(bounding_box_,threshold_NMS_[0]);
    }
}

void FACE::STEP2_Net()
{
    detect_net(1);
}

void FACE::STEP3_Net()
{
    detect_net(2);
}

void FACE::detect_net(int i)
{
    float thresh = threshold_[i];
    std::vector<cv::Rect> bounding_box;
    std::vector<float> confidence;
    std::vector<cv::Mat> cur_imgs;
    std::vector<std::vector<cv::Point> > alignment;

    if(bounding_box_.size() == 0)
        return;

    for (int j = 0; j < bounding_box_.size(); j++) {
        cv::Mat img = crop(img_, bounding_box_[j]);
        if (img.size() == cv::Size(0,0))
            continue;
        if (img.rows == 0 || img.cols == 0)
            continue;
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i], 0 , 0 ,INTER_AREA  );
        img.convertTo(img, CV_32FC3, 0.0078125,-127.5*0.0078125);
        cur_imgs.push_back(img);
    }


    Predict(cur_imgs, i);

    for(int j = 0; j < confidence_temp_.size()/2; j++)
    {
        float conf = confidence_temp_[2*j+1];
        if (conf > thresh) {

            if(conf>1)
                int a = 0;

            //bounding box
            cv::Rect bbox;

            //regression box : y x height width
            bbox.y = bounding_box_[j].y + regression_box_temp_[4*j] * bounding_box_[j].height;
            bbox.x = bounding_box_[j].x + regression_box_temp_[4*j+1] * bounding_box_[j].width ;
            bbox.height = bounding_box_[j].height + regression_box_temp_[4*j+2] * bounding_box_[j].height;
            bbox.width = bounding_box_[j].width + regression_box_temp_[4*j+3] * bounding_box_[j].width;


            if(bbox.x < -1000 || bbox.x > 1000)
                int a = 0;

            if(i == 2)
            {
                //face alignment
                std::vector<cv::Point> align(5);
                for(int k = 0; k < 5; k++)
                {

                    align[k].x = bounding_box_[j].x + bounding_box_[j].width * alignment_temp_[10*j+5+k] - 1;
                    align[k].y = bounding_box_[j].y + bounding_box_[j].height * alignment_temp_[10*j+k] - 1;
                }
                alignment.push_back(align);
            }

            confidence.push_back(conf);
            bounding_box.push_back(bbox);

        }
    }

    cur_imgs.clear();

    bounding_box_ = bounding_box;
    confidence_ = confidence;
    alignment_ = alignment;
    
}






void FACE::Predict(const cv::Mat& img, int i)
{
    boost::shared_ptr<Net<float> > net = nets_[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         img.rows, img.cols);

    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(img, &input_channels, i);
    net->Forward();

    Blob<float>* rect = net->output_blobs()[0];
    Blob<float>* confidence = net->output_blobs()[1];
    int count = confidence->count() / 2;

    const float* rect_begin = rect->cpu_data();
    const float* rect_end = rect_begin + rect->channels() * count;
    regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

    const float* confidence_begin = confidence->cpu_data() + count;
    const float* confidence_end = confidence_begin + count;

    confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
}


void FACE::Predict(const std::vector<cv::Mat> imgs, int i)
{
    boost::shared_ptr<Net<float> > net = nets_[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(imgs.size(), num_channels_,
                         input_geometry_[i].height, input_geometry_[i].width);
    int num = input_layer->num();

    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(imgs, &input_channels, i);

    net->Forward();
    
    

    if( i == 1)
    {
        Blob<float>* rect = net->output_blobs()[0];
        Blob<float>* confidence = net->output_blobs()[1];

        int count = confidence->count() / 2;
        const float* rect_begin = rect->cpu_data();
        const float* rect_end = rect_begin + rect->channels() * count;
        regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

        const float* confidence_begin = confidence->cpu_data();
        const float* confidence_end = confidence_begin + count * 2;

        confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
    }
    
    if(i == 2)
    {
        Blob<float>* rect = net->output_blobs()[0];
        Blob<float>* points = net->output_blobs()[1];
        Blob<float>* confidence = net->output_blobs()[2];

        int count = confidence->count() / 2;

        const float* rect_begin = rect->cpu_data();
        const float* rect_end = rect_begin + rect->channels() * count;
        regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

        const float* points_begin = points->cpu_data();
        const float* points_end = points_begin + points->channels() * count;
        alignment_temp_ = std::vector<float>(points_begin, points_end);

        const float* confidence_begin = confidence->cpu_data();
        const float* confidence_end = confidence_begin + count * 2;
        confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
    }
    
}

void FACE::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
{
    Blob<float>* input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    cv::split(img, *input_channels);

}

void FACE::WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i)
{
    Blob<float> *input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float *input_data = input_layer->mutable_cpu_data();

    for (int j = 0; j < num; j++) {
        
        for (int k = 0; k < input_layer->channels(); ++k) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
        cv::Mat img = imgs[j];
        cv::split(img, *input_channels);
        input_channels->clear();
    }
}

float FACE::IoU(cv::Rect rect1, cv::Rect rect2)
{
    int x_overlap, y_overlap, intersection, unions;
    float area1,area2;
    area1 = (rect1.width + 1) * (rect1.height +1);
    area2 = (rect2.width + 1) * (rect2.height +1);
    x_overlap = std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) - std::max(rect1.x, rect2.x));
    y_overlap = std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) - std::max(rect1.y, rect2.y));
    intersection = x_overlap * y_overlap;
    unions = area1 + area2 - intersection;
    return float(intersection)/unions;
}

float FACE::IoM(cv::Rect rect1, cv::Rect rect2)
{
    int x_overlap, y_overlap, intersection, min_area;
    float area1,area2;
    area1 = (rect1.width + 1) * (rect1.height +1);
    area2 = (rect2.width + 1) * (rect2.height +1);
    x_overlap = std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) - std::max(rect1.x, rect2.x));
    y_overlap = std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) - std::max(rect1.y, rect2.y));
    intersection = x_overlap * y_overlap;
    min_area = std::min(area1, area2);
    return float(intersection)/min_area;
}

void FACE::resize_img()
{
    cv::Mat img = img_;
    int height = img.rows;
    int width = img.cols;

    int minSize = minSize_;
    float factor = factor_;
    double scale = 12./minSize;
    int minWH = std::min(height, width) * scale;

    std::vector<cv::Mat> img_resized;

    while(minWH >= 12)
    {
        int resized_h = std::ceil(height*scale);
        int resized_w = std::ceil(width*scale);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA );
        resized.convertTo(resized, CV_32FC3, 0.0078125,-127.5*0.0078125);
        img_resized.push_back(resized);


        minWH *= factor;
        scale *= factor;
    }

    img_resized_ = img_resized;
}

void FACE::GenerateBoxs(cv::Mat img)
{
    int stride = 2;
    int cellSize = input_geometry_[0].width;
    int image_h = img.rows;
    int image_w = img.cols;
    double scale = double(image_w) / img_.cols ;
    int feature_map_h = std::ceil((image_h - cellSize)*1.0/stride)+1;
    int feature_map_w = std::ceil((image_w - cellSize)*1.0/stride)+1;
    int width = (cellSize) / scale;
    int count = confidence_temp_.size();
    float thresh = threshold_[0];

    std::vector<cv::Rect> bounding_box;
    std::vector<cv::Rect> regression_box;

    std::vector<float> confidence;

    for(int i = 0; i < count; i++)
    {
        if(confidence_temp_[i] < thresh)
            continue;

        confidence.push_back(confidence_temp_[i]);

        int y = i / feature_map_w;
        int x = i - feature_map_w * y;


        regression_box.push_back(cv::Rect(regression_box_temp_[i + count*1],regression_box_temp_[i + count*0],
                                regression_box_temp_[i + count*3],regression_box_temp_[i + count*2]));


         bounding_box.push_back(cv::Rect( ((x-0)*stride+1)/scale-1, ((y-0)*stride+1)/scale-1, 
                                          width, width ) );

        if((x*stride+1)/scale < -1000 || (x*stride+1)/scale > 1000)
            int a = 0;


    }

    confidence_.insert(confidence_.end(), confidence.begin(), confidence.end());
    bounding_box_.insert(bounding_box_.end(), bounding_box.begin(), bounding_box.end());
    regression_box_.insert(regression_box_.end(), regression_box.begin(), regression_box.end());
}


void FACE::NMS_Union(std::vector<cv::Rect>& bounding_box,float threshold)
{
    std::vector<cv::Rect> cur_rects = bounding_box;
    std::vector<float> confidence = confidence_;

    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {

            if(IoU(cur_rects[i], cur_rects[j]) > threshold)
            {
                if(confidence[i] >= confidence[j] && confidence[j] < 0.96)
               {
                    cur_rects.erase(cur_rects.begin() + j);
                    confidence.erase(confidence.begin() + j);
               }
                else if (confidence[i] < confidence[j]  && confidence[i] < 0.96)
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
                else
                {
                    j++;
                }
            }
            else
            {
                j++;
            }

        }
    }

    bounding_box = cur_rects;
    confidence_ = confidence;
}


void FACE::NMS_Min(std::vector<cv::Rect>& bounding_box)
{
    std::vector<cv::Rect> cur_rects = bounding_box;
    std::vector<float> confidence = confidence_;
    std::vector<std::vector<cv::Point> > alignment = alignment_;
    float threshold_IoM = threshold_NMS_[1];
   

    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            if( IoM(cur_rects[i], cur_rects[j]) > threshold_IoM)
            {
                if(confidence[i] >= confidence[j])
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    alignment.erase(alignment.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else if(confidence[i] < confidence[j])
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    alignment.erase(alignment.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
                else
                {
                    j++;
                }
            }
            else
            {
                j++;
            }
        }
    }

    bounding_box = cur_rects;
    confidence_ = confidence;
    alignment_ = alignment;
}




void FACE::BoxRegress(std::vector<cv::Rect>& bounding_box, std::vector<cv::Rect> regression_box,int step)
{
    
    for(int i=0;i<bounding_box.size();i++)
    {
        float regw = bounding_box[i].width;
        regw += (step == 1) ? 0 : 1;
        float regh = bounding_box[i].height;
        regh += (step == 1) ? 0 : 1;
        bounding_box[i].x += regression_box[i].x * regw;
        bounding_box[i].y += regression_box[i].y * regh;
        bounding_box[i].width += regression_box[i].width * regw;
        bounding_box[i].height += regression_box[i].height * regh;
    }
    
}

void FACE::Bbox2Square(std::vector<cv::Rect>& bounding_box){
    for(int i=0;i<bounding_box.size();i++){
        float width = bounding_box[i].width;
        float height = bounding_box[i].height;
        float side = height > width ? height:width;
        bounding_box[i].x = bounding_box[i].x + width*0.5 - side*0.5;
        bounding_box[i].y = bounding_box[i].y + height*0.5 - side*0.5;
        bounding_box[i].width = side;
        bounding_box[i].height = side;
    }

}





void FACE::Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h)
{
    for(int i=0;i<bounding_box.size();i++)
    {
        bounding_box[i].x = (bounding_box[i].x > 0)? bounding_box[i].x : 0;
        bounding_box[i].y = (bounding_box[i].y > 0)? bounding_box[i].y : 0;
        bounding_box[i].width = (bounding_box[i].x + bounding_box[i].width < img_w) ? bounding_box[i].width : img_w;
        bounding_box[i].height = (bounding_box[i].y + bounding_box[i].height < img_h) ? bounding_box[i].height : img_h;
    }
}

cv::Mat FACE::crop(cv::Mat img, cv::Rect& rect)
{
    cv::Rect rect_old = rect;


    cv::Rect padding;

    if(rect.x < 0)
    {
        padding.x = -rect.x;
        rect.x = 0;
    }
    if(rect.y < 0)
    {
        padding.y = -rect.y;
        rect.y = 0;
    }
    if(img.cols < (rect.x + rect.width))
    {
        padding.width = rect.x + rect.width - img.cols;
        rect.width = img.cols-rect.x;
    }
    if(img.rows < (rect.y + rect.height))
    {
        padding.height = rect.y + rect.height - img.rows;
        rect.height = img.rows - rect.y;
    }
    if(rect.width<0 || rect.height<0)
    {
        rect = cv::Rect(0,0,0,0);
        padding = cv::Rect(0,0,0,0);
    }
    cv::Mat img_cropped = img(rect);
    if(padding.x||padding.y||padding.width||padding.height)
    {
        cv::copyMakeBorder(img_cropped, img_cropped, padding.y, padding.height, padding.x, padding.width,cv::BORDER_CONSTANT,cv::Scalar(0));
        //here, the rect should be changed
        rect.x -= padding.x;
        rect.y -= padding.y;
        rect.width += padding.width + padding.x;
        rect.width += padding.height + padding.y;
    }


    return img_cropped;
}

void FACE::img_show(cv::Mat img, std::string name)
{
    cv::Mat img_show;
    img.copyTo(img_show);

    //cv::imwrite("../result/" + name + "test.jpg", img);

    for(int i = 0; i < bounding_box_.size(); i++)
    {
        rectangle(img_show, bounding_box_[i], cv::Scalar(0, 0, 255));
        cv::putText(img_show, std::to_string(confidence_[i]), cvPoint(bounding_box_[i].x + 3, bounding_box_[i].y + 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    }

    for(int i = 0; i < alignment_.size(); i++)
    {
        for(int j = 0; j < alignment_[i].size(); j++)
        {
            cv::circle(img_show, alignment_[i][j], 5, cv::Scalar(0, 255, 0));
        }
    }

    cv::imwrite("../result/" + name + ".jpg", img_show);

}

void FACE::img_show_T(cv::Mat img, std::string name)
{
    cv::Mat img_show;
    img.copyTo(img_show);


    for(int i = 0; i < bounding_box_.size(); i++)
    {
        rectangle(img_show, cv::Rect(bounding_box_[i].y, bounding_box_[i].x, bounding_box_[i].height, bounding_box_[i].width), cv::Scalar(0, 0, 255), 3);
        cv::putText(img_show, std::to_string(confidence_[i]), cvPoint(bounding_box_[i].y + 3, bounding_box_[i].x + 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    }

    for(int i = 0; i < alignment_.size(); i++)
    {
        for(int j = 0; j < alignment_[i].size(); j++)
        {
            cv::circle(img_show, cv::Point(alignment_[i][j].y, alignment_[i][j].x), 1, cv::Scalar(255, 255, 0), 3);
        }
    }
    
    cv::imwrite("../result/" + name + ".jpg", img_show);
    imshow("detect",img_show);
    cv::waitKey(0);
   
   
}


void FACE::detect(const cv::Mat& img, std::vector<cv::Mat>& cropFace)
{
        // std::vector<cv::Rect> rectangles;
        // cv::Mat src = img;
        // Preprocess(img);
        // STEP1_Net();
        // int numBox = bounding_box_.size();
        // if(numBox !=0){
        //     NMS_Union(bounding_box_,threshold_NMS_[1]);
        //     BoxRegress(bounding_box_, regression_box_,1);
        // }

        // numBox = bounding_box_.size();
        // if(numBox !=0){
        //     STEP2_Net();
        //     numBox = bounding_box_.size();
        //     if(numBox !=0){
        //         NMS_Union(bounding_box_,threshold_NMS_[1]);
        //         BoxRegress(bounding_box_, regression_box_,2);
        //     }
        //     numBox = bounding_box_.size();
        //     if(numBox !=0){
        //         STEP3_Net();
        //         numBox = bounding_box_.size();
        //         if(numBox !=0){
        //             BoxRegress(bounding_box_, regression_box_,3);
        //             NMS_Min(bounding_box_);

        //         }
        //     }                   
           
        // }

     timeval start, end; 
     unsigned  long t;


     std::vector<cv::Rect> rectangles;
     cv::Mat src = img;
     IplImage *gray = NULL;  
     const int rects_num = 10;
     struct square_rect rects[10];
     struct havon_xffd *ffd = havon_xffd_create(128, 128*4);
     Preprocess(img);
     IplImage tmp = img;
     IplImage *frame = cvCloneImage(&tmp);
     if (!gray) {
       gray = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);
     }
     cvCvtColor(frame, gray, CV_RGB2GRAY);  
     uint32_t num_saved = 0;
     //gettimeofday(&start, NULL); 
     havon_xffd_detect(ffd, (const uint8_t *)gray->imageData, gray->width, gray->height, gray->widthStep, rects, rects_num, &num_saved);
    // gettimeofday(&end, NULL); 
    // t=1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
    // cout<<"time is "<<t<<endl;
    for (uint32_t idx = 0; idx < num_saved; ++idx) {
      // if (rects[idx].score > 5.0) {
       // cvCircle(frame, cvPoint(rects[idx].cx, rects[idx].cy), rects[idx].size/2, CV_RGB(255,0,0), 4,8,0);}           
             bounding_box_.push_back(cv::Rect(rects[idx].cy-rects[idx].size/2,rects[idx].cx - rects[idx].size/2,rects[idx].size, rects[idx].size) );
         }
     //cvShowImage("xffd", frame);
    // cvWaitKey(0);
    cvReleaseImage(&gray);
    cvReleaseImage(&frame);
      STEP3_Net();
      NMS_Min(bounding_box_);





    for(int i = 0; i < bounding_box_.size(); i++)
    {
        rectangles.push_back(cv::Rect(bounding_box_[i].y, bounding_box_[i].x, bounding_box_[i].height, bounding_box_[i].width));
    }
 
   gettimeofday(&start, NULL); 
    std::vector<cv::Point2f> src5Points[alignment_.size()];
    cv::Point2f srctemp[5];
    cv::Point2f dst5Points[5];
    cv::Mat dst;
    cv::Point2f temp;

    for(int j = 0; j < alignment_.size(); j++)
    {
        for(int k = 0; k < 5; k++)
        {
            temp.x = alignment_[j][k].y;
            temp.y = alignment_[j][k].x;
            src5Points[j].push_back(temp);
        }
    }

    dst5Points[0] = cv::Point2f(30.2946,51.6963);
    dst5Points[1] = cv::Point2f(65.5318,51.5014);
    dst5Points[2] = cv::Point2f(48.0252,71.7366);
    dst5Points[3] = cv::Point2f(33.5493,92.3655);
    dst5Points[4] = cv::Point2f(62.7299,92.2041);
  
  
   for(int j = 0; j < alignment_.size(); j++)
   { 
        srctemp[0] = src5Points[j][0];
        srctemp[1] = src5Points[j][1];
        srctemp[2] = src5Points[j][2];
        srctemp[3] = src5Points[j][3];
        srctemp[4] = src5Points[j][4];

        Mat warpMat = getAffineTransformOverdetermined(srctemp, dst5Points,5);
         warpAffine(src, dst, warpMat, cv::Size(96, 112)); 
       // dst = crop_face(src, srctemp, dst5Points);
        cropFace.push_back(dst);
     
   }

     gettimeofday(&end, NULL); 
     t=1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
     //cout<<"time is "<<t<<endl;

}



void FACE::calculateDistance(std::vector<float> output1, std::vector<float> output2,float& l2)//modify
{

    float sqrtsum1 = 0;
    float sqrtsum2 = 0;

    for(int i = 0; i < 256; i++)//for sum
    {
        sqrtsum1 += (output1[i] * output1[i]);
        sqrtsum2 += (output2[i] * output2[i]);
    }

    sqrtsum1 = sqrt(sqrtsum1);//for square root
    sqrtsum2 = sqrt(sqrtsum2);

    for(int i = 0; i < 256; i++)//for normalization
    {
        output1[i] /= sqrtsum1;
        output2[i] /= sqrtsum2;
    }

    float distance_ = 0;
    for(int i = 0; i < 256; i++)
    {
        distance_ += ((output1[i] - output2[i]) * (output1[i] - output2[i]));
    }
    distance_= sqrt(distance_);
    l2 = distance_;
}


void FACE::interpolateCubic(float x, float* coeffs)
{
	const float A = -0.5f;

	coeffs[0] = ((A*(x + 1) - 5 * A)*(x + 1) + 8 * A)*(x + 1) - 4 * A;
	coeffs[1] = ((A + 2)*x - (A + 3))*x*x + 1;
	coeffs[2] = ((A + 2)*(1 - x) - (A + 3))*(1 - x)*(1 - x) + 1;
	coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

cv::Mat FACE::crop_face(const cv::Mat& image, cv::Point2f key_pt[5], cv::Point2f base_pt[5])
{
	
	float crop_size[2] = { 112, 96 };

	
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
		float x = sc*j + ss*i + tx;
		float y = -ss*j + sc*i + ty;
		int xx = int(x);
		int yy = int(y);

		float coeffs_x[4], coeffs_y[4];
		interpolateCubic(x - xx, coeffs_x);
		interpolateCubic(y - yy, coeffs_y);

		xx = xx - 1;
		yy = yy - 1;
		float b[3] = { 0, 0, 0 };
		for (int r = 0; r < 4; r++)
		{
			float a[3] = { 0, 0, 0 };
			for (int c = 0; c < 4; c++)
			{
				float p[3] = { 127.5, 127.5, 127.5 };
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
	//imwrite("crop.jpg", crop_img);
	return crop_img;
}
