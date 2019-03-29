#include "havon_ffd.h"
#include <opencv2/opencv.hpp>

#ifdef __GNUC__
#include <time.h>
float getticks()
{
	struct timespec ts;

	if(clock_gettime(CLOCK_MONOTONIC, &ts) < 0)
		return -1.0f;

	return ts.tv_sec + 1e-9f*ts.tv_nsec;
}
#else
#include <windows.h>
float getticks()
{
	static double freq = -1.0;
	LARGE_INTEGER lint;

	if(freq < 0.0)
	{
		if(!QueryPerformanceFrequency(&lint))
			return -1.0f;

		freq = lint.QuadPart;
	}

	if(!QueryPerformanceCounter(&lint))
		return -1.0f;

	return (float)( lint.QuadPart/freq );
}
#endif


int
main(int argc, char *argv[]) {
    CvCapture *cap = NULL;
    IplImage *frame, *gray = NULL;
    int stop = 0, key = 0;
    uint32_t idx = 0;
    const int rects_num = 10;
    struct square_rect rects[10];

    struct havon_xffd *ffd = havon_xffd_create(128, 128*4);
    if (!ffd) return -1;
    
    cap = cvCaptureFromFile("/home/hui/Downloads/C270.mov");
    // cap = cvCaptureFromCAM(0);
    
    if (!cap) return -1;

    while (!stop) {
        if (!cvGrabFrame(cap)) {
            stop = 1;
            frame = NULL;
        }
        else {
            frame = cvRetrieveFrame(cap, 1);
        }
        
        if (!frame || key == 'q') {
            stop = 1;
        }
        else {
            uint32_t num_saved = 0;
            if (!gray) {
                gray = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);
            }

            cvCvtColor(frame, gray, CV_RGB2GRAY);
            float t = getticks();
            havon_xffd_detect(ffd, (const uint8_t *)gray->imageData, gray->width, gray->height, gray->widthStep, rects, rects_num, &num_saved);
            t = getticks() - t;
            printf("# %f ms\n", 1000.0f*t); // use '#' to ignore this line when parsing the output of the program
            // stop = 1;
            // printf("num_saved: %d\n", num_saved);
            for (idx = 0; idx < num_saved; ++idx) {
                if (rects[idx].score > 5.0) {
                    cvCircle(frame, cvPoint(rects[idx].cx, rects[idx].cy), rects[idx].size/2, CV_RGB(255,0,0), 4,8,0);
                }
            }
            
            cvShowImage("xffd", frame);
        }
        
        key = cvWaitKey(5);
    }

    cvReleaseImage(&gray);
    cvReleaseCapture(&cap);
    havon_xffd_destroy(ffd);
    return 0;
}
