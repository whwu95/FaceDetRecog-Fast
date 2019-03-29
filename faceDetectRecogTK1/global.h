#ifndef GLOBAL_H
#define GLOBAL_H
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace std;
namespace std
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}



using namespace cv;
namespace cv
{
    cv::Mat getAffineTransformOverdetermined( const Point2f src[], const Point2f dst[], int n )
    {
        Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.data); // output
        double* a = (double*)malloc(12*n*sizeof(double));
        double* b = (double*)malloc(2*n*sizeof(double));
        Mat A(2*n, 6, CV_64F, a), B(2*n, 1, CV_64F, b); // input

        for( int i = 0; i < n; i++ )
        {
            int j = i*12;   // 2 equations (in x, y) with 6 members: skip 12 elements
            int k = i*12+6; // second equation: skip extra 6 elements
            a[j] = a[k+3] = src[i].x;
            a[j+1] = a[k+4] = src[i].y;
            a[j+2] = a[k+5] = 1;
            a[j+3] = a[j+4] = a[j+5] = 0;
            a[k] = a[k+1] = a[k+2] = 0;
            b[i*2] = dst[i].x;
            b[i*2+1] = dst[i].y;
        }
        solve( A, B, X, DECOMP_SVD );
        delete a;
        delete b;
        return M;
    }
}

#endif
