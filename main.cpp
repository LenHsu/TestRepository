#include <QCoreApplication>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <qdebug.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "lib.h"

#define qDebug_float() qDebug() << fixed << qSetRealNumberPrecision(4)
struct wstruct {
    int top, bottom, left, right;
};
struct vector_disp {
    int size;
    float* vecs;
};

typedef struct wstruct WINDOW;
float param[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
int dir = 0;
float *Sum, *Diff;

float Mean;
using namespace cv;
using namespace std;
IplImage *toOpenCV (IMAGE x)
{
    IplImage *img;
    int i=0, j=0;
    CvScalar s;

    img = cvCreateImage(cvSize(x->info->nc, x->info->nr),8, 1);
    for (i=0; i<x->info->nr; i++)
    {
        for (j=0; j<x->info->nc; j++)
        {
            s.val[0] = x->data[i][j];
            cvSet2D (img, i,j,s);
        }
    }
    return img;
}

IMAGE fromOpenCV (IplImage *x)
{
    IMAGE img;
    int color=0, i=0;
    int k=0, j=0;
    CvScalar s;

    if ((x->depth==IPL_DEPTH_8U) &&(x->nChannels==1))								// 1 Pixel (grey) image
        img = newimage (x->height, x->width);
    else if ((x->depth==8) && (x->nChannels==3)) //Color
    {
        color = 1;
        img = newimage (x->height, x->width);
    }
    else return 0;

    for (i=0; i<x->height; i++)
    {
        for (j=0; j<x->width; j++)
        {
            s = cvGet2D (x, i, j);
            if (color)
              k = (unsigned char)((s.val[0] + s.val[1] + s.val[2])/3);
            else k = (unsigned char)(s.val[0]);
            img->data[i][j] = k;
        }
    }
    return img;
}

/* Display an image on the screen */
void display_image (IMAGE x)
{
    IplImage *image = 0;
    char wn[20];
    int i;

    image = toOpenCV (x);
    cv::Mat m = cv::cvarrToMat(image);
    if (image <= (IplImage *)0) return;

    for (i=0; i<19; i++) wn[i] = (char)((drand32()*26) + 'a');
    wn[19] = '\0';
    cv::namedWindow( wn, cv::WINDOW_AUTOSIZE);
    cv::imshow(wn, m );
    cv::waitKey(0);
    //cvReleaseImage( &image );
}

float *Ps, *Pd;
void glcm (Mat image, int d, WINDOW *w)
{
    int ngl, p1, p2, i, j, k;
    float sum;
    uchar* im = image.data;
    int img_width = image.cols;

/* Assume that there are 256 grey levels */
    ngl = 256;

/* Allocate the matrix */
    if (Ps == 0)
    {
      Ps = (float *)calloc (ngl*2, sizeof(float));
      Pd = (float *)calloc (ngl*2, sizeof(float));
      Sum = Ps;
      Diff = Pd;
    }
    dir = (int)param[4];

/* Compute the histograms for any of 4 directions */
    k = 0;
    for (i = w->top; i < w->bottom; i++) {
      for (j = w->left; j < w->right; j++)
      {
        p1 = im[i*img_width + j];

        if (j+d < w->right && dir == 0)		/* Horizontal */
        p2 = im[i*img_width + j + d];
        else if (i+d < w->bottom && dir == 2)		/* Vertical */
        p2 = im[(i+d) * img_width + j];
        else if (i+d < w->bottom && j-d >= w->left && dir == 1)	/* 45 degree diagonal */
            p2 = im[(i+d) * img_width + j-d];
        else if (i+d < w->bottom &&
        j+d < w->right && dir == 3)		/*135 degree diagonal */
        p2 = im[(i+d) * img_width + j+d];
        else continue;
        k++;

        Ps[p1+p2]++;
        Pd[p1-p2+ngl]++;
#if 0
        qDebug() << "p1 = " << p1 <<" p2 = "<< p2 << " PS = " << Ps[p1+p2] << endl;
#endif
      }
    }
}

void glmc_img(Mat image, int d, int dir)
{
    int width = image.cols;
    int height = image.rows;
    int p1 = 0 , p2 = 0;
    qDebug() << "image width = " << width << " height = " << height << endl;
    uchar* im = image.data;
    /* Assume that there are 256 grey levels */
    int ngl = 256;

    if (Ps == 0)
    {
        Ps = (float *)calloc (ngl*2, sizeof(float));
        Pd = (float *)calloc (ngl*2, sizeof(float));
        Sum = Ps;
        Diff = Pd;
    }
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            p1 = im[i*width + j];
            if(dir == 0) { //horizontal
                if(j + d < width) {
                    p2 = im[i*width + j + d];
                }
            } else if(dir == 1) { //vertical
                if(i + d <= height) {
                    p2 = im[(i+d)*width + j];
                }
            } else if(dir == 2) { //diagonal 45
                if(i + d <= height && j - d >= 0) {
                    p2 = im[(i+d)*width + j - d];
                }
            } else { //diagonal 135
                if(i + d <= height && j + d <= width) {
                    p2 = im[(i+d)*width + j + d];
                }
            }
            Ps[p1+p2]++;
            Pd[p1-p2+ngl]++;
        }
    }
    /* Normalize */
    float sum = 0.0;
    for (int i=0; i < ngl * 2; i++)
    {
        Ps[i] /= width * height;
        sum += Ps[i]*i;
        Pd[i] /= width * height;
     }
    //Mean = sum/2.0f;
#if 0
    for(int i = 0; i < 512; i++) {
        qDebug_float() << "i = " << i << " Sum = " << Sum[i] << " Ps = " << Ps[i] << endl;
    }
#endif
}

float average ()
{
    return Mean;
}

float stddev ()
{
    float s1, s2, meanmean, var;
    int i,j;

    s1 = s2 = 0.0;
    meanmean = Mean+Mean;

    for (i=0; i<512; i++)
    {
      s1 += (i-meanmean)*(i-meanmean)*Sum[i];
      j = i-255;
      s2 += j*j*Diff[i];
    }
    var = (s1 + s2)/2.0f;
    return (float)sqrt ((double)var);
}

float p_max ()
{
    int i;
    float y;

    y = 0.0;
    for (i=0; i<512; i++)
        if (Sum[i] > y) y = Sum[i];
    return y;
}

float energy ()
{
    int i;
    float y=0.0, z = 0.0;

    for (i=0; i<512; i++)
    {
      y += Sum[i]*Sum[i];
      z += Diff[i]*Diff[i];
    }
    return y*z;
}

float contrast ()
{
    int i;
    float y=0.0;

    for (i= -255; i<= 255; i++)
      y += i*i*Diff[i+255];
    return y;
}


float homo ()
{
    int i;
    float y = 0.0;

    for (i= -255; i<255; i++)
    {
      y += 1.0f/(1.0f + i*i) * Diff[i+255];
    }
    return y;
}

float entropy ()
{
    int i;
    float y, h1=0.0, h2=0.0;

    y = 0.0;
    for (i=0; i<512; i++)
      if (Sum[i] > 0)
        h1 += Sum[i]*(float)log((double)Sum[i]);
    for (i=-255; i<= 255; i++)
      if (Diff[i+255] > 0)
        h2 += Diff[i+255]*(float)log((double)Diff[i+255]);
    y = -h1 - h2;

    return y;
}

void lsbfp (Mat image, int row, int col, float *a, float *b, float *c)
{
    int i,j, I, J, k;
    int width = image.cols;
    int g = 0, alp=0, bet=0, r2=0, c2=0;
    uchar* im = image.data;
    for (i=row-1; i<=row+1; i++) {
        for (j=col-1; j<=col+1; j++) {
            k = (int)im[i*width + j];
            I = i-row;
            J = j-col;
            g += k;
            alp += I*k;
            r2 += I*I;
            bet += J*k;
            c2 += J*J;
         }
    }
    *c = (float)g/9.0f;
    *b = (float)bet/(float)c2;
    *a = (float)alp/(float)r2;
}

/* Vector dispersion - fit planar surface to each region, use normals	*/

float vd_window(Mat image, int win_size, int row, int col)
{
    int i,j, n=0;
    float x=0, a=0, b=0, c=0, sa=0, sb=0, sc=0;
    float r2=0, r;
    int row_begin = row;
    int row_end = row + win_size + 1;
    int col_begin = col;
    int col_end = col + win_size + 1;

/* Compute the surface normal for each 3x3 region in the window */
    for (i = row_begin + 1; i <= row_end; i += 3) {
      for (j = col_begin + 1; j <= col_end; j += 3)
      {
        lsbfp (image, i, j, &a, &b, &c);

/* Normalize and average */
        x = (float)sqrt((double)(a*a + b*b + 1));
        //ki[n] = a / x;
        //li[n] = b / x;
        //mi[n] = (-1.0f)/x;
        sa += a/x;
        sb += b/x;
        sc += (-1.0f)/x;
#if 0
        qDebug() << "i " << i  << "j " << j << "n " << n << "ki " << a / x << "Li " << b / x << "Mi " << -1.0f/x <<  endl;
#endif
        n++;
      }
    }

    r2 = sa*sa + sb*sb + sc*sc;
    r = (float)sqrt ((double)r2);

/* Compute the descriptor Kappa */
    x = (float)(n - 1)/(float)(n-r);
    return x;
}

struct vector_disp * vector_dispersion(Mat img, int win_size)
{
    int col_size = img.cols;
    int row_size = img.rows;
    struct vector_disp* v_disp = (struct vector_disp *)malloc(sizeof(struct vector_disp));
    v_disp->size = (col_size) * (row_size);
    v_disp->vecs = (float *)calloc(v_disp->size, sizeof(float));
    for(int i = 0; i < row_size - win_size; i++) {
        for(int j = 0; j < col_size - win_size; j++) {
            v_disp->vecs[i * col_size + j] = vd_window(img, win_size, i, j);
            qDebug() <<"row " << i << " col " << j << " value " << v_disp->vecs[i * col_size + j] <<  endl;
        }
    }
    qDebug() <<"vector_dispersion " << endl;
    return v_disp;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    Mat img = imread("C:/Users/minia/Documents/academy/book/img7.jpg", IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
    }
#if 0
    for(int i = 0; i < img.cols * img.rows; i++) {
        if(img.data[i] >= 128) img.data[i] = 255;
        else img.data[i] = 0;
        //qDebug() << "i = " << i << " value = " << img.data[i] << endl;
    }
#endif
    //glmc_img(img, 1, 0);
    //qDebug() << "entropy " << entropy() << "stdv " << stddev() <<  endl;
    struct vector_disp * v_disp = vector_dispersion(img, 6);
#if 0
    for(int i = 0; i < v_disp->size; i++) {
         qDebug() << "vector i " << i <<" value " << v_disp->vecs[i] <<  endl;
    }
#endif

    imshow("Image", img);

    return a.exec();
}
