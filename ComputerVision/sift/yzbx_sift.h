#ifndef YZBX_SIFT_H
#define YZBX_SIFT_H
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <math.h>

#define _DEBUG 0

#if _DEBUG
//#define LOG_MESSAGE(x) std::cout << __FILE__ <<__FUNCTION__<< " (" << __LINE__ << "): " << x << std::endl;
#define LOG_MESSAGE(x) std::cout <<__FUNCTION__<< " (" << __LINE__ << "): " << x << std::endl;
#else
#define LOG_MESSAGE(x)
#endif

typedef double sift_wt;
using namespace std;
using namespace cv;

const int yzbx_debug_row=32;
const int yzbx_debug_col=94;

//the sift code from opencv, for debug!!!

static void calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scl,
                               int d, int n, float* dst );
static void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave );
void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale);
static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma );
static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n );

static bool adjustLocalExtrema(const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrastThreshold, float edgeThreshold, float sigma , Mat &keyPointsMat);

class yzbx_sift
{
public:
    yzbx_sift();
    void detectKeyPoints(const Mat &bgrOrGrayImage,vector<KeyPoint> &keyPoints);
    void computerDescriptors(const Mat &bgrOrGrayImage, const vector<KeyPoint> &keyPoints, Mat &descriptorMat);
    void matchDescriptors(const Mat &descriptors1,const Mat &descriptors2,vector<DMatch> matches);

private:
    void debug(InputArray _image,
                          vector<KeyPoint>& keypoints,
                          OutputArray _descriptors,
                          bool useProvidedKeypoints);
    void debug( const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
                                      vector<KeyPoint>& keypoints );

    void generateGaussOctaves(const Mat &inputMat,vector<vector<Mat>> &GaussOctaves,size_t octaveNum,size_t scaleNum);
    void generateDOGOctaves(const vector<vector<Mat>> &GaussOctaves,vector<vector<Mat>> &DOGOctaves);
    void getDOGMat(const Mat &inputMat,const Mat &inputMat_pre,Mat &DOGMat);
    void gaussConvolution(const Mat &inputMat32F,Mat &outputMat32F,double sigma);
    void findKeyPoints(const vector<vector<Mat>> &DOGOctaves, const vector<vector<Mat> > &GaussOctaves, vector<KeyPoint> &keyPoints);
    bool isInterpExtremun(const vector<vector<Mat>> &DOGOctaves, int oi, int &sj, int &i, int &j, double sigma, KeyPoint &kp, Mat &debugMat);
    void getInterpMat(const vector<vector<Mat>> &DOGOctaves,int oi,int sj,int i,int j,Mat &interpMat);
    void getHessianMat(const vector<vector<Mat>> &DOGOctaves,int oi,int sj,int i,int j,Mat &hessianMat);
    void getGradientMat(const vector<vector<Mat>> &DOGOctaves,int oi,int sj,int i,int j,Mat &gradientMat);
    double getOrientationHist(const vector<vector<Mat>> &GaussOctaves, int oi, int sj, int i, int j, int radius, double sigma, vector<double> &hist);
    void showOcatave(int i, int j, const Mat &mat, string addtionStr="");

    void setDescriptorMat(Mat &descriptorMat,vector<vector<vector<double>>> &hist,int rowNum);
    void getDescriptorsHist(const Mat&img,int r,int c,int descriptorWidth,double mainOritation,double size,vector<vector<vector<double>>> &hist);
    bool getDescriptorGradientInfo(const Mat&img,int r,int c,double &mag,double& ori);
    void interpDescriptorHist(vector<vector<vector<double>>> &hist, double rbin, double cbin, double obin, double mag, int descriptorWidth, int descriptorHistBinNum);

    //CONFIG
    //the octave number for guass octave
    size_t defaultOctaveNum;
    //the layer of each octave for guass octave
    size_t defaultScaleNum;
    //the sigma ratio between octaves.
    double defaultOctaveRatio;
    //the sigma ratio between layers in the same octave.
    double defaultScaleRatio;
    //the sigma for first layer of first octave.
    double defaultSigma=1/sqrt(2.0);

    double defaultContrastThreshold;
    double defaultEdgeThresholdRaw;
    double defaultEdgeThresholdFinal;
    double defaultInterpTimes;
    double defaultContrastThresholdAfterInterp;
    int defaultImageBorder;
    int defaultHOGBinNum;
    double defaultHistPeakRatio;
    int defaultDescriptorHistBinNum;
    int defaultDescriptorWidth;

    int debugNum=0;
    vector<vector<Mat>> globalGaussOctaves;
    vector<vector<Mat>> debugMatVector;
};

#endif // YZBX_SIFT_H
