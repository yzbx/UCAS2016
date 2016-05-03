#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include "yzbx_sift.h"
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    yzbx_sift yzbx_sift;
    vector<Mat> imgs,descriptorsMats;
    vector<vector<KeyPoint>> kpsVector;
    Mat img1=imread("1.JPG");
    Mat img2=imread("2.JPG");

    if(img1.empty()||img2.empty()){
        cout<<"cannot find picture 1.JPG or 2.JPG"<<endl;
    }

    //step 1: read image and show image
    cout<<"step 1: read image and show image"<<endl;
    imshow("img1",img1);
    imshow("img2",img2);

    imgs.push_back(img1);
    imgs.push_back(img2);

//    cv::SIFT cvsift;
//    SiftDescriptorExtractor cvDE;
//    bool useCVSIFT=false;

    //step 2: detect keypoint and compute descriptors
    for(int i=0;i<2;i++){
        cout<<"step 2 for img"<<i+1<<"*************************"<<endl;

        //step 2.1: detect key points
        cout<<"step 2.1: detect key points"<<endl;
        vector<KeyPoint> kps;
        yzbx_sift.detectKeyPoints(imgs[i],kps);

        kpsVector.push_back(kps);

        //step 2.2: draw key points
        cout<<"step 2.2: draw key points"<<endl;
        Mat keyPointMat;
        drawKeypoints(imgs[i],kps,keyPointMat,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        stringstream ss;
        ss<<i;
        string name=ss.str();
        imshow("keyPoints-"+name,keyPointMat);

        //step 2.3: compute descriptors
        cout<<"step 2.3: compute descriptors"<<endl;
        Mat descriptorsMat;
        yzbx_sift.computerDescriptors(imgs[i],kps,descriptorsMat);
        descriptorsMats.push_back(descriptorsMat);
    }

    //step 3: match key points
    cout<<"step 3: match key points"<<endl;
    Mat matchMat;
    vector<DMatch> matches;
    BFMatcher bf(NORM_L2,true);
    bf.match(descriptorsMats[0],descriptorsMats[1],matches);
    drawMatches(imgs[0],kpsVector[0],imgs[1],kpsVector[1],matches,matchMat);
    imshow("matchMat",matchMat);

    cv::waitKey(0);
    return 0;
}
