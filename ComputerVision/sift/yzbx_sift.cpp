#include "yzbx_sift.h"

yzbx_sift::yzbx_sift()
{
    //nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6
    //    int nOctaveLayers=2;
    defaultOctaveNum=8;
    defaultScaleNum=6;
    defaultOctaveRatio=2.0;
    defaultScaleRatio=pow(2,1.0/(defaultScaleNum-3));
    defaultSigma=1.6;
    //0.01
    defaultContrastThreshold=0.04;
    defaultEdgeThresholdRaw=10.0;
    defaultEdgeThresholdFinal=(1+defaultEdgeThresholdRaw)*(1+defaultEdgeThresholdRaw)/defaultEdgeThresholdRaw;
    defaultInterpTimes=5;
    defaultContrastThresholdAfterInterp=0.02;
    defaultImageBorder=5;
    defaultHOGBinNum=36;
    defaultHistPeakRatio=0.8;
    defaultDescriptorHistBinNum=8;
    defaultDescriptorWidth=4;
}

void yzbx_sift::detectKeyPoints(const Mat &bgrOrGrayImage,vector<KeyPoint> &keyPoints){   
    //step 1: bgr --> gray
    Mat inputMat;
    CV_Assert(!bgrOrGrayImage.empty());
    if(bgrOrGrayImage.type()==CV_8UC1){
        inputMat=bgrOrGrayImage;
    }
    else if(bgrOrGrayImage.type()==CV_8UC3){
        cvtColor(bgrOrGrayImage,inputMat,CV_BGR2GRAY);
    }
    else{
        cout<<"the bgrOrGrayImage must be CV_8UC3 or CV_8UC1"<<endl;
        CV_Assert(false);
    }

    //step 2: median blur, remove noise, resize the image.
    //    medianBlur(inputMat,inputMat,5);
//    double sigma_dif=sqrt(max(defaultSigma*defaultSigma-0.5*0.5,0.01));
//    GaussianBlur(inputMat,inputMat,Size(),sigma_dif);
    Mat base=createInitialImage(inputMat,true,1.6);
    base.convertTo(inputMat,CV_8UC1);

    //step 3: create gaussian octave and dog octave

    vector<vector<Mat>> GaussOctaves;
    vector<vector<Mat>> DOGOctaves;

//    int octaveNum=defaultOctaveNum;
    int octaveNum=cvRound(log( (double)std::min( base.cols, base.rows ) ) / log(2.) - 2);
    int scaleNum=defaultScaleNum;
    generateGaussOctaves(inputMat,GaussOctaves,octaveNum,scaleNum);

    generateDOGOctaves(GaussOctaves,DOGOctaves);

    //step 4: find key points
    findKeyPoints(DOGOctaves,GaussOctaves,keyPoints);

    //step 5: remove duplicated keypoints, retain the best k feature.

    int firstOctave=-1;
    if( firstOctave < 0 )
        for( size_t i = 0; i < keyPoints.size(); i++ )
        {
            KeyPoint& kpt = keyPoints[i];
            float scale = 1.f/(float)(1 << -firstOctave);
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
            kpt.pt *= scale;
            kpt.size *= scale;
        }


    int siz=keyPoints.size();
    KeyPointsFilter::removeDuplicated( keyPoints );
    CV_Assert(keyPoints.size()<siz);
//    KeyPointsFilter::retainBest(keyPoints,100);

    globalGaussOctaves.clear();
    globalGaussOctaves=GaussOctaves;
    CV_Assert(!globalGaussOctaves.empty());
    CV_Assert(!globalGaussOctaves[0].empty());
    CV_Assert(!globalGaussOctaves[0][0].empty());
}
void yzbx_sift::generateGaussOctaves(const Mat &inputMat, vector<vector<Mat> > &GaussOctaves, size_t octaveNum, size_t scaleNum){
    LOG_MESSAGE(0);
    CV_Assert(!inputMat.empty());
    CV_Assert(inputMat.type()==CV_8UC1);

    size_t img_cols=inputMat.cols;
    size_t img_rows=inputMat.rows;

    defaultOctaveNum=octaveNum;
    defaultScaleNum=scaleNum;

    CV_Assert(img_cols>pow(2,octaveNum));
    CV_Assert(img_rows>pow(2,octaveNum));

    for(int i=0;i<octaveNum;i++){
        vector<Mat> octaveMats;
        for(int j=0;j<scaleNum;j++){
            Mat M;
            octaveMats.push_back(M);
        }
        GaussOctaves.push_back(octaveMats);
    }


    vector<double> sigmaDifVec(scaleNum);
    sigmaDifVec[0]=defaultSigma;
    double k = pow( 2., 1. / (scaleNum-3) );
    for( int i = 1; i < scaleNum; i++ )
    {
        double sig_prev = pow(k, (double)(i-1))*defaultSigma;
        //the true sigma for image if guass blur on the first image.
        double sig_total = sig_prev*k;
        //the sigma for image if guass blur on the previous image.
        sigmaDifVec[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }


    for(int i=0;i<octaveNum;i++){
        for(int j=0;j<scaleNum;j++){
            if(i==0&&j==0){
                inputMat.convertTo(GaussOctaves[0][0],CV_64FC1,1.0/255);
            }
            else if(j==0){
                CV_Assert(!GaussOctaves[i-1][j].empty());
                cv::resize(GaussOctaves[i-1][scaleNum-3],GaussOctaves[i][j],Size(),0.5,0.5,INTER_LINEAR);
            }
            else{
                CV_Assert(!GaussOctaves[i][j-1].empty());
                GaussianBlur(GaussOctaves[i][j-1],GaussOctaves[i][j],Size(),sigmaDifVec[j]);
            }
        }

    }

    //    imshow("gaussOctave[0][0]",GaussOctaves[0][0]);
    //    imshow("gaussOctave[0][1]",GaussOctaves[0][1]);

    //    for(int oi=0;oi<octaveNum;oi++){
    //        int scaleNum=GaussOctaves[oi].size();
    //        for(int sj=0;sj<scaleNum;sj++){
    //           showOcatave(oi,sj,GaussOctaves[oi][sj],"Gauss-");
    //        }
    //    }

}

void yzbx_sift::gaussConvolution(const Mat &inputMat32F, Mat &outputMat32F, double sigma){
    LOG_MESSAGE(0);
    CV_Assert(!inputMat32F.empty());
    CV_Assert(inputMat32F.type()==CV_64FC1);
    outputMat32F.release();
    outputMat32F.create(inputMat32F.size(),CV_64FC1);

    const int radius=(int)ceil(3*sigma);
    const int kernalSize=radius*2+1;
    Mat kernelMat(kernalSize,kernalSize,CV_64FC1);
    double PI=3.1415926;
    double A=1.0/(2*PI*sigma*sigma);
    double B=1.0/(2*sigma*sigma);
    for(int i=0;i<kernalSize;i++){
        for(int j=0;j<kernalSize;j++){
            double x=i-radius;
            double y=j-radius;
            kernelMat.at<double>(i,j)=(double)A*exp(-(x*x+y*y)*B);
        }
    }
    int img_rows=inputMat32F.rows;
    int img_cols=inputMat32F.cols;
    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){

            double sumKernel=0;
            double sumConvolution=0;
            for(int a=0;a<kernalSize;a++){
                for(int b=0;b<kernalSize;b++){
                    int dx=a-radius;
                    int dy=b-radius;
                    int x=dx+i;
                    int y=dy+j;
                    if(x>=0&&x<img_rows&&y>=0&&y<img_cols){
                        sumKernel+=kernelMat.at<double>(a,b);
                        sumConvolution+=(kernelMat.at<double>(a,b)*inputMat32F.at<double>(x,y));
                    }
                }
            }

            outputMat32F.at<double>(i,j)=(double)(sumConvolution/sumKernel);
        }
    }
}

void yzbx_sift::generateDOGOctaves(const vector<vector<Mat> > &GaussOctaves, vector<vector<Mat> > &DOGOctaves){
    LOG_MESSAGE(0);
    int octaveNum=GaussOctaves.size();
    for(int i=0;i<octaveNum;i++){
        int scaleNum=GaussOctaves[i].size();
        vector<Mat> ocataveMats;
        for(int j=1;j<scaleNum;j++){
            Mat DOGMat;
            CV_Assert(!GaussOctaves[i][j].empty());
            CV_Assert(!GaussOctaves[i][j-1].empty());
            //            getDOGMat(GaussOctaves[i][j],GaussOctaves[i][j-1],DOGMat);
            DOGMat=GaussOctaves[i][j]-GaussOctaves[i][j-1];
            CV_Assert(!DOGMat.empty());
            ocataveMats.push_back(DOGMat);
        }
        DOGOctaves.push_back(ocataveMats);
    }

    //    for(int oi=0;oi<octaveNum;oi++){
    //        int scaleNum=DOGOctaves[oi].size();
    //        for(int sj=0;sj<scaleNum;sj++){
    //           showOcatave(oi,sj,DOGOctaves[oi][sj],"DOG-");
    //        }
    //    }
}

void yzbx_sift::getDOGMat(const Mat &inputMat, const Mat &inputMat_pre, Mat &DOGMat){
    LOG_MESSAGE(0);
    CV_Assert(!inputMat.empty());
    CV_Assert(!inputMat_pre.empty());
    DOGMat.release();
    DOGMat.create(inputMat.size(),CV_64FC1);

    int img_rows=inputMat.rows;
    int img_cols=inputMat.cols;

    //NOTE what is the DOGMat
    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            double dif=abs(inputMat.at<double>(i,j)-inputMat_pre.at<double>(i,j));
            if(dif<16){
                dif=dif*dif;
            }
            else{
                dif=255;
            }
        }
    }


}

void yzbx_sift::findKeyPoints(const vector<vector<Mat> > &DOGOctaves,const vector<vector<Mat>> &GaussOctaves,vector<KeyPoint> &keyPoints){
    LOG_MESSAGE(0);
    //what's key points?

    //1. max/min in 26 neighboorhood
    //2. removing low constrast features
    //3. not edge points
    //4. predict the ture max/min location.

    double maxCenter=0;
    static int count=0;
    vector<vector<Mat>> keyPointsOcataves;
    double threshold=defaultContrastThreshold*0.5/(defaultScaleNum-3.0);
    int octaveNum=DOGOctaves.size();
    LOG_MESSAGE("octaveNum="<<octaveNum);
    for(int oi=0;oi<octaveNum;oi++){
        int scaleNum=DOGOctaves[oi].size();
        vector<Mat> keyPointsOcataveMats;
        for(int sj=1;sj<scaleNum-1;sj++){
            CV_Assert(DOGOctaves[oi][sj].type()==CV_64FC1);
            //            Mat MatA=DOGOctaves[oi][sj-1];
            Mat MatB=DOGOctaves[oi][sj];
            //            Mat MatC=DOGOctaves[oi][sj+1];

            CV_Assert(!MatB.empty());
            Mat keyPointsMat=Mat::zeros(MatB.size(),CV_8UC1);
            CV_Assert(!keyPointsMat.empty());

            //for each point in MatB, find the key points in valid area.
            int img_rows=MatB.rows;
            int img_cols=MatB.cols;
//            double sigma=defaultSigma*pow(defaultScaleRatio,sj)*pow(defaultOctaveRatio,oi);
            double sigma=1.6;
            //            int radius=(int)ceil(3*sigma);
            int radius=defaultImageBorder;

            //the borden aree is not valid area.
            for(int i=radius;i<img_rows-radius;i++){
                for(int j=radius;j<img_cols-radius;j++){
                    bool isKeyPoint=true;
                    double center=MatB.at<double>(i,j);

                    maxCenter=max(maxCenter,fabs(center));
                    //1. remove low constrast point
                    if(fabs(center)<threshold){
                        isKeyPoint=false;
                        keyPointsMat.at<uchar>(i,j)=50;
                        continue;
                    }

                    //                    LOG_MESSAGE("extremum");
                    //2.1 max/min in 26 neighboorhood
                    bool max=true,min=true;

                    for(int a=-1;a<=1;a++){
                        for(int b=-1;b<=1;b++){
                            for(int c=-1;c<=1;c++){
                                if(!(a==0&&b==0&&c==0)){
                                    double neighboor3D=DOGOctaves[oi][sj+c].at<double>(i+a,b+j);
                                    //NOTE >= or > ?
                                    if(center>neighboor3D){
                                        min=false;
                                    }
                                    else if(center<neighboor3D){
                                        max=false;
                                    }
                                }
                            }
                        }
                    }
                    isKeyPoint=(max|min);
                    if(!isKeyPoint){
                        keyPointsMat.at<uchar>(i,j)=100;
                        continue;
                    }

                    keyPointsMat.at<uchar>(i,j)=150;

                    //                    LOG_MESSAGE("interp");
                    //2.2 max/min when interp ?
                    //3. remove edge points
                    KeyPoint kp;
                    int newsj=sj,newi=i,newj=j;
                    isKeyPoint=isInterpExtremun(DOGOctaves,oi,newsj,newi,newj,sigma,kp,keyPointsMat);
                    //                    LOG_MESSAGE("interp okay");
                    if(!isKeyPoint){
                        //                        keyPointsMat.at<uchar>(i,j)=200;
                        continue;
                    }

                    keyPointsMat.at<uchar>(i,j)=250;

                    //4. direction hog.
                    float scl_octv=kp.size*0.5f/(1<<oi);
//                    LOG_MESSAGE("[scl_octv,kp.size,oi,sj]=["<<scl_octv<<","<<kp.size<<","<<oi<<","<<sj<<"]");
                    vector<double> hist;
                    int radius=cvRound(3*scl_octv);
                    double sigma=1.5*scl_octv;
                    double maxhist=getOrientationHist(GaussOctaves,oi,newsj,newi,newj,radius,sigma,hist);

                    //5. generate sub feature points at sub peak
                    double histThreshold=maxhist*defaultHistPeakRatio;
                    int histSize=hist.size();


                    for(int u=0;u<histSize;u++){
                        int left=u>0 ? u-1:histSize-1;
                        int right=u<histSize-1 ? u+1:0;
                        if(hist[u]>hist[left]&&hist[u]>hist[right]&&hist[u]>=histThreshold){
                            double bin=u+0.5*(hist[left]-hist[right])/(hist[left]+hist[right]-2*hist[u]);
                            if(bin<0) bin=bin+histSize;
                            else if(bin>=histSize) bin=bin-histSize;

                            kp.angle=360-(float)((360.f/histSize)*bin);

                            if(std::abs(kp.angle-360.f)<FLT_EPSILON)
                                kp.angle=0.f;

                            LOG_MESSAGE("i,j,bin,angle="<<newi<<","<<newj<<bin<<","<<kp.angle);
                            keyPoints.push_back(kp);
                            keyPointsMat.at<uchar>(newi,newj)=255;
                        }
                    }

                    if(keyPointsMat.at<uchar>(newi,newj)==250){
                        LOG_MESSAGE("i,j,bin,angle="<<newi<<","<<newj<<","<<kp.angle);
                        LOG_MESSAGE("omax,mag_thr"<<maxhist<<","<<histThreshold);
                    }



                }
            }
            keyPointsOcataveMats.push_back(keyPointsMat);

//            double minVal,maxVal;
//            minMaxIdx(keyPointsMat,&minVal,&maxVal);
//            LOG_MESSAGE("minVal="<<minVal<<" maxVal="<<maxVal);
//            LOG_MESSAGE("maxCenter="<<maxCenter);
        }
        keyPointsOcataves.push_back(keyPointsOcataveMats);
    }

    int keyOctaveNum=keyPointsOcataves.size();
    for(int oi=0;oi<keyOctaveNum;oi++){
        int scaleNum=keyPointsOcataves[oi].size();
        for(int sj=0;sj<scaleNum;sj++){
            showOcatave(oi,sj,keyPointsOcataves[oi][sj],"keyPoints-");
        }
    }

    debugMatVector.swap(keyPointsOcataves);
}

void yzbx_sift::showOcatave(int i,int j,const Mat &mat,string addtionStr){
//    static showCount=0;
//    showCount++;
//    if(showCount>5) return;
//    if(i!=2||j>3) return;

    stringstream ss;
    ss<<i<<" - "<<j;
    string name=ss.str();
    imshow(addtionStr+"ocatave-"+name,mat);
}

bool yzbx_sift::isInterpExtremun(const vector<vector<Mat> > &DOGOctaves, int oi, int &sj, int &i, int &j, double sigma,KeyPoint &kp,Mat &debugMat){

    const uchar debug_outOfBorder=210;
    const uchar debug_convergence=220;
    const uchar debug_lowInterpContrast=230;
    const uchar debug_edgePoint=240;

    int octaveNum=DOGOctaves.size();
    CV_Assert(octaveNum>oi);
    int scaleNum=DOGOctaves[oi].size();
    CV_Assert(scaleNum>sj);
    int img_rows=DOGOctaves[oi][sj].rows;
    int img_cols=DOGOctaves[oi][sj].cols;
    int radius=defaultImageBorder;

    bool continueDebug=false;

    //step 1. compute interpMat which contain the position after interp.
    int interpTimes;
    //interpolated subpixel increment to j,i,sj
    Mat interpMat;
    double dx,dy,dz;
    for(interpTimes=0;interpTimes<defaultInterpTimes;interpTimes++){

        //step 1.1 computer the interp mat.
        //        LOG_MESSAGE("get interp mat");
        getInterpMat(DOGOctaves,oi,sj,i,j,interpMat);
        //        LOG_MESSAGE("okay");
        CV_Assert(!interpMat.empty());
        CV_Assert(interpMat.cols==1);
        CV_Assert(interpMat.rows==3);
        CV_Assert(interpMat.type()==CV_64FC1);

        dx=interpMat.at<double>(0,0);
        dy=interpMat.at<double>(1,0);
        dz=interpMat.at<double>(2,0);

//        if((oi==3&&i==yzbx_debug_row&&j==yzbx_debug_col)||continueDebug){
//            continueDebug=true;
//            LOG_MESSAGE("[oi,sj,i,j]=["<<oi<<","<<sj<<","<<i<<","<<j<<"]");
//            LOG_MESSAGE("[dx,dy,dz,interpTimes]=["<<dx<<","<<dy<<","<<dz<<","<<interpTimes<<"]");
//        }

        //step 1.2 check convergence
        if(fabs(dx)<0.5&&fabs(dy)<0.5&&fabs(dz)<0.5){
            break;
        }

        //step 1.3 iterate interp
        i+=round(dy);
        j+=round(dx);
        sj+=round(dz);


        //step 1.4 check valid interp whose location in valid area.
        //for gauss octaves, scaleNum=5
        //for DOG scaleNum=4, the index is 0,1,2,3
        //remove the first and last, the valid index should be 1 or 2.

        if(sj<=0||sj>=scaleNum-1||i<radius||j<radius||i>img_rows-radius||j>img_cols-radius){
            if(!debugMat.empty()){
                if(i<debugMat.rows&&j<debugMat.cols&&i>=0&&j>=0){
                    debugMat.at<uchar>(i,j)=debug_outOfBorder;
                }

//                debugMat.at<uchar>(i,j)=debug_outOfBorder;
            }
            return false;
        }

    }

    /* ensure convergence of interpolation */
    if(interpTimes>=defaultInterpTimes){
        if(!debugMat.empty()){
            if(i<debugMat.rows&&j<debugMat.cols&&i>=0&&j>=0)
                debugMat.at<uchar>(i,j)=debug_convergence;
//            debugMat.at<uchar>(i,j)=debug_convergence;
        }
        return false;
    }

    //TODO check the contrast after interp, sj,i,j may changed.
    Mat gradientMat;
    //    LOG_MESSAGE("get gradient Mat");
    getGradientMat(DOGOctaves,oi,sj,i,j,gradientMat);
    //    LOG_MESSAGE("okay");

    //the pixel value after interp
    double threshold=defaultContrastThreshold/(defaultScaleNum-3);
    double pixelValueAfterInterp=DOGOctaves[oi][sj].at<double>(i,j);
    for(int a=0;a<3;a++){
        pixelValueAfterInterp+=0.5*gradientMat.at<double>(a,0)*interpMat.at<double>(a,0);
    }
    if(fabs(pixelValueAfterInterp)*2<threshold){
        if(!debugMat.empty()){
            debugMat.at<uchar>(i,j)=debug_lowInterpContrast;
        }
        return false;
    }

    //3. remove edge points
    //CONFIG, edge in MatB or the first Mat in octaves ?
    int newi,newj,newsj;
    newsj=sj;
    newi=i;
    newj=j;
    const Mat &edgeMat=DOGOctaves[oi][newsj];
    double newcenter=edgeMat.at<double>(newi,newj);
    double dyy=edgeMat.at<double>(newi,newj+1)+edgeMat.at<double>(newi,newj-1)-2*newcenter;
    double dxx=edgeMat.at<double>(newi+1,newj)+edgeMat.at<double>(newi-1,newj)-2*newcenter;
    double dxy=((edgeMat.at<double>(newi+1,newj+1)-edgeMat.at<double>(newi+1,newj-1))-(edgeMat.at<double>(newi-1,newj+1)-edgeMat.at<double>(newi-1,newj-1)))/4;
    double tr=dxx+dyy;
    double det=dxx*dyy-dxy*dxy;
    if(det<=0||tr*tr/det>defaultEdgeThresholdFinal){

        if(!debugMat.empty()){
            debugMat.at<uchar>(i,j)=debug_edgePoint;
        }
        return false;
    }

    //    LOG_MESSAGE("set kp");
    int nOctaveLayers=defaultScaleNum-3;
    kp.octave=oi+(newsj<<8)+(cvRound((dz+0.5)*255)<<16);
    kp.pt.x=(newj+dx)*(1<<oi);
    kp.pt.y=(newi+dy)*(1<<oi);
    kp.size=sigma*pow(2,(newsj+dz)/nOctaveLayers)*(1<<oi)*2;
    kp.response=fabs(pixelValueAfterInterp);

//    static int count=0;
//    if(count<20){
//        LOG_MESSAGE("[octave,pt,size,response,sigma,nOctaveLayers]"<<kp.octave<<","<<kp.pt<<","<<kp.size<<","<<kp.response<<","<<sigma<<","<<nOctaveLayers);
//        count++;
//    }
    return true;
}

void yzbx_sift::getInterpMat(const vector<vector<Mat>> &DOGOctaves,int oi,int sj,int i,int j,Mat &interpMat){
    Mat gradientMat,hessianMat;
    //    LOG_MESSAGE("get gradient mat");
    getGradientMat(DOGOctaves,oi,sj,i,j,gradientMat);
    //    LOG_MESSAGE("get hessian mat");
    getHessianMat(DOGOctaves,oi,sj,i,j,hessianMat);
    //    LOG_MESSAGE("invert the mat");

    //    Mat invertMat;
    //    cv::invert(hessianMat,invertMat,CV_SVD);
    //    cout<<invertMat.size()<<" "<<gradientMat.size()<<endl;
    //    interpMat=-invertMat*gradientMat;
    //    bool flag=cv::solve(hessianMat,gradientMat.t(),interpMat,DECOMP_LU);

    CV_Assert(gradientMat.type()==CV_64FC1);
    CV_Assert(hessianMat.type()==CV_64FC1);



    Vec3d gradientVec;
    for(int i=0;i<3;i++){
        gradientVec[i]=gradientMat.at<double>(i,0);
    }

    Matx33d H;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            H(i,j)=hessianMat.at<double>(i,j);
            //            cout<<"hessianMat="<<hessianMat.at<double>(i,j);
            //            cout<<"H="<<H(i,j)<<endl;
        }
    }

    Vec3d X=H.solve(gradientVec,DECOMP_LU);
    interpMat.release();
    interpMat.create(Size(1,3),CV_64FC1);
    CV_Assert(interpMat.rows==3);
    CV_Assert(interpMat.cols==1);
    for(int i=0;i<3;i++){
        interpMat.at<double>(i,0)=-X[i];
    }

}

void yzbx_sift::getGradientMat(const vector<vector<Mat>> &DOGOctaves,int oi,int sj,int i,int j,Mat &gradientMat){
    CV_Assert(!DOGOctaves[oi][sj].empty());
    int img_rows=DOGOctaves[oi][sj].rows;
    int img_cols=DOGOctaves[oi][sj].cols;
    //NOTE
//    CV_Assert(i>defaultImageBorder&&i<img_rows-defaultImageBorder);
//    CV_Assert(j>defaultImageBorder&&j<img_cols-defaultImageBorder);
    CV_Assert(i>0&&i<img_rows-1);
    CV_Assert(j>0&&j<img_cols-1);

    double dy=(DOGOctaves[oi][sj].at<double>(i+1,j)-DOGOctaves[oi][sj].at<double>(i-1,j))/2;
    double dx=(DOGOctaves[oi][sj].at<double>(i,j+1)-DOGOctaves[oi][sj].at<double>(i,j-1))/2;
    double dz=(DOGOctaves[oi][sj+1].at<double>(i,j)-DOGOctaves[oi][sj-1].at<double>(i,j))/2;

    gradientMat.release();
    gradientMat.create(Size(1,3),CV_64FC1);
    CV_Assert(gradientMat.rows==3);
    CV_Assert(gradientMat.cols==1);
    gradientMat.at<double>(0,0)=dx;
    gradientMat.at<double>(1,0)=dy;
    gradientMat.at<double>(2,0)=dz;
}

void yzbx_sift::getHessianMat(const vector<vector<Mat>> &DOGOctaves,int oi,int sj,int i,int j,Mat &hessianMat)
{
    CV_Assert(!DOGOctaves[oi][sj].empty());
    int img_rows=DOGOctaves[oi][sj].rows;
    int img_cols=DOGOctaves[oi][sj].cols;
    //NOTE
//    CV_Assert(i>defaultImageBorder&&i<img_rows-defaultImageBorder);
//    CV_Assert(j>defaultImageBorder&&j<img_cols-defaultImageBorder);
    CV_Assert(i>0&&i<img_rows-1);
    CV_Assert(j>0&&j<img_cols-1);

    double center=DOGOctaves[oi][sj].at<double>(i,j);
    int dsj,di,dj;
    di=1;dj=0;dsj=0;
    double dyy=(DOGOctaves[oi][sj+dsj].at<double>(i+di,j+dj)+DOGOctaves[oi][sj+dsj].at<double>(i-di,j-dj))-2*center;
    di=0;dj=1;dsj=0;
    double dxx=(DOGOctaves[oi][sj+dsj].at<double>(i+di,j+dj)+DOGOctaves[oi][sj+dsj].at<double>(i-di,j-dj))-2*center;
    di=0;dj=0;dsj=1;
    double dzz=(DOGOctaves[oi][sj+dsj].at<double>(i+di,j+dj)+DOGOctaves[oi][sj-dsj].at<double>(i-di,j-dj))-2*center;

    di=1;dj=1;dsj=0;
    double dxy=((DOGOctaves[oi][sj+dsj].at<double>(i+di,j+dj)-DOGOctaves[oi][sj+dsj].at<double>(i+di,j-dj))-(DOGOctaves[oi][sj-dsj].at<double>(i-di,j+dj)-DOGOctaves[oi][sj-dsj].at<double>(i-di,j-dj)))/4;
    di=1;dj=0;dsj=1;
    double dyz=((DOGOctaves[oi][sj+dsj].at<double>(i+di,j+dj)-DOGOctaves[oi][sj+dsj].at<double>(i-di,j-dj))-(DOGOctaves[oi][sj-dsj].at<double>(i+di,j+dj)-DOGOctaves[oi][sj-dsj].at<double>(i-di,j-dj)))/4;
    di=0;dj=1;dsj=1;
    double dxz=((DOGOctaves[oi][sj+dsj].at<double>(i+di,j+dj)-DOGOctaves[oi][sj+dsj].at<double>(i-di,j-dj))-(DOGOctaves[oi][sj-dsj].at<double>(i+di,j+dj)-DOGOctaves[oi][sj-dsj].at<double>(i-di,j-dj)))/4;

    hessianMat.release();
    hessianMat.create(Size(3,3),CV_64FC1);
    hessianMat.at<double>(0,0)=dxx;
    hessianMat.at<double>(0,1)=dxy;
    hessianMat.at<double>(0,2)=dxz;
    hessianMat.at<double>(1,0)=dxy;
    hessianMat.at<double>(1,1)=dyy;
    hessianMat.at<double>(1,2)=dyz;
    hessianMat.at<double>(2,0)=dxz;
    hessianMat.at<double>(2,1)=dyz;
    hessianMat.at<double>(2,2)=dzz;
}

double yzbx_sift::getOrientationHist(const vector<vector<Mat>> &GaussOctaves,int oi,int sj,int i,int j,int radius,double sigma,vector<double> &hist){
    static int count=0;
    if(count<3)
    {
        LOG_MESSAGE("[count,i,j,radius,sigma]=["<<count<<","<<i<<","<<j<<","<<radius<<","<<sigma<<"]");
//        count++;
    }
//    static int count=0;
    int n=36;
    int len=(radius*2+1)*(radius*2+1);
    double expf_scale=-1.f/(2.f*sigma*sigma);

    CV_Assert(!GaussOctaves[oi][sj].empty());
    int img_rows=GaussOctaves[oi][sj].rows;
    int img_cols=GaussOctaves[oi][sj].cols;
    vector<double> dxs,dys,oris,mags,weights;
    vector<double> tmpHists;
    for(int i=0;i<len;i++){
        dxs.push_back(0);
        dys.push_back(0);
        oris.push_back(0);
        mags.push_back(0);
        weights.push_back(0);
    }

    for(int i=0;i<n;i++){
        tmpHists.push_back(0);
    }

    int idx=0;
    for(int di=-radius;di<=radius;di++){
        if(i+di<=0||i+di>=img_rows-1) continue;

        for(int dj=-radius;dj<=radius;dj++){
            if(dj+j<=0||j+dj>=img_cols-1) continue;

            double dx=GaussOctaves[oi][sj].at<double>(i+di,j+dj+1)-GaussOctaves[oi][sj].at<double>(i+di,j+dj-1);
            //NOTE !!!
//            double dy=GaussOctaves[oi][sj].at<double>(i+di+1,j+dj)-GaussOctaves[oi][sj].at<double>(i+di-1,j+dj);
            double dy=GaussOctaves[oi][sj].at<double>(i+di-1,j+dj)-GaussOctaves[oi][sj].at<double>(i+di+1,j+dj);

            dxs[idx]=dx;
            dys[idx]=dy;
            weights[idx]=(di*di+dj*dj)*expf_scale;
            idx++;
            CV_Assert(idx<=len);
        }
    }

    len=idx;
    for(int i=0;i<len;i++){
        weights[i]=exp(weights[i]);
        oris[i]=fastAtan2(dys[i],dxs[i]);
        mags[i]=sqrt(dxs[i]*dxs[i]+dys[i]*dys[i]);
    }

    for(int k=0;k<len;k++){
        int bin=cvRound(n*oris[k]/360.0);
//        if(bin<0||bin>n){

//            for(int i=0;i<len;i++){
//                cout<<"oris["<<i<<"]="<<oris[i]<<endl;
//            }
//            LOG_MESSAGE("bin="<<bin);
//        }
        CV_Assert(bin>=0&&bin<=n);
        if(bin==n) bin=0;
        tmpHists[bin]+=weights[k]*mags[k];
    }
    CV_Assert(tmpHists.size()==n);

    hist.clear();

    for(int k=0;k<n;k++){
        int a=(k-2+n)%n;
        int b=(k-1+n)%n;
        int c=k;
        int d=(k+1)%n;
        int e=(k+2)%n;

        double val=(tmpHists[a]+tmpHists[e])*(1/16.0)+(tmpHists[d]+tmpHists[b])*(4/16.0)+ \
                tmpHists[c]*(6/16.0);
        hist.push_back(val);
    }
    CV_Assert(hist.size()==n);

//    if(count==0){
////        imshow("count ?",)
//        cout<<"len="<<len;

//        cout<<"dxs=";
//        for(int i=0;i<len;i++){
//            cout<<dxs[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"dys=";
//        for(int i=0;i<len;i++){
//            cout<<dys[i]<<" ";
//        }
//        cout<<endl;


//        cout<<"Ori=";
//        for(int i=0;i<len;i++){
//            cout<<oris[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"Mag=";
//        for(int i=0;i<len;i++){
//            cout<<mags[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"W=";
//        for(int i=0;i<len;i++){
//            cout<<weights[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"tmpHist=";
//        for(int i=0;i<n;i++){
//            cout<<tmpHists[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"hist=";
//        for(int i=0;i<n;i++){
//            cout<<hist[i]<<" ";
//        }
//        cout<<endl;
//    }

    count++;

    double maxval=0;
    for(int k=0;k<n;k++){
        maxval=max(maxval,hist[k]);
    }

    return maxval;
}

void yzbx_sift::computerDescriptors(const Mat &bgrOrGrayImage,const vector<KeyPoint> &keyPoints,Mat &descriptorMat){
    int nOctaveLayers=3;
    float sigma=1.6;
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image =bgrOrGrayImage;

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( true )
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keyPoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            unpackOctave(keyPoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }

        firstOctave = std::min(firstOctave, 0);
        CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log( (double)std::min( base.cols, base.rows ) ) / log(2.) - 2) - firstOctave;

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    vector<vector<Mat>> GaussOctaves,DOGOctaves;
    base.convertTo(base,CV_8UC1);
    //    cvtColor(base,base,CV_GRAY2BGR);
    generateGaussOctaves(base,GaussOctaves,nOctaves,nOctaveLayers+3);
    generateDOGOctaves(GaussOctaves,DOGOctaves);



    Mat descriptorMat64F;
    descriptorMat.release();
    int img_rows=keyPoints.size();
    int img_cols=defaultDescriptorWidth*defaultDescriptorWidth*defaultDescriptorHistBinNum;

    descriptorMat.create(img_rows,img_cols,CV_8UC1);
    descriptorMat64F.create(img_rows,img_cols,CV_64FC1);

    for( size_t i = 0; i < keyPoints.size(); i++ )
    {
        KeyPoint kpt = keyPoints[i];
        int octave, layer;
        float scale;
        unpackOctave(kpt, octave, layer, scale);
        CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);

        float size=kpt.size*scale;
        Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
        int r=cvRound(ptf.y);
        int c=cvRound(ptf.x);
        const Mat& img = GaussOctaves[octave-firstOctave][layer];
        CV_Assert(!img.empty());

        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;

        int d=defaultDescriptorWidth;
        vector<vector<vector<double>>> hist;
        getDescriptorsHist(img, r,c, d,angle, size*0.5f,hist);
        if(debugNum==1){
            int count=0;
            for(int i=0;i<d;i++){
                for(int j=0;j<d;j++){
                    for(int k=0;k<8&&count<20;k++){
                        LOG_MESSAGE("hist["<<i<<"]["<<j<<"]["<<k<<"]="<<hist[i][j][k]);
                    }
                }
            }
        }
        setDescriptorMat(descriptorMat64F,hist,i);
    }

    descriptorMat64F.convertTo(descriptorMat,CV_8UC1);
}

void yzbx_sift::setDescriptorMat(Mat &descriptorMat,vector<vector<vector<double>>> &hist,int rowNum){
    int d=hist.size();
    CV_Assert(hist[0].size()==d);
    int n=hist[0][0].size();
    CV_Assert(descriptorMat.type()==CV_64FC1);
    double *dst=descriptorMat.ptr<double>(rowNum);

    // finalize histogram, since the orientation histograms are circular
    for(int i = 0; i < d; i++ )
        for(int j = 0; j < d; j++ )
        {
            for(int k = 0; k < n; k++ ){
                //                    dst[(i*d + j)*n + k] = hist[idx+k];
                dst[(i*d+j)*n+k]=hist[i][j][k];
            }

        }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    int len = d*d*n;
    for(int k = 0; k < len; k++ )
        nrm2 += dst[k]*dst[k];
    double SIFT_DESCR_MAG_THR=0.2;
    double thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    for(int i = 0, nrm2 = 0; i < len; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }

    double SIFT_INT_DESCR_FCTR=512.0;
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

    for(int k = 0; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
}

void yzbx_sift::getDescriptorsHist(const Mat&img, int r, int c, int descriptorWidth, double mainOritation, double size, vector<vector<vector<double> > > &hist){
    int d=descriptorWidth;
    int n=defaultDescriptorHistBinNum;
    double scl=size;
    double ori=mainOritation*CV_PI/180.0;
    double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
            grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
    int radius, i, j;


    for( i = 0; i < d; i++ )
    {
        vector<vector<double>> hist2d;
        for(int j=0;j<d;j++){
            vector<double> hist1d(n);
            for(int k=0;k<n;k++){
                hist1d[k]=0;
            }

            hist2d.push_back(hist1d);
        }

        hist.push_back(hist2d);
    }

    CV_Assert(hist.size()==d);
    CV_Assert(hist[0].size()==d);
    CV_Assert(hist[0][0].size()==n);

    cos_t = cos( ori );
    sin_t = sin( ori );
    bins_per_rad = n / PI2;
    exp_denom = d * d * 0.5;
    double SIFT_DESCR_SCL_FCTR=3.0;
    hist_width = SIFT_DESCR_SCL_FCTR * scl;
    radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;

    if(debugNum<3){
        debugNum++;
        LOG_MESSAGE("[r,c,hist_width,radius,d,scl]=["<<r<<","<<c<<","<<hist_width<<","<<radius<<","<<d<<","<<scl<<"]");
    }
    for( i = -radius; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            /*
          Calculate sample's histogram array coords rotated relative to ori.
          Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
          r_rot = 1.5) have full weight placed in row 1 after interpolation.
        */
            c_rot = ( j * cos_t - i * sin_t ) / hist_width;
            r_rot = ( j * sin_t + i * cos_t ) / hist_width;
            rbin = r_rot + d / 2 - 0.5;
            cbin = c_rot + d / 2 - 0.5;

            if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
                if( getDescriptorGradientInfo( img, r + i, c + j, grad_mag, grad_ori ))
                {
                    grad_ori -= ori;
                    while( grad_ori < 0.0 )
                        grad_ori += PI2;
                    while( grad_ori >= PI2 )
                        grad_ori -= PI2;

                    obin = grad_ori * bins_per_rad;
                    w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
                    interpDescriptorHist( hist, rbin, cbin, obin, grad_mag * w, d, n );
                }
        }
}

bool yzbx_sift::getDescriptorGradientInfo(const Mat&img,int r,int c,double &mag,double& ori){
    double dx, dy;
    static int featureNum=0;

    if( r > 0  &&  r < img.rows - 1  &&  c > 0  &&  c < img.cols - 1 )
    {
        dx = img.at<double>(r,c+1)-img.at<double>(r,c-1);//pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
        dy = img.at<double>(r-1,c)-img.at<double>(r+1,c);//pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
        mag = sqrt( dx*dx + dy*dy );
        ori = atan2( dy, dx );
        if(featureNum<20){
            featureNum++;
            LOG_MESSAGE("[dx,dy]=["<<dx<<","<<dy<<"]")
        }

        return true;
    }
    else
        return false;
}

void yzbx_sift::interpDescriptorHist(vector<vector<vector<double>>> &hist,double rbin,double cbin,double obin,double mag,int descriptorWidth,int descriptorHistBinNum){
    double d_r, d_c, d_o, v_r, v_c, v_o;
    int d=descriptorWidth;
    int n=descriptorHistBinNum;

    int r0, c0, o0, rb, cb, ob, r, c, o;
    static int featureNum=0;

    r0 = cvFloor( rbin );
    c0 = cvFloor( cbin );
    o0 = cvFloor( obin );
    d_r = rbin - r0;
    d_c = cbin - c0;
    d_o = obin - o0;

    /*
        The entry is distributed into up to 8 bins.  Each entry into a bin
        is multiplied by a weight of 1 - d for each dimension, where d is the
        distance from the center value of the bin measured in bin units.
      */
    for( r = 0; r <= 1; r++ )
    {
        rb = r0 + r;
        if( rb >= 0  &&  rb < d )
        {
            v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
            vector<vector<double>> &row = hist[rb];
            for( c = 0; c <= 1; c++ )
            {
                cb = c0 + c;
                if( cb >= 0  &&  cb < d )
                {
                    v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
                    vector<double> &h = row[cb];
                    for( o = 0; o <= 1; o++ )
                    {
                        ob = ( o0 + o ) % n;
                        v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
                        h[ob] += v_o;
                    }
                }
            }
        }
    }

    if(featureNum<20){
        LOG_MESSAGE("[Rbin,CBin]="<<rbin<<","<<cbin<<"]");
        featureNum++;
    }
}

void unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}
void yzbx_sift::matchDescriptors(const Mat &descriptors1,const Mat &descriptors2,vector<DMatch> matches){

}

static void calcSIFTDescriptor( const Mat& img, Point2f ptf, float ori, float scl,
                                int d, int n, float* dst )
{
    static int featureNum=0;
    static int debugNum=0;

    Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    //    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    float hist_width=3.0*scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt((double) img.cols*img.cols + img.rows*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;

    AutoBuffer<float> buf(len*6 + histlen);
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

    for( i = 0; i < d+2; i++ )
    {
        for( j = 0; j < d+2; j++ )
            for( k = 0; k < n+2; k++ )
                hist[(i*(d+2) + j)*(n+2) + k] = 0.;
    }


    for( i = -radius, k = 0; i <= radius; i++ )
        for( j = -radius; j <= radius; j++ )
        {
            // Calculate sample's histogram array coords rotated relative to ori.
            // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                    r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
            {
                float dx = (float)(img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }

    featureNum++;
    if(featureNum==1){
        LOG_MESSAGE("[r,c,radius,hist_width]=["<<pt.y<<","<<pt.x<<","<<radius<<","<<hist_width<<"]");
        for(i=0;i<k&&i<20;i++){
            LOG_MESSAGE("[i,X,Y,W,RBin,CBin]=["<<i<<","<<X[i]<<","<<Y[i]<<","<<W[i]<<","<<RBin[i]<<","<<CBin[i]<<"]")
        }
    }

    len = k;
    fastAtan2(Y, X, Ori, len, true);
    magnitude(X, Y, Mag, len);
    exp(W, W, len);

    for( k = 0; k < len; k++ )
    {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor( rbin );
        int c0 = cvFloor( cbin );
        int o0 = cvFloor( obin );
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if( o0 < 0 )
            o0 += n;
        if( o0 >= n )
            o0 -= n;

        // histogram update using tri-linear interpolation
        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // finalize histogram, since the orientation histograms are circular
    for( i = 0; i < d; i++ )
        for( j = 0; j < d; j++ )
        {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for( k = 0; k < n; k++ )
                dst[(i*d + j)*n + k] = hist[idx+k];
        }

    if(featureNum==1){
        int count=0;
        for(i=0;i<d;i++){
            for(j=0;j<d;j++){
                int idx = ((i+1)*(d+2) + (j+1))*(n+2);

                for(k=0;k<n&&count<20;k++){
                    count++;
                    //                    LOG_MESSAGE("hist[i][j][k]="<<hist[idx+k])
                    LOG_MESSAGE("hist["<<i<<"]["<<j<<"]["<<k<<"]="<<hist[idx+k]);
                }
            }
        }
    }
    // copy histogram to the descriptor,
    // apply hysteresis thresholding
    // and scale the result, so that it can be easily converted
    // to byte array
    float nrm2 = 0;
    len = d*d*n;
    for( k = 0; k < len; k++ )
        nrm2 += dst[k]*dst[k];
    //    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    float thr = std::sqrt(nrm2)*0.2;
    for( i = 0, nrm2 = 0; i < k; i++ )
    {
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    //    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);
    nrm2 = 512.0/std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
    for( k = 0; k < len; k++ )
    {
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
#else
    float nrm1 = 0;
    for( k = 0; k < len; k++ )
    {
        dst[k] *= nrm2;
        nrm1 += dst[k];
    }
    nrm1 = 1.f/std::max(nrm1, FLT_EPSILON);
    for( k = 0; k < len; k++ )
    {
        dst[k] = std::sqrt(dst[k] * nrm1);//saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
    }
#endif
}

static void calcDescriptors(const vector<Mat>& gpyr, const vector<KeyPoint>& keypoints,
                            Mat& descriptors, int nOctaveLayers, int firstOctave )
{
    //    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    int d=4,n=8;

    for( size_t i = 0; i < keypoints.size(); i++ )
    {
        KeyPoint kpt = keypoints[i];
        int octave, layer;
        float scale;
        unpackOctave(kpt, octave, layer, scale);
        CV_Assert(octave >= firstOctave && layer <= nOctaveLayers+2);
        float size=kpt.size*scale;
        Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
        const Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;
        calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    }
}


void yzbx_sift::debug(InputArray _image,
                      vector<KeyPoint>& keypoints,
                      OutputArray _descriptors,
                      bool useProvidedKeypoints)
{
//    LOG_MESSAGE("descriptors debug");
    int nOctaveLayers=3;
    float sigma=1.6;
    int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;
    Mat image = _image.getMat();

    if( image.empty() || image.depth() != CV_8U )
        CV_Error( CV_StsBadArg, "image is empty or has incorrect depth (!=CV_8U)" );

    if( useProvidedKeypoints )
    {
        firstOctave = 0;
        int maxOctave = INT_MIN;
        for( size_t i = 0; i < keypoints.size(); i++ )
        {
            int octave, layer;
            float scale;
            unpackOctave(keypoints[i], octave, layer, scale);
            firstOctave = std::min(firstOctave, octave);
            maxOctave = std::max(maxOctave, octave);
            actualNLayers = std::max(actualNLayers, layer-2);
        }

        firstOctave = std::min(firstOctave, 0);
        CV_Assert( firstOctave >= -1 && actualNLayers <= nOctaveLayers );
        actualNOctaves = maxOctave - firstOctave + 1;
    }

    Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
    vector<Mat> gpyr, dogpyr;
    int nOctaves = actualNOctaves > 0 ? actualNOctaves : cvRound(log( (double)std::min( base.cols, base.rows ) ) / log(2.) - 2) - firstOctave;

    //double t, tf = getTickFrequency();
    //t = (double)getTickCount();
    vector<vector<Mat>> GaussOctaves,DOGOctaves;
    base.convertTo(base,CV_8UC1);
    //    cvtColor(base,base,CV_GRAY2BGR);
    generateGaussOctaves(base,GaussOctaves,nOctaves,nOctaveLayers+3);
    generateDOGOctaves(GaussOctaves,DOGOctaves);

    for(int i=0;i<nOctaves;i++){
        for(int j=0;j<nOctaveLayers+3;j++){
            gpyr.push_back(GaussOctaves[i][j]);
        }

        for(int j=0;j<nOctaveLayers+2;j++){
            dogpyr.push_back(DOGOctaves[i][j]);
        }
    }
    //    buildGaussianPyramid(base, gpyr, nOctaves);
    //    buildDoGPyramid(gpyr, dogpyr);

    //t = (double)getTickCount() - t;
    //printf("pyramid construction time: %g\n", t*1000./tf);

    if( !useProvidedKeypoints )
    {
        //t = (double)getTickCount();
        debug(gpyr, dogpyr, keypoints);
        int size=keypoints.size();
//        KeyPointsFilter::removeDuplicated( keypoints );
        LOG_MESSAGE("keypoints.size="<<keypoints.size());

//        if( nfeatures > 0 )
//            KeyPointsFilter::retainBest(keypoints, nfeatures);
        //t = (double)getTickCount() - t;
        //printf("keypoint detection time: %g\n", t*1000./tf);

        if( firstOctave < 0 )
            for( size_t i = 0; i < keypoints.size(); i++ )
            {
                KeyPoint& kpt = keypoints[i];
                float scale = 1.f/(float)(1 << -firstOctave);
                kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
                kpt.pt *= scale;
                kpt.size *= scale;
            }

    }
    else
    {
        // filter keypoints by mask
        //KeyPointsFilter::runByPixelsMask( keypoints, mask );
    }

    if( _descriptors.needed() &&useProvidedKeypoints)
    {
//        LOG_MESSAGE("needed");
        //t = (double)getTickCount();
        //        int dsize = descriptorSize();
        int dsize=4*4*8;
        _descriptors.create((int)keypoints.size(), dsize, CV_32F);
        Mat descriptors = _descriptors.getMat();

        calcDescriptors(gpyr, keypoints, descriptors, nOctaveLayers, firstOctave);
        //t = (double)getTickCount() - t;
        //printf("descriptor extraction time: %g\n", t*1000./tf);
    }
}

void yzbx_sift::debug( const vector<Mat>& gauss_pyr, const vector<Mat>& dog_pyr,
                       vector<KeyPoint>& keypoints )
{
//    LOG_MESSAGE("key point debug");
    int nOctaveLayers=3;
    int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers + 3);
//    int threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
//    int threshold = cvFloor(0.5 * 0.04 / nOctaveLayers * 1.0);
    double threshold=0.04*0.5/nOctaveLayers;
//    double threshold=0.04;
//    const int n = SIFT_ORI_HIST_BINS;
    const int n=36;
    float hist[n];
    KeyPoint kpt;
    const int SIFT_IMG_BORDER=5;
    double edgeThreshold=10.0;
    double sigma=1.6;
    const int SIFT_ORI_RADIUS=3;
    const double SIFT_ORI_SIG_FCTR=1.5;
    const double SIFT_ORI_PEAK_RATIO=0.8;
    const double contrastThreshold=0.04;

    keypoints.clear();
    double maxVal=0;
//    LOG_MESSAGE("clear keypoints");

    vector<vector<Mat>> keyPointsMatOctaveVector;
    LOG_MESSAGE("nOctaves="<<nOctaves);
    for( int o = 0; o < nOctaves; o++ ){
        vector<Mat> keyPointsMatLayerVector;
        for( int i = 1; i <= nOctaveLayers; i++ )
        {


            int idx = o*(nOctaveLayers+2)+i;
            const Mat& img = dog_pyr[idx];
            const Mat& prev = dog_pyr[idx-1];
            const Mat& next = dog_pyr[idx+1];
            CV_Assert(!img.empty());
            CV_Assert(!prev.empty());
            CV_Assert(!next.empty());
            CV_Assert(img.type()==CV_64FC1);

            int step = (int)img.step1();
            int rows = img.rows, cols = img.cols;
            Mat keyPointsMat=Mat::zeros(rows,cols,CV_8UC1);

//            LOG_MESSAGE("big constrast and local extreama");
            for( int r = SIFT_IMG_BORDER; r < rows-SIFT_IMG_BORDER; r++)
            {
                const sift_wt* currptr = img.ptr<sift_wt>(r);
                const sift_wt* prevptr = prev.ptr<sift_wt>(r);
                const sift_wt* nextptr = next.ptr<sift_wt>(r);

                for( int c = SIFT_IMG_BORDER; c < cols-SIFT_IMG_BORDER; c++)
                {
                    sift_wt val = currptr[c];

                    maxVal=max(fabs(val),maxVal);
                    // find local extrema with pixel accuracy
                    if( fabs(val) > threshold &&
                            ((val > 0 && val >= currptr[c-1] && val >= currptr[c+1] &&
                              val >= currptr[c-step-1] && val >= currptr[c-step] && val >= currptr[c-step+1] &&
                              val >= currptr[c+step-1] && val >= currptr[c+step] && val >= currptr[c+step+1] &&
                              val >= nextptr[c] && val >= nextptr[c-1] && val >= nextptr[c+1] &&
                              val >= nextptr[c-step-1] && val >= nextptr[c-step] && val >= nextptr[c-step+1] &&
                              val >= nextptr[c+step-1] && val >= nextptr[c+step] && val >= nextptr[c+step+1] &&
                              val >= prevptr[c] && val >= prevptr[c-1] && val >= prevptr[c+1] &&
                              val >= prevptr[c-step-1] && val >= prevptr[c-step] && val >= prevptr[c-step+1] &&
                              val >= prevptr[c+step-1] && val >= prevptr[c+step] && val >= prevptr[c+step+1]) ||
                             (val < 0 && val <= currptr[c-1] && val <= currptr[c+1] &&
                              val <= currptr[c-step-1] && val <= currptr[c-step] && val <= currptr[c-step+1] &&
                              val <= currptr[c+step-1] && val <= currptr[c+step] && val <= currptr[c+step+1] &&
                              val <= nextptr[c] && val <= nextptr[c-1] && val <= nextptr[c+1] &&
                              val <= nextptr[c-step-1] && val <= nextptr[c-step] && val <= nextptr[c-step+1] &&
                              val <= nextptr[c+step-1] && val <= nextptr[c+step] && val <= nextptr[c+step+1] &&
                              val <= prevptr[c] && val <= prevptr[c-1] && val <= prevptr[c+1] &&
                              val <= prevptr[c-step-1] && val <= prevptr[c-step] && val <= prevptr[c-step+1] &&
                              val <= prevptr[c+step-1] && val <= prevptr[c+step] && val <= prevptr[c+step+1])))
                    {
//                        LOG_MESSAGE("adjust local extrema");
                        keyPointsMat.at<uchar>(r,c)=150;
                        int r1 = r, c1 = c, layer = i;
                        if( !adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                                nOctaveLayers, (float)contrastThreshold,
                                                (float)edgeThreshold, (float)sigma,keyPointsMat))
                            continue;

                        keyPointsMat.at<uchar>(r,c)=250;
                        float scl_octv = kpt.size*0.5f/(1 << o);
//LOG_MESSAGE("[scl_octv,kp.size,oi,sj]=["<<scl_octv<<","<<kpt.size<<","<<o<<","<<layer<<"]");
                        float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                Point(c1, r1),
                                cvRound(SIFT_ORI_RADIUS * scl_octv),
                                SIFT_ORI_SIG_FCTR * scl_octv,
                                hist, n);

                        float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);


                        for( int j = 0; j < n; j++ )
                        {
                            int l = j > 0 ? j - 1 : n - 1;
                            int r2 = j < n-1 ? j + 1 : 0;

                            if( hist[j] > hist[l]  &&  hist[j] > hist[r2]  &&  hist[j] >= mag_thr )
                            {
                                float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                                kpt.angle = 360.f - (float)((360.f/n) * bin);
                                if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                    kpt.angle = 0.f;

                                LOG_MESSAGE("i,j,bin,angle="<<r1<<","<<c1<<","<<bin<<","<<kpt.angle);
                                keypoints.push_back(kpt);
                                keyPointsMat.at<uchar>(r1,c1)=255;
                            }
                        }

                        if(keyPointsMat.at<uchar>(r1,c1)==250){
                            LOG_MESSAGE("i,j,bin,angle="<<r1<<","<<c1<<","<<","<<kpt.angle);
                            LOG_MESSAGE("omax,mag_thr"<<omax<<","<<mag_thr);
                        }
                    }
                    else{
                        if(fabs(val) > threshold)
                            keyPointsMat.at<uchar>(r,c)=100;
                        else
                            keyPointsMat.at<uchar>(r,c)=50;
                    }
                }
            }
            keyPointsMatLayerVector.push_back(keyPointsMat);
        }
        keyPointsMatOctaveVector.push_back(keyPointsMatLayerVector);
    }

//    LOG_MESSAGE("show keypoint Mat");
    int keyOctaveNum=keyPointsMatOctaveVector.size();
    for(int oi=0;oi<keyOctaveNum;oi++){
        int scaleNum=keyPointsMatOctaveVector[oi].size();
        for(int sj=0;sj<scaleNum;sj++){
            showOcatave(oi,sj,keyPointsMatOctaveVector[oi][sj],"cv-keyPoints-");
        }
    }

    debugMatVector.swap(keyPointsMatOctaveVector);
//    debugMatVector=keyPointsMatOctaveVector;
//    LOG_MESSAGE("[thershold,maxVal]="<<"["<<threshold<<","<<maxVal<<"]");
}


static Mat createInitialImage( const Mat& img, bool doubleImageSize, float sigma )
{
    Mat gray, gray_fpt;
    if( img.channels() == 3 || img.channels() == 4 )
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        img.copyTo(gray);

    //    gray.convertTo(gray_fpt, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
    gray.convertTo(gray_fpt, DataType<sift_wt>::type, 1.0, 0);

    float sig_diff;
    float SIFT_INIT_SIGMA=0.5;
    if( doubleImageSize )
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        Mat dbl;
        resize(gray_fpt, dbl, Size(gray.cols*2, gray.rows*2), 0, 0, INTER_LINEAR);
        GaussianBlur(dbl, dbl, Size(), sig_diff, sig_diff);
        return dbl;
    }
    else
    {
        sig_diff = sqrtf( std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        GaussianBlur(gray_fpt, gray_fpt, Size(), sig_diff, sig_diff);
        return gray_fpt;
    }
}

static float calcOrientationHist( const Mat& img, Point pt, int radius,
                                  float sigma, float* hist, int n )
{
    static int count=0;
    if(count<3)
    {
//        LOG_MESSAGE("[count,radius,sigma]=["<<count<<","<<radius<<","<<sigma<<"]");
        LOG_MESSAGE("[count,pt,radius,sigma]="<<count<<","<<pt<<","<<radius<<","<<sigma<<"]");
//        count++;
    }

    int i, j, k, len = (radius*2+1)*(radius*2+1);

    float expf_scale = -1.f/(2.f * sigma * sigma);
    AutoBuffer<float> buf(len*4 + n+4);
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;

    for( i = 0; i < n; i++ )
        temphist[i] = 0.f;

    for( i = -radius, k = 0; i <= radius; i++ )
    {
        int y = pt.y + i;
        if( y <= 0 || y >= img.rows - 1 )
            continue;
        for( j = -radius; j <= radius; j++ )
        {
            int x = pt.x + j;
            if( x <= 0 || x >= img.cols - 1 )
                continue;

            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1));
            float dy = (float)(img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x));

            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }

    len = k;

    // compute gradient values, orientations and the weights over the pixel neighborhood
    exp(W, W, len);
    fastAtan2(Y, X, Ori, len, true);
    magnitude(X, Y, Mag, len);

    for( k = 0; k < len; k++ )
    {
        int bin = cvRound((n/360.f)*Ori[k]);
        if( bin >= n )
            bin -= n;
        if( bin < 0 )
            bin += n;
        temphist[bin] += W[k]*Mag[k];
    }

    // smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];
    for( i = 0; i < n; i++ )
    {
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f) +
            (temphist[i-1] + temphist[i+1])*(4.f/16.f) +
            temphist[i]*(6.f/16.f);
    }

//    if(count==0){
////        imshow("count img",img);
//        cout<<"len="<<len<<endl;
//        cout<<"dxs=";
//        for(int i=0;i<len;i++){
//            cout<<X[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"dys=";
//        for(int i=0;i<len;i++){
//            cout<<Y[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"Ori=";
//        for(int i=0;i<len;i++){
//            cout<<Ori[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"Mag=";
//        for(int i=0;i<len;i++){
//            cout<<Mag[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"W=";
//        for(int i=0;i<len;i++){
//            cout<<W[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"tmpHist=";
//        for(int i=0;i<n;i++){
//            cout<<temphist[i]<<" ";
//        }
//        cout<<endl;

//        cout<<"hist=";
//        for(int i=0;i<n;i++){
//            cout<<hist[i]<<" ";
//        }
//        cout<<endl;
//    }
    count++;
    float maxval = hist[0];
    for( i = 1; i < n; i++ )
        maxval = std::max(maxval, hist[i]);

    return maxval;
}

static bool adjustLocalExtrema( const vector<Mat>& dog_pyr, KeyPoint& kpt, int octv,
                                int& layer, int& r, int& c, int nOctaveLayers,
                                float contrastThreshold, float edgeThreshold, float sigma ,Mat &keyPointsMat)
{
//    LOG_MESSAGE("");
    const double SIFT_FIXPT_SCALE=1.0;
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    const int SIFT_MAX_INTERP_STEPS=5;
    const int SIFT_IMG_BORDER=5;

    bool continueDebug=false;

    for( ; i < SIFT_MAX_INTERP_STEPS; i++ )
    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];

        Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                 (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                 (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = (float)img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        Matx33f H(dxx, dxy, dxs,
                  dxy, dyy, dys,
                  dxs, dys, dss);

        Vec3f X = H.solve(dD, DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

//        if((octv==3&&r==yzbx_debug_row&&c==yzbx_debug_col)||continueDebug){
//            continueDebug=true;
//            LOG_MESSAGE("[oi,sj,i,j]=["<<octv<<","<<layer<<","<<r<<","<<c<<"]");
//            LOG_MESSAGE("[dx,dy,dz,interpTimes]=["<<xc<<","<<xr<<","<<xi<<","<<i<<"]");
//            LOG_MESSAGE("dD"<<dD);
//            LOG_MESSAGE("H"<<H);
//        }

        if( std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f )
            break;

        if( std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3) )
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

//        if((octv==3&&r==yzbx_debug_row&&c==yzbx_debug_col)||continueDebug){
//            LOG_MESSAGE("[oi,sj,i,j]=["<<octv<<","<<layer<<","<<r<<","<<c<<"]");
//            LOG_MESSAGE("[dx,dy,dz,interpTimes]=["<<xc<<","<<xr<<","<<xi<<","<<i<<"]");
//            LOG_MESSAGE("dD"<<dD);
//            LOG_MESSAGE("H"<<H);
//        }

        if( layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER  ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER ){

            if(r<keyPointsMat.rows&&c<keyPointsMat.cols&&r>=0&&c>=0){
//                LOG_MESSAGE("[oi,sj,i,j]=["<<octv<<","<<layer<<","<<r<<","<<c<<"]");
//                LOG_MESSAGE("[dx,dy,dz,interpTimes]=["<<xc<<","<<xr<<","<<xi<<","<<i<<"]");
                keyPointsMat.at<uchar>(r,c)=210;
            }
            return false;
        }

    }

    // ensure convergence of interpolation
    if( i >= SIFT_MAX_INTERP_STEPS ){
        if(r<keyPointsMat.rows&&c<keyPointsMat.cols&&r>=0&&c>=0)
            keyPointsMat.at<uchar>(r,c)=220;

        return false;
    }

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const Mat& img = dog_pyr[idx];
        const Mat& prev = dog_pyr[idx-1];
        const Mat& next = dog_pyr[idx+1];
        Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                   (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                   (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
//        LOG_MESSAGE("[contr,contrastThreshold]=["<<contr*255<<","<<contrastThreshold<<"]");
        contr=contr*255;
        if( std::abs( contr ) * nOctaveLayers < contrastThreshold ){
            keyPointsMat.at<uchar>(r,c)=230;
            return false;
        }

//        LOG_MESSAGE("remove edge");
        // principal curvatures are computed using the trace and det of Hessian
        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if( det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det ){
            keyPointsMat.at<uchar>(r,c)=240;
            return false;
        }
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);
//    static int count=0;
//    if(count<20){
//        KeyPoint& kp=kpt;
//        LOG_MESSAGE("[octave,pt,size,response,sigma,nOctaveLayers]"<<kp.octave<<","<<kp.pt<<","<<kp.size<<","<<kp.response<<","<<sigma<<","<<nOctaveLayers<<"]");
//        count++;
//    }
    return true;
}
