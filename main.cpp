#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <alpr.h>
#include <unistd.h>
#include <ctime>
#include "RaspiCamCV.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


/** Function Headers */
std::vector<Rect> detectByCascade( Mat frame );

void detectByCascadeAndKalmanFilter( Mat frame, map<int,KalmanFilter>& KFm);

void bgSub(Ptr<BackgroundSubtractor> bg, SimpleBlobDetector& detector, Mat& frame);

void bgSubAndKalmanFilter(BackgroundSubtractorMOG2& bg, SimpleBlobDetector& detector, Mat& frame, map<int,KalmanFilter>& KFm);

/** Global variables */
String car_cascade_name = "../classifier/lbpcars1.xml";
String input_file, output_file;
CascadeClassifier car_cascade;
string window_name = "Capture - Car detection";
RNG rng(12345);

int openalprDemo() {
    // Initialize the library using United States style license plates.  
    // You can use other countries/regions as well (for example: "eu", "au", or "kr")
    alpr::Alpr openalpr("sg", "~/openalpr/config/openalpr.conf.in", "~/openalpr/runtime_data");

    // Optionally specify the top N possible plates to return (with confidences).  Default is 10
    openalpr.setTopN(20);

    // Optionally, provide the library with a region for pattern matching.  This improves accuracy by 
    // comparing the plate text with the regional pattern.
    openalpr.setDefaultRegion("sg");

    // Make sure the library loaded before continuing.  
    // For example, it could fail if the config/runtime_data is not found
    if (openalpr.isLoaded() == false)
    {
        std::cerr << "Error loading OpenALPR" << std::endl;
        return 1;
    }

    // Recognize an image file.  You could alternatively provide the image bytes in-memory.
    alpr::AlprResults results = openalpr.recognize("/path/to/image.jpg");

    // Iterate through the results.  There may be multiple plates in an image, 
    // and each plate return sthe top N candidates.
    for (int i = 0; i < results.plates.size(); i++)
    {
      alpr::AlprPlateResult plate = results.plates[i];
      std::cout << "plate" << i << ": " << plate.topNPlates.size() << " results" << std::endl;

        for (int k = 0; k < plate.topNPlates.size(); k++)
        {
          alpr::AlprPlate candidate = plate.topNPlates[k];
          std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
          std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
        }
    }
}

//detect and draw bounding box by backgroun subtraction
void bgSub(Ptr<BackgroundSubtractor> bg, SimpleBlobDetector& detector, Mat& frame){
    // Mat back;
    Mat fore;
    Mat frame_gray;



    vector<vector<Point> > contours;
    vector<KeyPoint> keypoints;

    namedWindow("Frame");
    // namedWindow("Background");
    cvtColor(frame,frame_gray,CV_BGR2GRAY);
    // imshow("gray", frame_gray);
    bg->apply(frame_gray,fore);
    // bg.getBackgroundImage(back);
    
    GaussianBlur( fore, fore, Size( 3, 3), 1,1);
    // threshold(fore,fore,100,255,THRESH_TRUNC);
    // Canny(fore,fore,30,90,3);

    
    // detector.detect(fore,keypoints);
    erode(fore,fore,Mat());
    dilate(fore,fore,Mat());
    
    findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // drawContours(frame_gray,contours,-1,Scalar(0,0,255),2);
    for (int i = 0; i < contours.size(); i++){
        vector<Point> contours_poly;
        approxPolyDP( contours[i], contours_poly, 3 , true);
        Rect boundRect = boundingRect(contours_poly);
        if (norm(boundRect.tl() - boundRect.br()) < 30)
            continue;
        rectangle(frame, boundRect.tl(), boundRect.br(), Scalar(255,0,255));
    }
    // drawKeypoints( frame, keypoints, frame, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    // for (int i = 0; i < keypoints.size(); i++){
    //     circle(frame,keypoints[i].pt,keypoints[i].size,Scalar(255,0,255));
    // }
    // imshow("Frame",frame);
    return;
}

//detect and display bounding box by haar cascade classifier
std::vector<Rect> detectByCascade( Mat frame )
{
    std::vector<Rect> cars;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    
    //-- Detect cars
    car_cascade.detectMultiScale( frame_gray, cars, 1.1, 6,0, Size(200,200), Size(300,300));
    return cars;
}

int main( int argc, const char** argv )
{
    std::vector<Rect> cars;
	time_t begin, end;
    int nCount = 0;
	RaspiCamCvCapture * capture;
	VideoCapture video_input;
	VideoWriter output;
	bool output_video = false;			
    Mat frame;
    if (argc > 1)
    {
        video_input.open(argv[1]);
		output.open(output_file, CV_FOURCC('F','M','P','4'), 14.999, Size(320,240));
		output_video = true;
    }
	else {
		cout << "using camera" <<endl;
		RASPIVID_CONFIG * config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
		config->width=1280;
		config->height=720;
		config->bitrate=0;	// zero: leave as default
		config->framerate=60;
		config->monochrome=1;
		capture = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config); 
		free(config);
	}

    if( !car_cascade.load( car_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    double scaleUp = 1;
    double scaleDown = 1;
	namedWindow("VideoCaptureTest");
	time( &begin);
    while( true ){
		//IplImage* image = raspiCamCvQueryFrame(capture);
		frame = cvarrToMat(raspiCamCvQueryFrame(capture));
        // float scale_w = frame.cols/160;
        // float scale_h = frame.rows/120;
        if (!frame.empty()) {
			//Mat canvas;                    
			//resize(frame, canvas, Size(), scaleDown, scaleDown);
            //cars = detectByCascade(canvas);
            // detectByCascadeAndKalmanFilter(frame, KFm);
            //for (int i = 0; i < cars.size(); i++){
				// rectangle(frame, cars[i].tl()*scaleUp, cars[i].br()*scaleUp, Scalar(255,0,255), 3);
				// rectangle(canvas, cars[i], Scalar(255,0,255));
				// Mat ROI(frame, cars[i]);
                // imshow("ROI", ROI);
			    // cvtColor(ROI, ROI_hsv, CV_BGR2HSV);
                // calcHist(&ROI_hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
            //}
        //}
		nCount++;
        //imshow("VideoCaptureTest", frame);
		if(output_video)
	        output << frame;
			int c = waitKey(1);
            if( (char)c == 'q' ) { break; }
        }
        else {
            cout << "empty frame\n";
            exit(0);
        }
    }
	time( &end);
	double elapsed = difftime(end, begin);
	cout << elapsed << " FPS = " << (float) ((float) (nCount) /elapsed) <<endl;
    return 0;
}

