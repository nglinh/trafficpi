#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <alpr.h>
#include <unistd.h>
#include <ctime>
#include <thread>
#include "RaspiCamCV.h"

#include <iostream>
#include <stdio.h>

/** Global variables */
std::string car_cascade_name = "../classifier/carsg.xml";
std::string input_file, output_file;
cv::CascadeClassifier car_cascade;
std::string window_name = "Capture - Car detection";
int const NUM_THREAD = 4;

void detectCars(cv::VideoCapture video_capture, RaspiCamCvCapture* capture, bool video_mode, int* nCount) {
	std::vector<cv::Rect> cars;
	cv::Mat frame, mask, canvas;
	char c;
	double scaleUp = 6.25;
    double scaleDown = 0.16;
	while(!(video_mode && !video_capture.isOpened())) { 
		if(video_mode) {
			video_capture.read(frame);
			cv::cvtColor(frame, canvas, CV_BGR2GRAY);
		}
		else {
			frame = cv::cvarrToMat(raspiCamCvQueryFrame(capture));
			canvas = frame;
		}
		cv::resize(canvas, mask, cv::Size(), scaleDown, scaleDown);
		(*nCount)++;
		double car_min_pct = 0.05;
		double car_max_pct = 0.3;
		//cout << "Image size: " << mask.size() <<endl;
		int car_min_size = mask.size().height*car_min_pct;
		int car_max_size = mask.size().height*car_max_pct;
		//-- Detect cars
		car_cascade.detectMultiScale( mask, cars, 1.2, 4,0, cv::Size(car_min_size, car_min_size), cv::Size(car_max_size, car_max_size));
		if (cars.size()){
			std::cout << "detected " << cars.size() << std::endl;
		}
		int c = cv::waitKey(1);
        if( (char)c == 'q' )
			 break;
	}
	return;
}

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
void bgSub(cv::Ptr<cv::BackgroundSubtractor> bg, cv::SimpleBlobDetector& detector, cv::Mat& frame){
    // Mat back;
    cv::Mat fore;
    cv::Mat frame_gray;
    
	std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::KeyPoint> keypoints;

    cv::namedWindow("Frame");
    // namedWindow("Background");
    cv::cvtColor(frame,frame_gray,CV_BGR2GRAY);
    // imshow("gray", frame_gray);
    bg->apply(frame_gray,fore);
    // bg.getBackgroundImage(back);
    
    cv::GaussianBlur( fore, fore, cv::Size( 3, 3), 1,1);
    // threshold(fore,fore,100,255,THRESH_TRUNC);
    // Canny(fore,fore,30,90,3);

    
    // detector.detect(fore,keypoints);
	cv::erode(fore, fore, cv::Mat());
    cv::dilate(fore, fore, cv::Mat());
    
    cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    // drawContours(frame_gray,contours,-1,Scalar(0,0,255),2);    for (int i = 0; i < contours.size(); i++){
    for (int i = 0; i < contours.size(); i++){
        std::vector<cv::Point> contours_poly;
        cv::approxPolyDP( contours[i], contours_poly, 3 , true);
        cv::Rect boundRect = cv::boundingRect(contours_poly);
        if (cv::norm(boundRect.tl() - boundRect.br()) < 30)
            continue;
        cv::rectangle(frame, boundRect.tl(), boundRect.br(), cv::Scalar(255,0,255));
    }
    // drawKeypoints( frame, keypoints, frame, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    // for (int i = 0; i < keypoints.size(); i++){
    //     circle(frame,keypoints[i].pt,keypoints[i].size,Scalar(255,0,255));
    // }
    // imshow("Frame",frame);
    return;
}


//detect and display bounding box by haar cascade classifier
int main( int argc, const char** argv )
{
    std::vector<cv::Rect> cars;
	time_t begin, end;
    int nCount = 0;
	RaspiCamCvCapture * capture;
	cv::VideoCapture video_input;
	cv::VideoWriter output;
	bool video_mode = false;			
    cv::Mat frame;
    if (argc > 1)
    {
        video_input.open(argv[1]);
		//output.open(output_file, CV_FOURCC('F','M','P','4'), 14.999, Size(320,240));
		video_mode = true;
    }
	else {
		std::cout << "using camera" << std::endl;
		RASPIVID_CONFIG * config = (RASPIVID_CONFIG*)malloc(sizeof(RASPIVID_CONFIG));
		config->width=1280;
		config->height=720;
		config->bitrate=0;	// zero: leave as default
		config->framerate=60;
		config->monochrome=1;
		capture = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config); 
		free(config);
	}

    if( !car_cascade.load( car_cascade_name ) ){ 
		std::cout<< "--(!)Error loading" << std::endl; 
		return -1; 
	}
	cv::namedWindow("VideoCaptureTest");
	time( &begin);
	std::thread pool[NUM_THREAD];
	for (int i = 0; i < NUM_THREAD; i++) {
		pool[i] = std::thread(detectCars, video_input, capture, video_mode, &nCount);
	}
	std::cout << "All threads started" << std::endl;
	for (int i = 0; i < NUM_THREAD; i++) {
		pool[i].join();
		std::cout << "Thread " << i << "joined" <<std::endl;
	}
	time( &end);
	double elapsed = difftime(end, begin);
	//std::cout << elapsed << " FPS = " << (float) ((float) (nCount) /elapsed) << std::endl;
    return 0;
}

