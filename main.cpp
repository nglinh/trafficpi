#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <alpr.h>
#include <unistd.h>
#include <ctime>
#include <thread>
#include "RaspiCamCV.h"
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Path.h>
#include <Poco/URI.h>
#include <Poco/Exception.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

/** Global variables */
std::string car_cascade_name = "../classifier/lbpcars1.xml";
std::string input_file, output_file;
cv::CascadeClassifier car_cascade;
std::string window_name = "Capture - Car detection";
int const NUM_THREAD = 4;

void detectCars(cv::VideoCapture& video_capture, RaspiCamCvCapture* capture, bool video_mode) {
	alpr::Alpr openalpr("sg", "/home/pi/openalpr/config/openalpr.conf.in", "/home/pi/openalpr/runtime_data");
	openalpr.setDefaultRegion("sg");	
	if (!openalpr.isLoaded()){
		std::cerr << "Error loading OpenALPR" << std::endl;
		return;
	}

	std::string server_uri = std::getenv("TRAFFIC_EYE_URI");
	std::string rpi_id = std::getenv("RPI_ID");
	std::cout << rpi_id << std::endl;
	server_uri += rpi_id.substr(1);
	std::cout << server_uri << std::endl;

	//prepare uri
	Poco::URI uri(server_uri);

	//prepare session
	Poco::Net::HTTPClientSession session(uri.getHost(), uri.getPort());
	session.setKeepAlive(true);
	
	//prepare path
	std::string path(uri.getPathAndQuery());
	if(path.empty()) path = "/";


	std::vector<cv::Rect> cars;
	cv::Mat frame, mask, canvas, bgr_mask;
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
			cv::cvtColor(frame, frame, CV_GRAY2BGR);	//assuming it is captured in monochrome mode
			canvas = frame;
		}
		
		cv::resize(canvas, mask, cv::Size(), scaleDown, scaleDown);
		
		//Detect plates directly	
		/*std::vector<alpr::AlprRegionOfInterest> masks;
		masks.push_back(alpr::AlprRegionOfInterest(0,0,frame.size().width, frame.size().height));
		alpr::AlprResults results = openalpr.recognize(frame.data, frame.elemSize(), frame.size().width, frame.size().height, masks);
		for (int i = 0; i < results.plates.size(); i++) {
			alpr::AlprPlateResult plate = results.plates[i];
			alpr::AlprPlate candidate = plate.topNPlates[0];	//getting only the top result
			std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
			std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
			Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_POST, uri.getPathAndQuery(), Poco::Net::HTTPMessage::HTTP_1_1);
			req.setContentType("application/x-www-form-urlencoded");
			req.setKeepAlive(true);
			time_t detection_time = time(0);
			std::ostringstream convert;
			convert << detection_time;
			std::string reqBody("plate_id=" + candidate.characters + "&detection_time=" + convert.str());
			req.setContentLength( reqBody.length() );
			std::ostream& stream = session.sendRequest(req);
			stream << reqBody;
			req.write(std::cout);
		}*/

		//Detect car first, then detect plate
		double car_min_pct = 0.1;
		double car_max_pct = 0.7;
		//cout << "Image size: " << mask.size() <<endl;
		int car_min_size = mask.size().height*car_min_pct;
		int car_max_size = mask.size().height*car_max_pct;
		//-- Detect cars
		car_cascade.detectMultiScale( mask, cars, 1.1, 5,0, cv::Size(car_min_size, car_min_size), cv::Size(car_max_size, car_max_size));
		if (cars.size()){
			std::cout << "detected " << cars.size() << std::endl;
			std::vector<alpr::AlprRegionOfInterest> masks;
			for (int i = 0; i < cars.size() ; i++) {
				masks.push_back(alpr::AlprRegionOfInterest((int) cars[i].tl().x*scaleUp, (int) cars[i].tl().y*scaleUp, (int) cars[i].size().width*scaleUp, (int) cars[i].size().height*scaleUp));
				//cv::rectangle(frame, cars[i].tl() * scaleUp, cars[i].br() * scaleUp, cv::Scalar(0,0,255), 3);
			}
			//cv::cvtColor(frame, bgr_mask, CV_GRAY2BGR);
			alpr::AlprResults results = openalpr.recognize(frame.data, frame.elemSize(), frame.size().width, frame.size().height, masks);
			for (int i = 0; i < results.plates.size(); i++) {
				alpr::AlprPlateResult plate = results.plates[i];
				alpr::AlprPlate candidate = plate.topNPlates[0];	//getting only the top result
				std::cout << "    - " << candidate.characters << "\t confidence: " << candidate.overall_confidence;
				std::cout << "\t pattern_match: " << candidate.matches_template << std::endl;
				Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_POST, uri.getPathAndQuery(), Poco::Net::HTTPMessage::HTTP_1_1);
				req.setContentType("application/x-www-form-urlencoded");
				req.setKeepAlive(true);
				std::string reqBody("plate_id=" + candidate.characters); 
				req.setContentLength( reqBody.length() );
				std::ostream& stream = session.sendRequest(req);
				stream << reqBody;
				req.write(std::cout);
			}	
		}
		//cv::imshow(window_name, frame);
		int c = cv::waitKey(1);
		if( (char)c == 'q' )
			break;
	}
	return;
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
		std::cout << "video mode" << std::endl;
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
		config->framerate=49;
		config->monochrome=1;
		capture = (RaspiCamCvCapture *) raspiCamCvCreateCameraCapture2(0, config); 
		free(config);
	}

    if( !car_cascade.load( car_cascade_name ) ){ 
		std::cout<< "--(!)Error loading" << std::endl; 
		return -1; 
	}

	//cv::namedWindow(window_name);
	time( &begin);
	std::thread pool[NUM_THREAD];
	for (int i = 0; i < NUM_THREAD; i++) {
		pool[i] = std::thread(detectCars, std::ref(video_input), capture, video_mode);
	}
	std::cout << "All threads started" << std::endl;
	for (int i = 0; i < NUM_THREAD; i++) {
		pool[i].join();
		std::cout << "Thread " << i << "joined" <<std::endl;
	}
	//detectCars(video_input, capture, video_mode);
	time( &end);
	double elapsed = difftime(end, begin);
	//std::cout << elapsed << " FPS = " << (float) ((float) (nCount) /elapsed) << std::endl;
	if (!video_mode) {
		raspiCamCvReleaseCapture(&capture);
	}
    return 0;
}
