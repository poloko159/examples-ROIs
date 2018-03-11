#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <dlib/matrix.h>
#include <dlib/opencv/cv_image.h>

#include <iostream>
#include <fstream>
#include <iomanip>      // std::setprecision


using namespace dlib;

int total = 0;
int main(int argc, char** argv)
{
	cv::VideoCapture cap("QuestionVideo.avi"); //capture the video from webcam
	std::ofstream txt_out("Output.txt");
	if (!cap.isOpened())  // if not success, exit program
	{
		std::cout << "Cannot open the web cam" << std::endl;
		return -1;
	}
	int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
	int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT)); //get the height of frames of the video
	int frame_fps = cap.get(CV_CAP_PROP_FPS);
	cv::Size frame_size(frame_width, frame_height);
	cv::VideoWriter oVideoWriter("D:/project/VS2017/try/try/MyVideo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),frame_fps, frame_size, true);
	//std::cout << cap.get(CV_CAP_PROP_FOURCC);
	std::ifstream fin("sssssss.svm", std::ios::binary);
	if (!fin)
	{
		std::cout << "Can't find a trained object detector file no_entry.svm. " << std::endl;
		return EXIT_FAILURE;
	}
	typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;

	dlib::object_detector<image_scanner_type> detector;
	dlib::deserialize(detector, fin);
	int ch = 2;
	int iLowH_RED, iHighH_RED, iLowS_RED, iHighS_RED, iLowV_RED, iHighV_RED;
	if(ch==1){
	//RED THREADHOLD
		 iLowH_RED = 155;
		 iHighH_RED = 179;

		 iLowS_RED = 100;
		 iHighS_RED = 255;

		 iLowV_RED = 100;
		 iHighV_RED = 255;
	}
	if(ch==2){
	//BLUE THREADHOLD
		 iLowH_RED = 75;
		 iHighH_RED = 140;

		 iLowS_RED = 100;
		 iHighS_RED = 255;

		 iLowV_RED = 100;
		 iHighV_RED = 255;
	}
	//WHITE THREADHOLD
	int iLowH_WHITE = 0;
	int iHighH_WHITE = 180;

	int iLowS_WHITE = 0;
	int iHighS_WHITE = 255;

	int iLowV_WHITE = 200;
	int iHighV_WHITE = 255;
	while (true)
	{
		cv::Mat imgOriginal, imgOriginal_blr;
		bool bSuccess = cap.cv::VideoCapture::read(imgOriginal); // read a new frame from video
		if (!bSuccess) //if not success, break loop
		{
			std::cout << "Cannot read a frame from video stream" << std::endl;
			break;
		}
		cap.cv::VideoCapture::read(imgOriginal_blr);
		cv::Mat frame_HSV, frame_gray;

		// Creat mark matrix
		cv::cvtColor(imgOriginal_blr, frame_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);//Convert the captured frame from BGR to GRAY													//RGB to HSV
		cv::cvtColor(imgOriginal, frame_HSV, cv::ColorConversionCodes::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
		cv::Mat imgThresholded_RED;

		//RED DET
		int a = 5;
		cv::inRange(frame_HSV, cv::Scalar(iLowH_RED, iLowS_RED, iLowV_RED), cv::Scalar(iHighH_RED, iHighS_RED, iHighV_RED), imgThresholded_RED); //Threshold the image	RED

		//morphological opening (remove small objects from the foreground)
		cv::dilate(imgThresholded_RED, imgThresholded_RED, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(a, a)));																										  //morphological opening (removes small objects from the foreground)
		cv::erode(imgThresholded_RED, imgThresholded_RED, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(a, a)));

		//morphological closing(fill small holes in the foreground)
		cv::dilate(imgThresholded_RED, imgThresholded_RED, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(a, a)));
		cv::erode(imgThresholded_RED, imgThresholded_RED, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(a, a)));


		//Find and draw contours_RED
		std::vector <std::vector<cv::Point> > contours_RED; // Vector for storing contour
		std::vector<cv::Vec4i> hierarchy;
		cv::Mat threshold_output;

		cv::findContours(imgThresholded_RED, contours_RED, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// Approximate contours_RED to polygons + get bounding rects and circles
		std::vector<std::vector<cv::Point2i> > contours_RED_poly(contours_RED.size());
		std::vector<cv::Rect2i> boundRect_RED(contours_RED.size());
		std::vector<cv::Point_<float>>center(contours_RED.size());
		std::vector<float>radius(contours_RED.size());

		for (int i = 0; i < contours_RED.size(); i++)
		{
			cv::approxPolyDP(cv::Mat(contours_RED[i]), contours_RED_poly[i], 3, true);
			boundRect_RED[i] = cv::boundingRect(cv::Mat(contours_RED_poly[i]));
			cv::minEnclosingCircle((cv::Mat)contours_RED_poly[i], center[i], radius[i]);
		}

		for (int i = 0; i < contours_RED.size(); i++)
		{
			if (cv::moments(imgThresholded_RED(boundRect_RED[i])).m00>0)
			{

				cv::Mat cr = cv::Mat(imgOriginal(boundRect_RED[i]));
				cv::resize(cr, cr, cv::Size(std::min(cr.rows, cr.cols), std::min(cr.rows, cr.cols)));
			
				cv::imshow("s", cr);
				dlib::array2d<dlib::bgr_pixel> cimg;
				dlib::assign_image(cimg, dlib::cv_image<dlib::bgr_pixel>(cr));
				dlib::pyramid_up(cimg);
				dlib::pyramid_up(cimg);
				dlib::pyramid_up(cimg);

				std::vector<dlib::rectangle> rects = detector(cimg);
				std::cout << "Number of detections: " << int(rects.size()) << std::endl;
				if (int(rects.size()) != 0) {
					txt_out<<int(cap.get(CV_CAP_PROP_POS_FRAMES))<<" " << "2"<< " "<< boundRect_RED[i].x<< " "<< boundRect_RED[i].y<<" "<< boundRect_RED[i].x+ boundRect_RED[i].width<<" " << boundRect_RED[i].y + boundRect_RED[i].height<<"\n";
					cv::putText(imgOriginal_blr, "turn_left", cv::Point(boundRect_RED[i].x + boundRect_RED[i].width, boundRect_RED[i].y + boundRect_RED[i].height), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(225, 0, 0), 1);
					cv::rectangle(imgOriginal_blr, boundRect_RED[i], (255, 0, 0), 1, 8, 0);
					total++;
					//cv::imwrite("root.jpg", imgOriginal_blr);
				}

			}

		}
		cv::imshow("binary_RED", imgThresholded_RED);
		cv::imshow("org", imgOriginal_blr);
		oVideoWriter.write(imgOriginal_blr);

		if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			std::cout << "esc key is pressed by user" << std::endl;
			break;
		}


	}
	txt_out << total;
	oVideoWriter.release();

	return 0;
}