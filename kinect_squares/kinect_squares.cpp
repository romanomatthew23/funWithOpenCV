#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////
//                                                        //
//       The following code is from squares.cpp           ///////////////////////////////////////////////////////
//                                                        //
////////////////////////////////////////////////////////////


int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";   //probably getting rid of

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow("Square Detection", image);
}

////////////////////////////////////////////////////////////
//                                                        //
// End of code from squares.cpp (just functions, really)  ///////////////////////////////////////////////////////
//                                                        //
////////////////////////////////////////////////////////////




















static void help()
{
        cout << "\nThis program demonstrates usage of depth sensors (Kinect, XtionPRO,...).\n"
                        "The user gets some of the supported output images.\n"
            "\nAll supported output map types:\n"
            "1.) Data given from depth generator\n"
            "   CAP_OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
            "   CAP_OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
            "   CAP_OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
            "   CAP_OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
            "   CAP_OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
            "2.) Data given from RGB image generator\n"
            "   CAP_OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
            "   CAP_OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
         << endl;
}

static void colorizeDisparity( const Mat& gray, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f )
{
    CV_Assert( !gray.empty() );
    CV_Assert( gray.type() == CV_8UC1 );

    if( maxDisp <= 0 )
    {
        maxDisp = 0;
        minMaxLoc( gray, 0, &maxDisp );
    }

    rgb.create( gray.size(), CV_8UC3 );
    rgb = Scalar::all(0);
    if( maxDisp < 1 )
        return;

    for( int y = 0; y < gray.rows; y++ )
    {
        for( int x = 0; x < gray.cols; x++ )
        {
            uchar d = gray.at<uchar>(y,x);
            unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

            unsigned int hi = (H/60) % 6;
            float f = H/60.f - H/60;
            float p = V * (1 - S);
            float q = V * (1 - f * S);
            float t = V * (1 - (1 - f) * S);

            Point3f res;

            if( hi == 0 ) //R = V,  G = t,  B = p
                res = Point3f( p, t, V );
            if( hi == 1 ) // R = q, G = V,  B = p
                res = Point3f( p, V, q );
            if( hi == 2 ) // R = p, G = V,  B = t
                res = Point3f( t, V, p );
            if( hi == 3 ) // R = p, G = q,  B = V
                res = Point3f( V, q, p );
            if( hi == 4 ) // R = t, G = p,  B = V
                res = Point3f( V, p, t );
            if( hi == 5 ) // R = V, G = p,  B = q
                res = Point3f( q, p, V );

            uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
            uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
            uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

            rgb.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);
        }
    }
}

static float getMaxDisparity( VideoCapture& capture )
{
    const int minDistance = 400; // mm
    float b = (float)capture.get( CAP_OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
    float F = (float)capture.get( CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
    return b * F / minDistance;
}

static void printCommandLineParams()
{
    cout << "-cd=       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
    cout << "-fmd=      Fixed max disparity? (0 or 1; 0 by default) Ignored if disparity map is not colorized (-cd 0)." << endl;
    cout << "-mode=     image mode: resolution and fps, supported three values:  0 - CAP_OPENNI_VGA_30HZ, 1 - CAP_OPENNI_SXGA_15HZ," << endl;
    cout << "          2 - CAP_OPENNI_SXGA_30HZ (0 by default). Ignored if rgb image or gray image are not selected to show." << endl;
    cout << "-m=        Mask to set which output images are need. It is a string of size 5. Each element of this is '0' or '1' and" << endl;
    cout << "          determine: is depth map, disparity map, valid pixels mask, rgb image, gray image need or not (correspondently)?" << endl ;
    cout << "          By default -m=01010 i.e. disparity map and rgb image will be shown." << endl ;
    cout << "-r=        Filename of .oni video file. The data will grabbed from it." << endl ;
}

static void parseCommandLine( int argc, char* argv[], bool& isColorizeDisp, bool& isFixedMaxDisp, int& imageMode, bool retrievedImageFlags[],
                       string& filename, bool& isFileReading )
{
    filename.clear();
    cv::CommandLineParser parser(argc, argv, "{h help||}{cd|1|}{fmd|0|}{mode|0|}{m|01010|}{r||}");
    if (parser.has("h"))
    {
        help();
        printCommandLineParams();
        exit(0);
    }
    isColorizeDisp = (parser.get<int>("cd") != 0);
    isFixedMaxDisp = (parser.get<int>("fmd") != 0);
    imageMode = parser.get<int>("mode");
    int flags = parser.get<int>("m");
    isFileReading = parser.has("r");
    if (isFileReading)
        filename = parser.get<string>("r");
    if (!parser.check())
    {
        parser.printErrors();
        help();
        exit(-1);
    }
    if (flags % 100000 == 0)
    {
        cout << "No one output image is selected." << endl;
        exit(0);
    }
    for (int i = 0; i < 5; i++)
    {
        retrievedImageFlags[4 - i] = (flags % 10 != 0);
        flags /= 10;
    }
}

/*
 * To work with Kinect or XtonPRO the user must install OpenNI library and PrimeSensorModule for OpenNI and
 * configure OpenCV with WITH_OPENNI flag is ON (using CMake).
 */
int main( int argc, char* argv[] )
{
    //cout << "Hello. Can you hear me? I've been California Dreaming" << endl;
    //int counter=0;
    bool isColorizeDisp, isFixedMaxDisp;
    int imageMode;
    bool retrievedImageFlags[5];
    string filename;
    bool isVideoReading;
    parseCommandLine( argc, argv, isColorizeDisp, isFixedMaxDisp, imageMode, retrievedImageFlags, filename, isVideoReading );

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if( isVideoReading )
        capture.open( filename );
    else
    {
        capture.open( CAP_OPENNI2 );
        if( !capture.isOpened() )
            capture.open( CAP_OPENNI );
    }

    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    if( !isVideoReading )
    {
        bool modeRes=false;
        switch ( imageMode )
        {
            case 0:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
                break;
            case 1:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_15HZ );
                break;
            case 2:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_30HZ );
                break;
                //The following modes are only supported by the Xtion Pro Live
            case 3:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_30HZ );
                break;
            case 4:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_60HZ );
                break;
            default:
                CV_Error( Error::StsBadArg, "Unsupported image mode property.\n");
        }
        if (!modeRes)
            cout << "\nThis image mode is not supported by the device, the default value (CV_CAP_OPENNI_SXGA_15HZ) will be used.\n" << endl;
    }

    // Print some avalible device settings.
    cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH      " << capture.get( CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT     " << capture.get( CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FRAME_MAX_DEPTH  " << capture.get( CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
            "FPS              " << capture.get( CAP_PROP_FPS ) << endl <<
            "REGISTRATION     " << capture.get( CAP_PROP_OPENNI_REGISTRATION ) << endl;
    if( capture.get( CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT  " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS           " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
        if (!retrievedImageFlags[0] && !retrievedImageFlags[1] && !retrievedImageFlags[2])
            return 0;
    }
    
    
    for(;;)
    {
        Mat depthMap;
        Mat validDepthMap;
        Mat disparityMap;
        Mat bgrImage;
        Mat grayImage;
	//counter++;		             //for frame grab
	int square_flag = 1;		     // 1 to have square detection window
	vector<vector<Point> > squares;      //for squares
        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if( retrievedImageFlags[0] && capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.05f;
                Mat show; depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow( "depth map", show );
            }

            if( retrievedImageFlags[1] && capture.retrieve( disparityMap, CAP_OPENNI_DISPARITY_MAP ) )
            {
                if( isColorizeDisp )
                {
                    Mat colorDisparityMap;
                    colorizeDisparity( disparityMap, colorDisparityMap, isFixedMaxDisp ? getMaxDisparity(capture) : -1 );
                    Mat validColorDisparityMap;
                    colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );
		    //validColorDisparityMap = validColorDisparityMap.t();         //having fun
                    imshow( "colorized disparity map", validColorDisparityMap );
                }
                else
                {
                    imshow( "original disparity map", disparityMap );
                }
            }

            if( retrievedImageFlags[2] && capture.retrieve( validDepthMap, CAP_OPENNI_VALID_DEPTH_MASK ) )
                imshow( "valid depth mask", validDepthMap );

            if( retrievedImageFlags[3] && capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE ) )  {
                imshow( "rgb image", bgrImage );
		if(square_flag)			{
			findSquares(bgrImage, squares);
        		drawSquares(bgrImage, squares);
		}
		
	    }

            if( retrievedImageFlags[4] && capture.retrieve( grayImage, CAP_OPENNI_GRAY_IMAGE ) )
                imshow( "gray image", grayImage );
        }

        if( waitKey( 30 ) >= 0 )
            break;
    }

    return 0;
}
