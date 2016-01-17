#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

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

    //imshow("Square Detection", image);
}

////////////////////////////////////////////////////////////
//                                                        //
// End of code from squares.cpp (just functions, really)  ///////////////////////////////////////////////////////
//                                                        //
////////////////////////////////////////////////////////////






static void help()
{
    cout << "\nThis program demonstrates circle finding with the Hough transform.\n"
            "Usage:\n"
            "./houghcircles <image_name>, Default is ../data/board.jpg\n" << endl;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{help h ||}{@image|../data/board.jpg|}"
    );
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>("@image");
    if (filename.empty())
    {
        help();
        cout << "no image_name provided" << endl;
        return -1;
    }
    Mat bgrImage = imread(filename, 1);
    if(bgrImage.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    //square detection
    vector<vector<Point> > squares;   
    findSquares(bgrImage, squares);		
			
    //circle detection
    Mat grayImage;
    cvtColor(bgrImage, grayImage, COLOR_BGR2GRAY);
    medianBlur(grayImage, grayImage, 5);
    vector<Vec3f> circles;		     //for circles
    HoughCircles(grayImage, circles, HOUGH_GRADIENT, 1, 10,100, 30, 1, 30); // change the last two parameters
		// (min_radius & max_radius) to detect larger circles
    for( size_t i = 0; i < circles.size(); i++ )			//draw the circles
    {
	    Vec3i c = circles[i];
	    circle( bgrImage, Point(c[0], c[1]), c[2], Scalar(0,0,255), 3, LINE_AA);
	    circle( bgrImage, Point(c[0], c[1]), 2, Scalar(0,255,0), 3, LINE_AA);
    }
    
    drawSquares(bgrImage, squares); 				//now draw the squares
    imshow("Square and Circle Detection", bgrImage);
    cout << "Number of Squares = " << squares.size() << endl;

    for( size_t i = 0; i < squares.size(); i++ )
    {
	int n = (int)squares[i].size();
	cout << "Size of Square #" << i +1<< " = " << n << endl;
    }

    waitKey();
	
    return 0;

}
