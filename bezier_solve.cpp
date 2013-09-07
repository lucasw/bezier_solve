/*
 
  Copyright 2013 Lucas Walter

     This file is part of bezier_solve.

    Vimjay is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Vimjay is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Vimjay.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "opencv2/highgui/highgui.hpp"

// bash color codes
#define CLNRM "\e[0m"
#define CLWRN "\e[0;43m"
#define CLERR "\e[1;41m"
#define CLVAL "\e[1;36m"
#define CLTXT "\e[1;35m"
// BOLD black text with blue background
#define CLTX2 "\e[1;44m"

DEFINE_int32(o1x, 500, "offset from starting point for control point");
DEFINE_int32(o1y, 20, "offset from starting point for control point");
DEFINE_int32(o2x, -500, "offset from ending point for control point");
DEFINE_int32(o2y, 0, "offset from ending point for control point");

// from vimjay
bool getBezier(
    // TBD currently has to be 4
    const std::vector<cv::Point2f>& control_points,
    std::vector<cv::Point2f>& output_points,
    // number of intermediate points to generate
    const int num) {
  if (control_points.size() != 4) {
    LOG(ERROR) << control_points.size() << " != 4";
    return false;
  }

  /*
  // 2nd order 1 2 1
  double coeff_raw[4][4] = {
  { 1, 0, 0},
  {-2, 2, 0},
  { 1,-2, 1},
  };
  // 4th order 1 4 6 4 1

  general pattern
  bc(1) =    1 1
  bc(2) =   1 2 1
  bc(3) =  1 3 3 1
  bc(4) = 1 4 6 4 1

  bc(3,0) = 1
  bc(3,1) = 3

  (1-x)(1-x)(1-x) = 1 -3x 3x^2 -x^3
  (1 -2x x^2) (1-x)

  bc(+/-0) =   1
  bc(-1) =    1 -1
  bc(-2) =   1 -2  1
  bc(-3) =  1 -3  3 -1
  bc(-4) = 1 -4  6 -4  1
  ...

  { 
  bc(-3)*bc(3,0),  0     0      0
  bc(-2)*bc(3,1),        0      0
  bc(-1)*bc(3,2),               0
  bc(-0)*bc(3,3)
  }

  bc(3,0) is 1, bc(3,1) is 3, etc.

  Next higher order desired matrix:

  ' 1  0   0   0  0
  '-4  4   0   0  0
  ' 6 -12  6   0  0
  '-4  12 -12  4  0
  ' 1 -4   6  -4  1

*/

  // TBD how to generate programmatically
  // 1 3 3 1
  double coeff_raw[4][4] = {
    { 1,  0,  0, 0},
    {-3,  3,  0, 0},
    { 3, -6,  3, 0},
    {-1,  3, -3, 1}
  };
  cv::Mat coeff = cv::Mat(4, 4, CV_64F, coeff_raw);
  cv::Mat control = cv::Mat::zeros(4, 2, CV_64F);

  for (int i = 0; i < control.rows; i++) {
    control.at<double>(i, 0) = control_points[i].x;
    control.at<double>(i, 1) = control_points[i].y;
  }

  //VLOG(5) << CLTXT << "coeff " << CLNRM << std::endl << logMat(coeff);
  VLOG(5) << CLTXT << "coeff " << CLNRM << std::endl << (coeff);
  //VLOG(5) << CLTXT <<"control " << CLNRM << std::endl << logMat(control);

  cv::Point2f old_pt;

  output_points.clear();

  for (int i = 0; i < num; i++) {
    float t = static_cast<float>(i)/static_cast<float>(num - 1);

    // concentrate samples near beginning and end
    /*
    if (t < 0.5) {
      t *= t;
    } else {
      t = 1.0 - (1.0 - t) * (1.0 - t);
    }
    */
    double tee_raw[1][4] = {{ 1.0, t, t * t, t * t * t}};

    cv::Mat tee = cv::Mat(1, 4, CV_64F, tee_raw);
    cv::Mat pos = tee * coeff * control;

    cv::Point new_pt = cv::Point2f(pos.at<double>(0, 0), pos.at<double>(0, 1));

    output_points.push_back(new_pt);

    VLOG(5) << "pos " << t << " "
        << new_pt.x << " " << new_pt.y;
     // << std::endl << logMat(tee)
     // << std::endl << logMat(pos);
  }

  return true;
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  google::ParseCommandLineFlags(&argc, &argv, false);

  // TBD gflags
  static const int wd = 1280;
  static const int ht = 720;
  cv::Mat out = cv::Mat(cv::Size(wd, ht), CV_8UC3, cv::Scalar::all(0));
  
  std::vector<cv::Point2f> control_points;
  control_points.resize(4);
  control_points[0] = cv::Point2f( 100, 100);
  control_points[1] = control_points[0] + cv::Point2f(FLAGS_o1x, FLAGS_o1y); 
  control_points[3] = cv::Point2f( wd - 100, ht - 100);
  control_points[2] = control_points[3] + cv::Point2f(FLAGS_o2x, FLAGS_o2y); 
  
  for (size_t i = 1; i < control_points.size(); i++) {
    cv::line(out, control_points[i-1], control_points[i], 
        cv::Scalar(155, 255, 0), 2, CV_AA ); 
  }

  std::vector<cv::Point2f> bezier_points;
  // TBD gflag
  static const int num_points = 300;
  getBezier(control_points, bezier_points, num_points);
  
  for (size_t i = 1; i < bezier_points.size(); i++) {
    cv::line(out, bezier_points[i-1], bezier_points[i], 
        cv::Scalar(255, 255, 255), 2); 
  }

  cv::imshow("bezier_solve", out);
  cv::waitKey(0);

  return 0;
}
