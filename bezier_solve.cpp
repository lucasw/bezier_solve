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

#include "ceres/ceres.h"
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

  // VLOG(5) << CLTXT << "coeff " << CLNRM << std::endl << logMat(coeff);
  VLOG(5) << CLTXT << "coeff " << CLNRM << std::endl << (coeff);
  // VLOG(5) << CLTXT <<"control " << CLNRM << std::endl << logMat(control);

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

void getBezier(const double* const x,
    const int num_control_points,
    const int num_line_points,
    std::vector<cv::Point2f>& bezier_points) {
  std::vector<cv::Point2f> control_points;
  control_points.resize(num_control_points);
  for (int i = 0; i < num_control_points; i++) {
    control_points[i] = cv::Point2f(x[i * 2], x[i * 2 + 1]);
  }
  getBezier(control_points, bezier_points, num_line_points);

  return;
}

struct BezFunctor {
  BezFunctor(
      const size_t num_control_points,
      const int num_line_points,
      const cv::Rect obstacle,
      cv::Mat& out) :
    num_control_points(num_control_points),
    num_line_points(num_line_points),
    obstacle(obstacle),
    out(out) {
  }

  bool operator() (const double* const x, double* residual) const {
    // there are going to be a lot of redundant calls to getBezier
    // how to cache results?
    std::vector<cv::Point2f> bezier_points;
    getBezier(x, num_control_points, num_line_points, bezier_points);
    // obstacle center
    const float cx = obstacle.x + obstacle.width/2;
    const float cy = obstacle.y + obstacle.height/2;

    for (size_t i = 0; i < bezier_points.size(); i++) {
      const cv::Point2f bp = bezier_points[i];
      if (obstacle.contains(bp)) {
        // find how close point is to nearest rectangle edge
        if (bp.x > cx) {
          residual[i * 2] = obstacle.x + obstacle.width - bp.x;
        } else {
          residual[i * 2] = bp.x - obstacle.x;
        }

        if (bp.y > cy) {
          residual[i * 2 + 1] = obstacle.y + obstacle.height - bp.y;
        } else {
          residual[i * 2 + 1] = bp.y - obstacle.y;
        }
      } else {
        residual[i * 2] = 0;
        residual[i * 2 + 1] = 0;
      }
      //if (bezier_points[i].x
    }
    return true;
  }

private:
  const size_t num_control_points;
  const int num_line_points;
  const cv::Rect obstacle;
  cv::Mat out;
};

struct PathDistFunctor {
  PathDistFunctor(
      const size_t num_control_points,
      const int num_line_points,
      const std::vector<cv::Rect> obstacles,
      cv::Mat& out) :
    num_control_points(num_control_points),
    num_line_points(num_line_points),
    obstacles(obstacles),
    out(out) {
  }

  bool operator() (const double* const x, double* residual) const {
    std::vector<cv::Point2f> bezier_points;
    getBezier(x, num_control_points, num_line_points, bezier_points);
    LOG(INFO) << CLVAL << bezier_points.size() << CLNRM;
    residual[0] = 0;
    residual[1] = 0;
    for (size_t i = 1; i < bezier_points.size(); i++) {
      const cv::Point2f bp1 = bezier_points[i-1];
      const cv::Point2f bp2 = bezier_points[i];

      const float dx = bp2.x - bp1.x;
      const float dy = bp2.y - bp1.y;
      const float sc = 10.0;
      residual[0] += dx * dx * sc;
      residual[1] += dy * dy * sc;
    }
    return true;
  }

private:
  const size_t num_control_points;
  const int num_line_points;
  const std::vector<cv::Rect> obstacles;
  cv::Mat out;
};

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
  google::ParseCommandLineFlags(&argc, &argv, false);

  // TBD gflags
  static const int wd = 1280;
  static const int ht = 720;
  cv::Mat out = cv::Mat(cv::Size(wd, ht), CV_8UC3, cv::Scalar::all(0));

  const int num_control_points = 4;
  std::vector<cv::Point2f> control_points;
  control_points.resize(num_control_points);

  bool run = true;
  float o1x = FLAGS_o1x;
  float o1y = FLAGS_o1y;
  float o2x = FLAGS_o2x;
  float o2y = FLAGS_o2y;

  static const int num_obstacles = 3;
  std::vector<cv::Rect> obstacles;
  obstacles.push_back(cv::Rect(320, 250, 100, 120));
  obstacles.push_back(cv::Rect(800, 490, 100, 110));
  obstacles.push_back(cv::Rect(600, 320, 100, 110));
  
  static const int num_line_points = 300;

  while (run) {
    out *= 0.97;
  control_points[0] = cv::Point2f( 100, 100);
  control_points[1] = control_points[0] + cv::Point2f(o1x, o1y);
  control_points[3] = cv::Point2f( wd - 100, ht - 100);
  control_points[2] = control_points[3] + cv::Point2f(o2x, o2y);

  for (size_t i = 1; i < control_points.size(); i++) {
    cv::line(out, control_points[i-1], control_points[i],
        cv::Scalar(155, 255, 0), 2, CV_AA);
  }

  std::vector<cv::Point2f> bezier_points;
  // TBD gflag
  getBezier(control_points, bezier_points, num_line_points);

  for (size_t i = 1; i < bezier_points.size(); i++) {
    cv::line(out, bezier_points[i-1], bezier_points[i],
        cv::Scalar(255, 255, 255), 2);
  }

  for (size_t i = 0; i < obstacles.size(); i++) {
    const cv::Rect ob = obstacles[i];
    cv::rectangle(out, ob, cv::Scalar(128, 128, 100),
        -1); //cv::CV_FILLED);

    for (size_t i = 1; i < bezier_points.size(); i++) {
      if (ob.contains(bezier_points[i])) {
        VLOG(3) << i << " " << bezier_points[i] << " " 
            << ob.x << " " << ob.y << " " << ob.width << " " << ob.height;
        cv::line(out, bezier_points[i - 1], bezier_points[i],
            cv::Scalar(0, 0, 255), 3);
      }
    }
  }

  cv::imshow("bezier_solve", out);
  char key = cv::waitKey(0);

  if (key == 'q') { run = false; }
  if (key == 'd') { o1x += 5; }
  if (key == 'a') { o1x -= 4; }
  if (key == 'w') { o1y -= 5; }
  if (key == 's') { o1y += 4; }

  if (key == 'l') { o2x += 5; }
  if (key == 'j') { o2x -= 4; }
  if (key == 'i') { o2y -= 5; }
  if (key == 'k') { o2y += 4; }
  }

  ceres::Problem problem;
  double parameters[num_control_points * 2];

  for (size_t i = 0; i < num_control_points; i++) {
    parameters[i * 2] = control_points[i].x; 
    parameters[i * 2 + 1] = control_points[i].y; 
  }

  for (size_t i = 0; i < obstacles.size(); i++) {
  }
  
    ceres::CostFunction* cost_function = 
        new ceres::NumericDiffCostFunction<
            PathDistFunctor, 
            ceres::CENTRAL, 
            2, // num residuals 
            num_control_points * 2>( // num parameters 
                new PathDistFunctor(
                  num_control_points,
                  num_line_points,
                  obstacles,
                  out));
  
    problem.AddResidualBlock(
        cost_function, NULL, parameters);
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << summary.BriefReport() << "\n";

  for (size_t i = 1; i < num_control_points; i++) {
    LOG(INFO) << control_points[i];
  }
  for (size_t i = 1; i < num_control_points; i++) {
    LOG(INFO) << i << " " << parameters[i * 2] << " " << 
        parameters[i * 2 + 1];
  }

  // visualize output
  {
    std::vector<cv::Point2f> bezier_points;
    getBezier(parameters, num_control_points, 
        num_line_points, bezier_points);
  
    for (size_t i = 1; i < bezier_points.size(); i++) {
      //LOG(INFO) << i << " " << bezier_points[i];
      cv::line(out, bezier_points[i-1], bezier_points[i],
          cv::Scalar(5, 255, 55), 3);
    }
    
    cv::imshow("bezier_solve", out);
    cv::waitKey(0);
  }

  return 0;
}
