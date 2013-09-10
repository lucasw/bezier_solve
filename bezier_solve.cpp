/*

  Copyright 2013 Lucas Walter

     This file is part of bezier_solve.

    bezier_solve is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    bezier_solve is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with bezier_solve.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <sstream>
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

//DEFINE_int32(line_points, 100, "number of points to break bezier line into");
DEFINE_double(o1x, 500.03, "offset from starting point for control point");
DEFINE_double(o1y, 20.1, "offset from starting point for control point");
DEFINE_double(o2x, -500.03, "offset from ending point for control point");
DEFINE_double(o2y, 0.004, "offset from ending point for control point");

void drawBezier(cv::Mat& out, 
    const std::vector<cv::Point2d>& control_points, 
    const cv::Scalar cp_col,
    const std::vector<cv::Point2d>& bezier_points, 
    const cv::Scalar bz_col,
    const std::vector<cv::Rect>& obstacles) {
  for (size_t i = 1; i < control_points.size(); i++) {
    cv::line(out, control_points[i-1], control_points[i],
        cp_col, 2, CV_AA);
  }
  for (size_t i = 0; i < control_points.size(); i++) {
    cv::circle(out, control_points[i], 4, cp_col, -1);
  }

  for (size_t i = 1; i < bezier_points.size(); i++) {
    //LOG(INFO) << i << " " << bezier_points[i];
    cv::line(out, bezier_points[i-1], bezier_points[i],
        bz_col, 2);
  }
  for (size_t i = 0; i < bezier_points.size(); i++) {
    cv::circle(out, bezier_points[i], 4, bz_col, -1);
  }

  for (size_t i = 0; i < obstacles.size(); i++) {
    const cv::Rect ob = obstacles[i];
    cv::rectangle(out, ob, cv::Scalar(128, 128, 100),
        2); //cv::CV_FILLED);

    for (size_t i = 1; i < bezier_points.size(); i++) {
      if (ob.contains(bezier_points[i])) {
        VLOG(3) << i << " " << bezier_points[i] << " " 
            << ob.x << " " << ob.y << " " << ob.width << " " << ob.height;
        cv::line(out, bezier_points[i - 1], bezier_points[i],
            bz_col/4.0 + cv::Scalar(0, 0, 255), 3);
      }
    }
  }

}

// from vimjay
bool getBezier(
    // TBD currently has to be 4
    const std::vector<cv::Point2d>& control_points,
    std::vector<cv::Point2d>& output_points,
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
    { 1.0,  0.0,  0.0, 0.0},
    {-3.0,  3.0,  0.0, 0.0},
    { 3.0, -6.0,  3.0, 0.0},
    {-1.0,  3.0, -3.0, 1.0}
  };
  cv::Mat coeff = cv::Mat(4, 4, CV_64F, coeff_raw);
  cv::Mat control = cv::Mat::zeros(4, 2, CV_64F);

  for (int i = 0; i < control.rows; i++) {
    control.at<double>(i, 0) = control_points[i].x;
    control.at<double>(i, 1) = control_points[i].y;
  }
  VLOG(2) << "control matrix " << std::endl << control;

  // VLOG(5) << CLTXT << "coeff " << CLNRM << std::endl << logMat(coeff);
  VLOG(5) << CLTXT << "coeff " << CLNRM << std::endl << (coeff);
  // VLOG(5) << CLTXT <<"control " << CLNRM << std::endl << logMat(control);

  cv::Point2d old_pt;

  output_points.clear();

  for (int i = 0; i < num; i++) {
    double t = static_cast<double>(i)/static_cast<double>(num - 1);

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

    cv::Point2d new_pt = cv::Point2d(pos.at<double>(0, 0), pos.at<double>(0, 1));
    
    VLOG(2) << i << " " << t << " pos " << pos << " " << new_pt.x << " " 
        << new_pt.y << ", " << pos.at<double>(0, 0) << "  " 
        << pos.at<double>(0, 0) - new_pt.x;

    output_points.push_back(new_pt);

    VLOG(5) << "pos " << t << " "
        << new_pt.x << " " << new_pt.y;
     // << std::endl << logMat(tee)
     // << std::endl << logMat(pos);
  }

  return true;
}

// version that takes the raw pointer
void getBezier(
    const double* const x,
    const cv::Point2d start_point,
    const cv::Point2d end_point,
    const int num_param_points,
    const int num_line_points,
    std::vector<cv::Point2d>& bezier_points,
    std::vector<cv::Point2d>& control_points
    ) {
  control_points.resize(num_param_points + 2);
  control_points[0] = start_point;
  control_points[control_points.size() - 1] = end_point;
  for (int i = 0; i < num_param_points; i++) {
    control_points[i + 1] = cv::Point2d(x[i * 2], x[i * 2 + 1]);
    VLOG(2) << i << " " << CLVAL 
        << control_points[i + 1].x << " "  
        << control_points[i + 1].y 
        << CLNRM;
  }
  getBezier(control_points, bezier_points, num_line_points);

  return;
}

struct BezFunctor {
  BezFunctor(
      const cv::Point2d start_point,
      const cv::Point2d end_point,
      const size_t num_control_points,
      const int num_line_points,
      const cv::Rect obstacle,
      cv::Mat& out) :
    start_point(start_point),
    end_point(end_point),
    num_control_points(num_control_points),
    num_line_points(num_line_points),
    obstacle(obstacle),
    out(out) {
  }

  bool operator() (const double* const x, double* residual) const {
    // there are going to be a lot of redundant calls to getBezier
    // how to cache results?
    std::vector<cv::Point2d> bezier_points;
    std::vector<cv::Point2d> control_points;
    getBezier(x, start_point, end_point, num_control_points, 
        num_line_points, bezier_points, control_points);
    // obstacle center
    const double cx = obstacle.x + obstacle.width/2;
    const double cy = obstacle.y + obstacle.height/2;
    
    double total_residual = 0;
    for (size_t i = 0; i < bezier_points.size(); i++) {
      const cv::Point2d bp = bezier_points[i];
      if (obstacle.contains(bp)) {
        // find how close point is to nearest rectangle edge
        double dx = 0;
        if (bp.x > cx) {
          dx = obstacle.x + obstacle.width - bp.x;
        } else {
          dx = bp.x - obstacle.x;
        }
        residual[i * 2] = dx * dx;

        double dy = 0;
        if (bp.y > cy) {
          dy = obstacle.y + obstacle.height - bp.y;
        } else {
          dy = bp.y - obstacle.y;
        }
        residual[i * 2 + 1] = dy * dy;
      } else {
        residual[i * 2] = 0;
        residual[i * 2 + 1] = 0;
      }
      total_residual += residual[i * 2] + residual[i * 2 + 1];
    }
    VLOG(1) << total_residual;
    return true;
  }

private:
  const cv::Point2d start_point;
  const cv::Point2d end_point;
  const size_t num_control_points;
  const int num_line_points;
  const cv::Rect obstacle;
  cv::Mat out;
};

struct PathDistFunctor {
  PathDistFunctor(
      const cv::Point2d start_point,
      const cv::Point2d end_point,
      const size_t num_control_points,
      const int num_line_points,
      const std::vector<cv::Rect> obstacles,
      cv::Mat& out) :
    start_point(start_point),
    end_point(end_point),
    num_control_points(num_control_points),
    num_line_points(num_line_points),
    obstacles(obstacles),
    out(out) {
  }

  bool operator() (const double* const x, double* residual) const {
    std::vector<cv::Point2d> bezier_points;
    std::vector<cv::Point2d> control_points;
    getBezier(x, start_point, end_point, 
        num_control_points, num_line_points, bezier_points,
        control_points);
    VLOG(5) << CLVAL << bezier_points.size() << CLNRM;
    residual[0] = 0;
    residual[1] = 0;
    VLOG(1) << 0 << " " << bezier_points[0] << " " << start_point;
    for (size_t i = 1; i < bezier_points.size(); i++) {
      const cv::Point2d bp1 = bezier_points[i-1];
      const cv::Point2d bp2 = bezier_points[i];

      {
        const double dx = bp2.x - bp1.x;
        const double dy = bp2.y - bp1.y;
        const double sc = 0.1;
        residual[0] += sqrt(dx * dx + dy * dy) * sc; 
        VLOG(2) << i << " " << dx << " " << dy << ", " << bp2.x << " " << bp2.y;
      }
      
      #if 0
      // penalize proximity to obstacles even if there isn't an intersection
      for (size_t i = 0; i < obstacles.size(); i++) {
        const cv::Point2d oc = cv::Point2d(
            obstacles[i].x + obstacles[i].width/2,
            obstacles[i].y + obstacles[i].height/2);
        const double dx = bp1.x - oc.x; 
        const double dy = bp1.y - oc.y;
        double dist = sqrt(dx * dx + dy * dy); 
        if (dist < 1) dist = 1;
        residual[1] += 10.0 * 1.0/dist;
      }
      #endif
    }
   
    #if 0
    drawBezier(out,
        control_points, cv::Scalar(5, 45, 10),
        bezier_points,  cv::Scalar(5, 128, 35),
        obstacles);
    cv::waitKey(1);
    #endif

    VLOG(1) << "residual " << residual[0] << " " << residual[1];
    return true;
  }

private:
  const cv::Point2d start_point;
  const cv::Point2d end_point;
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
  std::vector<cv::Point2d> control_points;
  control_points.resize(num_control_points);
  control_points[0] = cv::Point2d( 150.001, 150.020);
  control_points[3] = cv::Point2d( wd - 150.2, ht - 150.01);

  double o1x = FLAGS_o1x;
  double o1y = FLAGS_o1y;
  double o2x = FLAGS_o2x;
  double o2y = FLAGS_o2y;

  static const int num_obstacles = 5; // 3;
  std::vector<cv::Rect> obstacles;
  obstacles.resize(num_obstacles);
  
  cv::RNG rng;
  static const int num_line_points = 150; //FLAGS_line_points;
  int i = 0;

  bool run = true;
  while (run) {
    out *= 0; //92;
    
    for (int i = 0; i < num_obstacles; i++) {
      do {
        int width  = rng.uniform(20, ht/3);
        int height = rng.uniform(20, ht/3);
        obstacles[i] = cv::Rect(
            rng.uniform(0, wd - width),
            rng.uniform(0, ht - height),
            width,
            height);
      } while (
          (obstacles[i].contains(control_points[0])) || 
          (obstacles[i].contains(control_points[control_points.size()-1]))); 
    }


    control_points[1] = control_points[0] + cv::Point2d(o1x, o1y);
    control_points[2] = control_points[3] + cv::Point2d(o2x, o2y);
    std::vector<cv::Point2d> bezier_points;
    // TBD gflag
    getBezier(control_points, bezier_points, num_line_points);

    drawBezier(out, 
        control_points, cv::Scalar(125, 70, 60),
        bezier_points,  cv::Scalar(235, 235, 235),
        obstacles);

    {
      std::stringstream file_name;
      file_name << "bezier_solve_" << (i + 1000000) << ".jpg";
      cv::imwrite(file_name.str(), out);
      i++;
    }

    //cv::imshow("bezier_solve", out);
    //cv::waitKey(10);  
  #if 0
  if (key == 'q') { run = false; }
  if (key == 'd') { o1x += 5; }
  if (key == 'a') { o1x -= 4; }
  if (key == 'w') { o1y -= 5; }
  if (key == 's') { o1y += 4; }

  if (key == 'l') { o2x += 5; }
  if (key == 'j') { o2x -= 4; }
  if (key == 'i') { o2y -= 5; }
  if (key == 'k') { o2y += 4; }
  #endif

  ceres::Problem problem;
  // subtract the two end points, those are fixed
  const int num_param_points = num_control_points - 2;
  double parameters[num_param_points * 2];


  for (int i = 0; i < num_param_points; i++) {
    parameters[i * 2] = control_points[i + 1].x; 
    parameters[i * 2 + 1] = control_points[i + 1].y; 
  }

  for (size_t i = 0; i < obstacles.size(); i++) {
    ceres::CostFunction* cost_function = 
        new ceres::NumericDiffCostFunction<
            BezFunctor, 
            ceres::CENTRAL, 
            num_line_points * 2,
            num_param_points * 2>( // num parameters 
                new BezFunctor(
                  control_points[0],
                  control_points[control_points.size() - 1],
                  num_param_points,
                  num_line_points,
                  obstacles[i],
                  out));
  
    problem.AddResidualBlock(
        cost_function, NULL, parameters);
  }
  
  #if 1
    ceres::CostFunction* cost_function = 
        new ceres::NumericDiffCostFunction<
            PathDistFunctor, 
            ceres::CENTRAL, 
            2, // num residuals 
            num_param_points * 2>( // num parameters 
                new PathDistFunctor(
                  control_points[0],
                  control_points[control_points.size()-1],
                  num_param_points,
                  num_line_points,
                  obstacles,
                  out));
  
    problem.AddResidualBlock(
        cost_function, NULL, parameters);
  #endif

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  LOG(INFO) << "starting solve";
  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << summary.BriefReport() << "\n";
  
  LOG(INFO) << "control points";
  for (int i = 0; i < num_control_points; i++) {
    LOG(INFO) << i << " " << control_points[i];
  }
  for (int i = 0; i < num_param_points; i++) {
    LOG(INFO) << i + 1 << " " << parameters[i * 2] << " " << 
        parameters[i * 2 + 1];
  }

  // visualize output
  {
    std::vector<cv::Point2d> bezier_points;
    std::vector<cv::Point2d> control_points_tmp;
    getBezier(parameters, 
        control_points[0], 
        control_points[control_points.size() - 1], 
        num_param_points, 
        num_line_points, bezier_points, control_points);

    drawBezier(out,
        control_points, cv::Scalar(5, 85, 10),
        bezier_points,  cv::Scalar(5, 255, 55),
        obstacles);
    
    // TBD print bezier line length - have getBezier compute it?
    
    {
      std::stringstream file_name;
      file_name << "bezier_solve_" << (i + 1000000) << ".jpg";
      cv::imwrite(file_name.str(), out);
      i++;
    }
    cv::imshow("bezier_solve", out);
  }
    char key = cv::waitKey(1);
  
    if (key == 'q') run = false;
  } // end of run loop
  return 0;
}
