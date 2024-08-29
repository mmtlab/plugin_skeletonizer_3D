/*
  ____  _        _      _              _             _____ ____
 / ___|| | _____| | ___| |_ ___  _ __ (_)_______ _ _|___ /|  _ \
 \___ \| |/ / _ \ |/ _ \ __/ _ \| '_ \| |_  / _ \ '__||_ \| | | |
  ___) |   <  __/ |  __/ || (_) | | | | |/ /  __/ |  ___) | |_| |
 |____/|_|\_\___|_|\___|\__\___/|_| |_|_/___\___|_| |____/|____/
 */

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "skeletonizer3D"
#endif

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "../source.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pugg/Kernel.h>
#include <string>

#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include <models/hpe_model_openpose.h>
#include <models/input_data.h>
#include <openvino/openvino.hpp>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/common.hpp>

#ifdef KINECT_AZURE
// include Kinect libraries
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#endif

using namespace cv;
using namespace std;
using json = nlohmann::json;

struct color_point_t
{
    int16_t xyz[3];
    uint8_t rgb[3];
};

// Map of OpenPOSE keypoint names
// TODO: update with Miroscic names
map<int, string> keypoints_map = {
    {0, "NOS_"}, {1, "NEC_"}, {2, "SHOR"}, {3, "ELBR"}, {4, "WRIR"}, {5, "SHOL"}, {6, "ELBL"}, {7, "WRIL"}, {8, "HIPR"}, {9, "KNER"}, {10, "ANKR"}, {11, "HIPL"}, {12, "KNEL"}, {13, "ANKL"}, {14, "EYER"}, {15, "EYEL"}, {16, "EARR"}, {17, "EARL"}};

map<int, string> keypoints_map_azure = {
    {27, "NOS_"}, {3, "NEC_"}, {12, "SHOR"}, {13, "ELBR"}, {14, "WRIR"}, {5, "SHOL"}, {6, "ELBL"}, {7, "WRIL"}, {22, "HIPR"}, {23, "KNER"}, {24, "ANKR"}, {18, "HIPL"}, {19, "KNEL"}, {20, "ANKL"}, {30, "EYER"}, {28, "EYEL"}, {31, "EARR"}, {29, "EARL"}};

/**
 * @class Skeletonizer3D
 *
 * @brief Skeletonizer3D is a plugin that computes the 3D skeleton of a human
 * body from a depth map.
 *
 */
class Skeletonizer3D : public Source<json>
{

  /*
    ____  _        _   _                                 _
   / ___|| |_ __ _| |_(_) ___   _ __ ___   ___ _ __ ___ | |__   ___ _ __ ___
   \___ \| __/ _` | __| |/ __| | '_ ` _ \ / _ \ '_ ` _ \| '_ \ / _ \ '__/ __|
    ___) | || (_| | |_| | (__  | | | | | |  __/ | | | | | |_) |  __/ |  \__ \
   |____/ \__\__,_|\__|_|\___| |_| |_| |_|\___|_| |_| |_|_.__/ \___|_|  |___/

  */

  static cv::Mat renderHumanPose(HumanPoseResult &_result,
                                 OutputTransform &outputTransform)
  {
    if (!_result.metaData)
    {
      throw invalid_argument("ERROR: Renderer: metadata is null");
    }

    auto output_img = _result.metaData->asRef<ImageMetaData>().img;

    if (output_img.empty())
    {
      throw invalid_argument(
          "ERROR: Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(output_img);
    static const cv::Scalar colors[HPEOpenPose::keypointsNumber] = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 85, 0),
        cv::Scalar(255, 170, 0), cv::Scalar(255, 255, 0),
        cv::Scalar(170, 255, 0), cv::Scalar(85, 255, 0),
        cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 85),
        cv::Scalar(0, 255, 170), cv::Scalar(0, 255, 255),
        cv::Scalar(0, 170, 255), cv::Scalar(0, 85, 255),
        cv::Scalar(0, 0, 255), cv::Scalar(85, 0, 255),
        cv::Scalar(170, 0, 255), cv::Scalar(255, 0, 255),
        cv::Scalar(255, 0, 170), cv::Scalar(255, 0, 85)};
    static const pair<int, int> keypointsOP[] = {
        {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
    static const pair<int, int> keypointsAE[] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
    const int stick_width = 4;
    const cv::Point2f absent_keypoint(-1.0f, -1.0f);
    for (auto &pose : _result.poses)
    {
      for (size_t keypoint_idx = 0; keypoint_idx < pose.keypoints.size();
           keypoint_idx++)
      {
        if (pose.keypoints[keypoint_idx] != absent_keypoint)
        {
          outputTransform.scaleCoord(pose.keypoints[keypoint_idx]);
          cv::circle(output_img, pose.keypoints[keypoint_idx], 4,
                     colors[keypoint_idx], -1);
        }
      }
    }
    vector<pair<int, int>> limb_keypoints_ids;
    if (!_result.poses.empty())
    {
      if (_result.poses[0].keypoints.size() == HPEOpenPose::keypointsNumber)
      {
        limb_keypoints_ids.insert(limb_keypoints_ids.begin(),
                                  begin(keypointsOP), end(keypointsOP));
      }
      else
      {
        limb_keypoints_ids.insert(limb_keypoints_ids.begin(),
                                  begin(keypointsAE), end(keypointsAE));
      }
    }
    cv::Mat pane = output_img.clone();
    for (auto pose : _result.poses)
    {
      for (const auto &limb_keypoints_id : limb_keypoints_ids)
      {
        pair<cv::Point2f, cv::Point2f> limb_keypoints(
            pose.keypoints[limb_keypoints_id.first],
            pose.keypoints[limb_keypoints_id.second]);
        if (limb_keypoints.first == absent_keypoint ||
            limb_keypoints.second == absent_keypoint)
        {
          continue;
        }

        data_t mean_x = (limb_keypoints.first.x + limb_keypoints.second.x) / 2;
        data_t mean_y = (limb_keypoints.first.y + limb_keypoints.second.y) / 2;
        cv::Point difference = limb_keypoints.first - limb_keypoints.second;
        data_t length =
            sqrt(difference.x * difference.x + difference.y * difference.y);
        int angle =
            static_cast<int>(atan2(difference.y, difference.x) * 180 / CV_PI);
        vector<cv::Point> polygon;
        cv::ellipse2Poly(cv::Point2d(mean_x, mean_y),
                         cv::Size2d(length / 2, stick_width), angle, 0, 360, 1,
                         polygon);
        cv::fillConvexPoly(pane, polygon, colors[limb_keypoints_id.second]);
      }
    }
    cv::addWeighted(output_img, 0.4, pane, 0.6, 0, output_img);
    return output_img;
  }

  /*
    __  __      _   _               _
   |  \/  | ___| |_| |__   ___   __| |___
   | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
   | |  | |  __/ |_| | | | (_) | (_| \__ \
   |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

  */

public:
  // Constructor
  Skeletonizer3D() : _agent_id(PLUGIN_NAME) {}

  // Destructor
  ~Skeletonizer3D()
  {
    _cap.release();
#ifdef KINECT_AZURE

#endif
    delete _pipeline;
  }

  /* CONTEXT_RELATED METHODS **************************************************/
#ifdef KINECT_AZURE

#endif
  void setup_OpenPoseModel()
  {
    // setup inference model
    data_t aspect_ratio = _rgb_width / static_cast<data_t>(_rgb_height);
    _model.reset(new HPEOpenPose(_model_file, aspect_ratio, _tsize,
                                 static_cast<data_t>(_threshold), _layout));
  }

  void setup_Pipeline()
  {
    // setup pipeline
    _pipeline =
        new AsyncPipeline(std::move(_model),
                          ConfigFactory::getUserConfig(
                              _inference_device, _nireq, _nstreams, _nthreads),
                          _core);
    _frame_num = _pipeline->submitData(
        ImageInputData(_rgb), make_shared<ImageMetaData>(_rgb, _start_time));
  }

  /* COMMON METHODS ***********************************************************/

  void setup_VideoCapture()
  {

#ifdef KINECT_AZURE
    _device_config =
        K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    _device_config.color_format =
        K4A_IMAGE_FORMAT_COLOR_BGRA32; // <==== For Color image
    _device_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
    _device_config.depth_mode =
        K4A_DEPTH_MODE_NFOV_UNBINNED; // <==== For Depth image

    if (_params.contains("azure_device"))
    {
      _azure_device = _params["azure_device"];
      cout << "   Camera id: " << _azure_device << endl;
    }
    else
    {
      cout << "   Camera id (default): " << _azure_device << endl;
    }

    _device = k4a::device::open(_azure_device);
    _device.start_cameras(&_device_config);

    k4a::calibration sensor_calibration = _device.get_calibration(
        _device_config.depth_mode, _device_config.color_resolution);
    cout << "   Camera calibrated!" << endl;

    _pc_transformation = k4a_transformation_create(&sensor_calibration);

    k4abt_tracker_configuration_t trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
    if (_params.contains("CUDA"))
    {
      cout << "   Body tracker CUDA processor enabled: " << _params["CUDA"]
           << endl;
      if (_params["CUDA"] == true)
        trackerConfig.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
    }
    _tracker = k4abt::tracker::create(sensor_calibration, trackerConfig);

    // acquire a frame just to get the resolution
    _device.get_capture(&_k4a_rgbd,
                        std::chrono::milliseconds(K4A_WAIT_INFINITE));

    k4a::image colorImage = _k4a_rgbd.get_color_image();

    // from k4a::image to cv::Mat --> color image
    if (colorImage != NULL)
    {
      // get raw buffer
      uint8_t *buffer = colorImage.get_buffer();

      // convert the raw buffer to cv::Mat
      int rows = colorImage.get_height_pixels();
      int cols = colorImage.get_width_pixels();
      _rgb = cv::Mat(rows, cols, CV_8UC4, (void *)buffer, cv::Mat::AUTO_STEP);
      cvtColor(_rgb, _rgb, cv::COLOR_BGRA2BGR);
    }
#else
    // setup video capture
    _cap.open(_camera_device);
    if (!_cap.isOpened())
    {
      throw invalid_argument("ERROR: Cannot open the video camera");
    }
    _cap >> _rgb;

#endif
    _start_time = chrono::steady_clock::now();
    cv::Size resolution = _rgb.size();
    size_t found = _resolution_rgb.find("x");
    if (found != string::npos)
    {
      cout << "found? " << found << endl;
      resolution = cv::Size{
          stoi(_resolution_rgb.substr(0, found)),
          stoi(_resolution_rgb.substr(found + 1, _resolution_rgb.length()))};

      _output_transform = OutputTransform(_rgb.size(), resolution);
      resolution = _output_transform.computeResolution();

      cv::resize(_rgb, _rgb, cv::Size(resolution.width, resolution.height));
    }

    _rgb_height = resolution.height; //_rgb.rows;
    _rgb_width = resolution.width;   //_rgb.cols;
    cout << "   RGB Camera resolution: " << _rgb_width << "x" << _rgb_height
         << endl;
  }

  #ifdef KINECT_AZURE
  return_type create_point_cloud(k4a_transformation_t transformation_handle,
                                       const k4a_image_t depth_image,
                                       const k4a_image_t color_image)
{
    int depth_image_width_pixels = k4a_image_get_width_pixels(depth_image);
    int depth_image_height_pixels = k4a_image_get_height_pixels(depth_image);
    k4a_image_t transformed_color_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32,
                                                 depth_image_width_pixels,
                                                 depth_image_height_pixels,
                                                 depth_image_width_pixels * 4 * (int)sizeof(uint8_t),
                                                 &transformed_color_image))
    {
        printf("Failed to create transformed color image\n");
        return return_type::error;
    }

    k4a_image_t point_cloud_image = NULL;
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
                                                 depth_image_width_pixels,
                                                 depth_image_height_pixels,
                                                 depth_image_width_pixels * 3 * (int)sizeof(int16_t),
                                                 &point_cloud_image))
    {
        printf("Failed to create point cloud image\n");
        return return_type::error;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_transformation_color_image_to_depth_camera(transformation_handle,
                                                                               depth_image,
                                                                               color_image,
                                                                               transformed_color_image))
    {
        printf("Failed to compute transformed color image\n");
        return return_type::error;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_point_cloud(transformation_handle,
                                                                              depth_image,
                                                                              K4A_CALIBRATION_TYPE_DEPTH,
                                                                              point_cloud_image))
    {
        printf("Failed to compute point cloud\n");
        return return_type::error;
    }

    std::vector<color_point_t> points;

    int width = k4a_image_get_width_pixels(point_cloud_image);
    int height = k4a_image_get_height_pixels(transformed_color_image);

    int16_t *point_cloud_image_data = (int16_t *)(void *)k4a_image_get_buffer(point_cloud_image);
    uint8_t *color_image_data = k4a_image_get_buffer(transformed_color_image);

    for (int i = 0; i < width * height; i++)
    {
        color_point_t point;
        point.xyz[0] = point_cloud_image_data[3 * i + 0];
        point.xyz[1] = point_cloud_image_data[3 * i + 1];
        point.xyz[2] = point_cloud_image_data[3 * i + 2];
        if (point.xyz[2] == 0)
        {
            continue;
        }

        point.rgb[0] = color_image_data[4 * i + 0];
        point.rgb[1] = color_image_data[4 * i + 1];
        point.rgb[2] = color_image_data[4 * i + 2];
        uint8_t alpha = color_image_data[4 * i + 3];

        if (point.rgb[0] == 0 && point.rgb[1] == 0 && point.rgb[2] == 0 && alpha == 0)
        {
            continue;
        }

        points.push_back(point);
    }

    // convert the points to a point cloud of Mat type
    _point_cloud = cv::Mat(points.size(), 6, CV_32F);
    for (size_t i = 0; i < points.size(); i++)
    {
        _point_cloud.at<float>(i, 0) = points[i].xyz[0];
        _point_cloud.at<float>(i, 1) = points[i].xyz[1];
        _point_cloud.at<float>(i, 2) = points[i].xyz[2];
        _point_cloud.at<float>(i, 3) = points[i].rgb[2];
        _point_cloud.at<float>(i, 4) = points[i].rgb[1];
        _point_cloud.at<float>(i, 5) = points[i].rgb[0];
    }
    

    // Save the point cloud to a ply file
    //write_ply_from_points_vector(points, "../plugin_skeletonizer_3D/test_points.ply");
    //write_ply_from_cv_mat(_point_cloud, "../plugin_skeletonizer_3D/test_cv_mat.ply");

    return return_type::success;
}

// Write the point cloud to a ply file
void write_ply_from_cv_mat(cv::Mat point_cloud, const char *file_name){

#define PLY_START_HEADER "ply"
#define PLY_END_HEADER "end_header"
#define PLY_ASCII "format ascii 1.0"
#define PLY_ELEMENT_VERTEX "element vertex"

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << PLY_START_HEADER << std::endl;
    ofs << PLY_ASCII << std::endl;
    ofs << PLY_ELEMENT_VERTEX << " " << point_cloud.rows << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "property uchar red" << std::endl;
    ofs << "property uchar green" << std::endl;
    ofs << "property uchar blue" << std::endl;
    ofs << PLY_END_HEADER << std::endl;
    ofs.close();

    std::stringstream ss;
    for (int i = 0; i < point_cloud.rows; ++i)
    {
      ss << point_cloud.at<float>(i, 0) << " " << point_cloud.at<float>(i, 1) << " " << point_cloud.at<float>(i, 2);
      ss << " " << point_cloud.at<float>(i, 3) << " " << point_cloud.at<float>(i, 4) << " " << point_cloud.at<float>(i, 5);
      ss << std::endl;
    }
    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
}

// Write the point cloud to a ply file
static void write_ply_from_points_vector(std::vector<color_point_t> points,
                                             const char *file_name){

#define PLY_START_HEADER "ply"
#define PLY_END_HEADER "end_header"
#define PLY_ASCII "format ascii 1.0"
#define PLY_ELEMENT_VERTEX "element vertex"

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << PLY_START_HEADER << std::endl;
    ofs << PLY_ASCII << std::endl;
    ofs << PLY_ELEMENT_VERTEX << " " << points.size() << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "property uchar red" << std::endl;
    ofs << "property uchar green" << std::endl;
    ofs << "property uchar blue" << std::endl;
    ofs << PLY_END_HEADER << std::endl;
    ofs.close();

    std::stringstream ss;
    for (size_t i = 0; i < points.size(); ++i)
    {
        // image data is BGR
        ss << (float)points[i].xyz[0] << " " << (float)points[i].xyz[1] << " " << (float)points[i].xyz[2];
        ss << " " << (float)points[i].rgb[2] << " " << (float)points[i].rgb[1] << " " << (float)points[i].rgb[0];
        ss << std::endl;
    }
    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
}

  #endif

  /**
   * @brief Acquire a frame from a camera device. Camera ID is defined in the
   * parameters list.
   *
   * The acquired frame is stored in the #_k4a_rgbd, #_rgbd and #_rgb
   * attributes.
   *
   * @see set_params
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type acquire_frame(bool dummy = false)
  {
    // acquire last frame from the camera device
    // if camera device is a Kinect Azure, use the Azure SDK
    // and translate the frame in OpenCV format

    if (dummy)
    {
      // TODO: load a file
      throw invalid_argument("ERROR: Dummy not implemented");
    }
    else
    {
#ifdef KINECT_AZURE
      // acquire and translate into _rgb and _rgbd
      const clock_t begin_time = clock();

      // acquire and translate into _rgb and _rgbd
      if (!_device.get_capture(&_k4a_rgbd,
                               std::chrono::milliseconds(K4A_WAIT_INFINITE)))
        return return_type::error;

      // acquire and store into _rgb (RGB) and _rgbd (RGBD), if available
      k4a::image colorImage = _k4a_rgbd.get_color_image();
      k4a_image_t color_handle = colorImage.handle();
      k4a_image_reference(color_handle);	

      // from k4a::image to cv::Mat --> color image
      if (colorImage != NULL)
      {
        // get raw buffer
        uint8_t *buffer = colorImage.get_buffer();

        // convert the raw buffer to cv::Mat
        int rows = colorImage.get_height_pixels();
        int cols = colorImage.get_width_pixels();
        _rgb = cv::Mat(rows, cols, CV_8UC4, (void *)buffer, cv::Mat::AUTO_STEP);
        cvtColor(_rgb, _rgb, cv::COLOR_BGRA2BGR);
        //_rgb.convertTo(_rgb, CV_8UC3);
      }

      _depth_image = _k4a_rgbd.get_depth_image();
      k4a_image_t depth_handle = _depth_image.handle();
      k4a_image_reference(depth_handle);	

      // from k4a::image to cv::Mat --> depth image
      if (_depth_image != NULL)
      {
        // get raw buffer
        uint8_t *buffer = _depth_image.get_buffer();

        // convert the raw buffer to cv::Mat
        int rows = _depth_image.get_height_pixels();
        int cols = _depth_image.get_width_pixels();
        _rgbd = cv::Mat(rows, cols, CV_16U, (void *)buffer, cv::Mat::AUTO_STEP);
      }

      // Debug function to save the point cloud in a .ply file
      //point_cloud_color_to_depth(_pc_transformation, depth_handle, color_handle, "../plugin_skeletonizer_3D/test.ply");


#else
      _cap >> _rgb;
#endif

      if (_rgb.empty())
      {
        // Input stream is over
        return return_type::error;
      }
      cv::resize(_rgb, _rgb, cv::Size(_rgb_width, _rgb_height));
    }
    _start_time = chrono::steady_clock::now();

    return return_type::success;
  }

  /* LEFT BRANCH =============================================================*/
  
  /**
   * @brief Compute the skeleton from the depth map.
   *
   * Compute the skeleton from the depth map. The resulting skeleton is stored
   * in #_skeleton3D attribute as a map of 3D points.
   *
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type skeleton_from_depth_compute(bool debug = false)
  {
#ifdef KINECT_AZURE

    if (!_tracker.enqueue_capture(_k4a_rgbd))
    {
      // It should never hit timeout when K4A_WAIT_INFINITE is set.
      cout << "Error! Add capture to tracker process queue timeout!" << endl;
      return return_type::error;
    }

    _body_frame = _tracker.pop_result();
    if (_body_frame != nullptr)
    {      
      // Only take one body (always the first one)
      k4abt_body_t body = _body_frame.get_body(0);

      json frame_result_json;

      for (const auto& [index, keypoint_name] : keypoints_map_azure)
      {
        k4a_float3_t position = body.skeleton.joints[index].position;
        k4abt_joint_confidence_level_t confidence_level = body.skeleton.joints[index].confidence_level;

        frame_result_json["poses"][keypoint_name] = { position.v[0], position.v[1], position.v[2] };
        frame_result_json["cov"][keypoint_name] = { confidence_level };
      }

      if (debug)
        cout << frame_result_json.dump(4) << endl;
        
    }
    else
    {
      //  It should never hit timeout when K4A_WAIT_INFINITE is set.
      cout << "Error! Pop body frame result time out!" << endl;
      return return_type::error;
    }

    return return_type::success;
#else
    // NOOP
    return return_type::success;
#endif
  }

  /**
   * @brief Remove unnecessary points from the point cloud
   *
   * Make the point cloud lighter by removing unnecessary points, so that it
   * can be sent to the database via network
   *
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type point_cloud_filter(bool debug = false)
  {

#ifdef KINECT_AZURE
    // get the body index map from _k4a_rgbd

    // TODO: _body_frame is obtained in the skeleton_from_depth_compute method! We don't like this dependency
    // Retrieve the body tracking result

    if (_body_frame != nullptr)
    {
      k4a::image body_index_map = _body_frame.get_body_index_map();


      // mask the depth image with the body index map
      k4a::image masked_depth_image = k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
                                                          _depth_image.get_width_pixels(), _depth_image.get_height_pixels(),
                                                          _depth_image.get_stride_bytes());
      
      mask_depth_with_body_index(_depth_image, body_index_map, masked_depth_image);

      k4a_image_t masked_depth_handle = masked_depth_image.handle();
      k4a_image_reference(masked_depth_handle);	

      // from k4a::image to cv::Mat --> depth image
      if (masked_depth_image != NULL)
      {
        // get raw buffer
        uint8_t *buffer = masked_depth_image.get_buffer();
        cout << "Depth image buffer:" << buffer << endl;

        // convert the raw buffer to cv::Mat
        int rows = masked_depth_image.get_height_pixels();
        int cols = masked_depth_image.get_width_pixels();
        _rgbd_filtered = cv::Mat(rows, cols, CV_16U, (void *)buffer, cv::Mat::AUTO_STEP);
      }

      // convert the depth image to a point cloud
      k4a::image colorImage = _k4a_rgbd.get_color_image();
      k4a_image_t color_handle = colorImage.handle();
      k4a_image_reference(color_handle);

      create_point_cloud(_pc_transformation, masked_depth_handle, color_handle);

    }
    else
    {
      cout << "No body frame detected!" << endl;
    }
    
#endif

    return return_type::success;
  }

#ifdef KINECT_AZURE
  void mask_depth_with_body_index(const k4a::image &depth_image, const k4a::image &body_index_map, k4a::image &masked_depth_image) {
      // Get image dimensions
      int width = depth_image.get_width_pixels();
      int height = depth_image.get_height_pixels();

      // Get pointers to the image data
      const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_image.get_buffer());
      const uint8_t* body_index_data = reinterpret_cast<const uint8_t*>(body_index_map.get_buffer());
      uint16_t* masked_depth_data = reinterpret_cast<uint16_t*>(masked_depth_image.get_buffer());

      // Iterate over each pixel
      for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
              int idx = y * width + x;

              // Check if the pixel belongs to a body
              if (body_index_data[idx] != K4ABT_BODY_INDEX_MAP_BACKGROUND) {
                  // Copy the depth value
                  masked_depth_data[idx] = depth_data[idx];
              } else {
                  // Set to zero or any background value (e.g., 0 for no depth)
                  masked_depth_data[idx] = 0;
              }
          }
      }
      
  }
#endif

  /**
   * @brief Transform the 3D skeleton coordinates in the global reference frame
   *
   * Use the extrinsic camera parameters to transorm the 3D skeleton coordinates
   * just before sending them as plugin output.
   *
   * @return return_type
   */
  return_type coordinate_transfrom(bool debug = false)
  {
    return return_type::success;
  }

  /* RIGHT BRANCH ============================================================*/

  /**
   * @brief Compute the skeleton from RGB images only
   *
   * Compute the skeleton from RGB images only. On success, the field
   * #_skeleton2D is updated (as a map of 2D points).
   * Also, the field #_heatmaps is updated with the joints heatmaps (one per
   * joint).
   *
   * There is a configuration flag for optionally skipping this branch
   * on Azure agents.
   *
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type skeleton_from_rgb_compute(bool debug = false)
  {
    if (_pipeline->isReadyToProcess())
    {
      _frame_num = _pipeline->submitData(
          ImageInputData(_rgb), make_shared<ImageMetaData>(_rgb, _start_time));
    }
    else
    {
      return return_type::warning;
    }

    // Waiting for free input slot or output data available. Function will
    // return immediately if any of them are available.
    _pipeline->waitForData();
    if (!(_result = _pipeline->getResult()))
    {
      return return_type::warning;
    }

    if (debug)
    {
      renderHumanPose(_result->asRef<HumanPoseResult>(), _output_transform);
    }
    _frames_processed++;

    return return_type::success;
  }

  /**
   * @brief Compute the hessians for joints
   *
   * Compute the hessians for joints on the RGB frame based on the #_heatmaps
   * field.
   *
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type hessian_compute(bool debug = false)
  {
    // y -> rows
    // x -> cols

    size_t n_pixel = 10; // of how many pixels I move

    _keypoints_list.clear();
    _keypoints_cov.resize(HPEOpenPose::keypointsNumber);
    _keypoints_cov.clear();
    _poses.clear();
    _poses = _result->asRef<HumanPoseResult>().poses;
    // cout << "poses.size()-----> " << poses.size() << endl;
    if (_poses.size() > 0)
    { // at least one person

      for (auto &keypoint :
           _poses[0].keypoints)
      { // if I have more than one person, I take the
        // first with id[0]

        if (keypoint.x > _rgb_width)
        {
          keypoint.x = _rgb_width - 1;
        }
        if (keypoint.y > _rgb_height)
        {
          keypoint.y = _rgb_height - 1;
        }
        _keypoints_list.push_back(cv::Point2i(
            keypoint.x,
            keypoint.y)); // I always have 18 keypoints, if there is no (-1,-1)
      }

      for (int ii = 0; ii < HPEOpenPose::keypointsNumber; ii++)
      {

        if (_keypoints_list[ii].x > 0 && _keypoints_list[ii].y > 0)
        {

          if (_keypoints_list[ii].y < n_pixel)
          {
            _keypoints_list[ii].y = n_pixel;
          }
          else if (_keypoints_list[ii].y >= _rgb_height - n_pixel)
          {
            _keypoints_list[ii].y = _rgb_height - n_pixel - 1;
          }

          if (_keypoints_list[ii].x < n_pixel)
          {
            _keypoints_list[ii].x = n_pixel;
          }
          else if (_keypoints_list[ii].x >= _rgb_width - n_pixel)
          {
            _keypoints_list[ii].x = _rgb_width - n_pixel - 1;
          }

          cv::Mat _heat_map = _result->asRef<HumanPoseResult>().heatMaps[ii];
          cv::resize(_heat_map, _heat_map, cv::Size(_rgb_width, _rgb_height));

          data_t H_ri_ci = _heat_map.at<data_t>(_keypoints_list[ii].y,
                                                _keypoints_list[ii].x);
          data_t H_ri_ciPLUSn = _heat_map.at<data_t>(
              _keypoints_list[ii].y, _keypoints_list[ii].x + n_pixel);
          data_t H_ri_ciMINn = _heat_map.at<data_t>(
              _keypoints_list[ii].y, _keypoints_list[ii].x - n_pixel);
          data_t H_riPLUSn_ci = _heat_map.at<data_t>(
              _keypoints_list[ii].y + n_pixel, _keypoints_list[ii].x);
          data_t H_riMINn_ci = _heat_map.at<data_t>(
              _keypoints_list[ii].y - n_pixel, _keypoints_list[ii].x);

          data_t H22 = (1.0 / (n_pixel * n_pixel)) *
                       (H_ri_ciPLUSn - 2 * H_ri_ci + H_ri_ciMINn);
          data_t H11 = (1.0 / (n_pixel * n_pixel)) *
                       (H_riPLUSn_ci - 2 * H_ri_ci + H_riMINn_ci);

          data_t H_riMINn_ciMINn = _heat_map.at<data_t>(
              _keypoints_list[ii].y - n_pixel, _keypoints_list[ii].x - n_pixel);
          data_t H_riMINn_ciPLUSn = _heat_map.at<data_t>(
              _keypoints_list[ii].y - n_pixel, _keypoints_list[ii].x + n_pixel);
          data_t H_riPLUSn_ciMINn = _heat_map.at<data_t>(
              _keypoints_list[ii].y + n_pixel, _keypoints_list[ii].x - n_pixel);
          data_t H_riPLUSn_ciPLUSn = _heat_map.at<data_t>(
              _keypoints_list[ii].y + n_pixel, _keypoints_list[ii].x + n_pixel);

          data_t H12 = (1.0 / (4 * n_pixel * n_pixel)) *
                       (H_riPLUSn_ciPLUSn - H_riPLUSn_ciMINn -
                        H_riMINn_ciPLUSn + H_riMINn_ciMINn);
          data_t H21 = H12;

          Eigen::Matrix2f A;
          A << H11, H12, H21, H22;

          Eigen::Matrix2f C = A.inverse();
          Eigen::EigenSolver<Eigen::Matrix2f> s(C); // the instance s(C)
                                                    // includes the eigensystem

          complex<data_t> D11_tmp = s.eigenvalues()[0];
          data_t D11 = D11_tmp.real();
          complex<data_t> D22_tmp = s.eigenvalues()[1];
          data_t D22 = D22_tmp.real();

          complex<data_t> V11_tmp = s.eigenvectors()(0, 0);
          data_t V11 = V11_tmp.real();
          complex<data_t> V21_tmp = s.eigenvectors()(1, 0);
          data_t V21 = V21_tmp.real();

          data_t perc_prob = 0.95;
          data_t xradius = sqrt(-D11 * (-2) * log(1 - perc_prob));
          data_t yradius = sqrt(-D22 * (-2) * log(1 - perc_prob));

          data_t alpha = atan2(V21, V11);

          _keypoints_cov[ii].x = xradius;
          _keypoints_cov[ii].y = yradius;
          _keypoints_cov[ii].z = alpha;

          if (debug)
          {
            vector<data_t> theta;
            for (data_t j = 0; j < 2 * M_PI; j += 2 * (M_PI / 40))
            {
              theta.push_back(j);
            }

            vector<data_t> x_ellips;
            vector<data_t> y_ellips;

            for (int j = 0; j < theta.size(); j++)
            {
              x_ellips.push_back(xradius * cos(theta[j]));
              y_ellips.push_back(yradius * sin(theta[j]));
            }

            vector<cv::Point2f> ellipse_points;
            for (int j = 0; j < x_ellips.size(); j++)
            {
              data_t element_1 =
                  (cos(alpha) * x_ellips[j] + (-sin(alpha)) * y_ellips[j]) +
                  _keypoints_list[ii].x;
              data_t element_2 =
                  (sin(alpha) * x_ellips[j] + cos(alpha) * y_ellips[j]) +
                  _keypoints_list[ii].y;
              ellipse_points.push_back(cv::Point2f(element_1, element_2));
            }

            // Draw keypoints
            cv::circle(_rgb,
                       cv::Point(_keypoints_list[ii].x, _keypoints_list[ii].y),
                       5, cv::Scalar(0, 255, 0), cv::FILLED);

            // Draw ellipse points
            for (int i = 0; i < ellipse_points.size(); ++i)
            {

              if (ellipse_points[i].x < 0)
              {
                ellipse_points[i].x = 1;
              }
              if (ellipse_points[i].y < 0)
              {
                ellipse_points[i].y = 1;
              }
              if (ellipse_points[i].x > _rgb_width)
              {
                ellipse_points[i].x = _rgb_width - 1;
              }
              if (ellipse_points[i].y > _rgb_height)
              {
                ellipse_points[i].y = _rgb_height - 1;
              }

              cv::circle(_rgb, ellipse_points[i], 2, cv::Scalar(255, 0, 0),
                         cv::FILLED, 8, 0);
            }
          }
        }
      }
    }
    return return_type::success;
  }

  /**
   * @brief Compute the 3D covariance matrix
   *
   * Compute the 3D covariance matrix.
   * Two possible cases:
   *   1. one Azure camera: use the 3D to uncertainty in the view axis, use
   *      the 2D image to uncertainty in the projection plane
   *   2. one RGB camera: calculates a 3D ellipsoid based on the 2D covariance
   *      plus the "reasonable" depth range as a third azis (direction of view)
   *
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type cov3D_compute(bool debug = false) { return return_type::success; }

  /**
   * @brief Consistency check of the 3D skeleton according to human physiology
   *
   * @authors Marco, Matteo
   * @return result status ad defined in return_type
   */
  return_type consistency_check(bool debug = false)
  {
    return return_type::success;
  }

  /*
    ____  _             _             _          __  __
   |  _ \| |_   _  __ _(_)_ __    ___| |_ _   _ / _|/ _|
   | |_) | | | | |/ _` | | '_ \  / __| __| | | | |_| |_
   |  __/| | |_| | (_| | | | | | \__ \ |_| |_| |  _|  _|
   |_|   |_|\__,_|\__, |_|_| |_| |___/\__|\__,_|_| |_|
                  |___/
  */

  /**
   * @brief Set the parameters of the plugin
   *
   * The parameters are stored in the #_params attribute. This method shall be
   * called imediately after the plugin is instantiated
   *
   * @author Paolo
   * @param params
   */
  void set_params(void *params) override
  {
    _params = *(json *)params;
    if (_params.contains("camera_device"))
    {
      _camera_device = _params["camera_device"];
    }

    if (_params.contains("model_file"))
    {
      _model_file = _params["model_file"];
    }
    else
    {
      throw invalid_argument("ERROR: Missing model_file parameter");
    }

    if (_params.contains("resolution_rgb"))
    {
      _resolution_rgb = _params["resolution_rgb"];
    }

    if (_params.contains("fps"))
    {
      _fps = _params["fps"];
    }

    if (_params.contains("dummy"))
    {
      _dummy = _params["dummy"];
    }

    setup_VideoCapture();
    setup_OpenPoseModel();
    setup_Pipeline();
  }

  /**
   * @brief Get the output of the plugin
   *
   * This method acquires a new image and computes the skeleton from it.
   *
   * @author Paolo
   * @param out The output of the plugin as JSON
   * @param blob Possible additional binary data
   * @return return_type
   */
  return_type get_output(json *out,
                         vector<unsigned char> *blob = nullptr) override
  {

    out->clear();
    (*out)["agent_id"] = _agent_id;

    acquire_frame(_dummy);
    skeleton_from_depth_compute(_params["debug"]["skeleton_from_depth_compute"]);
    point_cloud_filter(_params["debug"]["point_cloud_filter"]);
    skeleton_from_rgb_compute(_params["debug"]["skeleton_from_rgb_compute"]);

    if (!(_result))
    {
      return return_type::warning;
    }

    hessian_compute(_params["debug"]["hessian_compute"]);

    if (_params["debug"]["viewer"])
    {

      Mat rgb_flipped;
      flip(_rgb, rgb_flipped, 1);
      imshow("Human Pose Estimation Results", rgb_flipped);

      int max_depth;
#ifdef KINECT_AZURE
      // Configure the colormap range based on the depth mode
      // Values get from: https://docs.microsoft.com/en-us/azure/kinect-dk/hardware-specification

      switch (_device_config.depth_mode)
      {
      case K4A_DEPTH_MODE_NFOV_UNBINNED:
        max_depth = 3860;
        break;
      case K4A_DEPTH_MODE_NFOV_2X2BINNED:
        max_depth = 5460;
        break;
      case K4A_DEPTH_MODE_WFOV_UNBINNED:
        max_depth = 2210;
        break;
      case K4A_DEPTH_MODE_WFOV_2X2BINNED:
        max_depth = 2880;
        break;

      default:
        max_depth = 3860;
        break;
      }
#else
      max_depth = 2000;
#endif
      
      Mat rgbd_flipped;
      flip(_rgbd_filtered, rgbd_flipped, 1);
      rgbd_flipped.convertTo(rgbd_flipped, CV_8U, 255.0 / max_depth);
      Mat rgbd_flipped_color;
      applyColorMap(rgbd_flipped, rgbd_flipped_color, COLORMAP_HSV); // Apply the colormap:
      imshow("rgbd", rgbd_flipped_color);
      
      int key = cv::waitKey(1000.0 / _fps);
      //system("pause");
      if (27 == key || 'q' == key || 'Q' == key)
      { // Esc
#ifdef KINECT_AZURE
        _device.close();
#else
        _cap.release();
#endif
        destroyAllWindows();

        return return_type::error;
      }
    }

    // Prepare output

    if (_poses.size() > 0)
    {
      for (int kp = 0; kp < HPEOpenPose::keypointsNumber; kp++)
      {
        if (_keypoints_list[kp].x < 0 || _keypoints_list[kp].y < 0)
          continue;
        (*out)["poses"][keypoints_map[kp]] = {_keypoints_list[kp].x,
                                              _keypoints_list[kp].y};
        (*out)["cov"][keypoints_map[kp]] = {
            _keypoints_cov[kp].x, _keypoints_cov[kp].y, _keypoints_cov[kp].z};
      }
    }

    // store the output in the out parameter json and the point cloud in the
    // blob parameter
    return return_type::success;
  }

  /**
   * @brief Provide further info to Miroscic agent
   *
   * Provide the Miroscic agent loading this plugin with further info to be
   * printed after initialization
   *
   * @return a map with the information of the plugin
   */
  map<string, string> info() override
  {
    map<string, string> info;
    info["kind"] = kind();
    info["model_file"] = _model_file;
    return info;
  }

  /**
   * @brief The plugin identifier
   *
   * @author Paolo
   * @return a string with plugin kind
   */
  string kind() override { return PLUGIN_NAME; }

protected:
  Mat _rgbd; /**< the last RGBD frame */
  Mat _rgb;  /**< the last RGB frame */
  Mat _rgbd_filtered; /**< the last RGBD frame */
  map<string, vector<unsigned char>>
      _skeleton2D; /**< the skeleton from 2D cameras only*/
  map<string, vector<unsigned char>>
      _skeleton3D;       /**< the skeleton from 3D cameras only*/
  vector<Mat> _heatmaps; /**< the joints heatmaps */
  Mat _point_cloud;      /**< the filtered body point cloud */
  Mat _cov3D;            /**< the 3D covariance matrix */
  Mat _cov3D_adj;        /**< the adjusted 3D covariance matrix */
  json _params;          /**< the parameters of the plugin */

  uint32_t _tsize = 0;              /**< target size*/
  data_t _threshold = 0.1;          /**< probability threshold*/
  string _layout = "";              /**< inputs layouts (NCHW, NHWC)*/
  string _inference_device = "CPU"; /**< computation device*/
  uint32_t _nireq = 0;              /**< number of infer requests*/
  string _nstreams = "";            /**< number of streams*/
  uint32_t _nthreads = 0;           /**< number of CPU threads*/
  uint32_t _frames_processed = 0;
  int64_t _frame_num = 0;

  bool _dummy = false;

  int _camera_device = 0;
  data_t _fps = 25;
  string _resolution_rgb = "";
  int _rgb_height; /**< image size rows */
  int _rgb_width;  /**< image size cols */
  vector<cv::Point2i> _keypoints_list;
  vector<cv::Point3f> _keypoints_cov;
  string _model_file;
  string _agent_id;
  VideoCapture _cap;
  chrono::steady_clock::time_point _start_time;

#ifdef KINECT_AZURE
  int _azure_device = 0; /**< the azure device ID */
  k4a_device_configuration_t _device_config;
  k4a::device _device;
  k4abt::tracker _tracker;
  k4a::capture _k4a_rgbd; /**< the last capture */
  k4a::image _depth_image;
  k4abt::frame _body_frame;
  k4a_transformation_t _pc_transformation; /**< the transformation */

#endif
  ov::Core _core;
  unique_ptr<ResultBase> _result;
  OutputTransform _output_transform;
  unique_ptr<ModelBase> _model;
  AsyncPipeline *_pipeline;
  vector<HumanPose>
      _poses; /**<  contains all the keypoints of all identified people */
};

INSTALL_SOURCE_DRIVER(Skeletonizer3D, json);

/*
Example of JSON parameters:
{
    "device": id or relative_video_uri,
    "_resolution_rgbd": "lxh",
    "_resolution_rgb": "lxh",
    "_fps": 30,
    "extrinsic": [[diagonal],[off_diagonal],[translation]]
}
*/

int main(int argc, char const *argv[])
{

  Skeletonizer3D sk;

  return_type rt = return_type::success;
  try
  {

    // Aprire il file JSON in lettura
    ifstream file("params.json");

    if (!file.is_open())
    {
      cerr << "Errore: Impossibile aprire il file." << endl;
      return 1;
    }

    // Leggere il contenuto del file JSON
    json params;
    file >> params;
    file.close();

    sk.set_params(&params);

    json output = {};

    while ((rt = sk.get_output(&output)) != return_type::error)
    {
      if (rt == return_type::warning)
      {
        cout << endl
             << "*** Warning: no result." << endl;
      } /*else {
        cout << "Output: " << output.dump() << endl;
      }*/
    }
    cout << endl;
  }
  catch (const exception &error)
  {
    cerr << error.what() << endl;
    return 1;
  }

  return 0;
}