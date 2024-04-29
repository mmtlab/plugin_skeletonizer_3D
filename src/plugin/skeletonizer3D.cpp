/*
  ____  _        _      _              _             _____ ____
 / ___|| | _____| | ___| |_ ___  _ __ (_)_______ _ _|___ /|  _ \
 \___ \| |/ / _ \ |/ _ \ __/ _ \| '_ \| |_  / _ \ '__||_ \| | | |
  ___) |   <  __/ |  __/ || (_) | | | | |/ /  __/ |  ___) | |_| |
 |____/|_|\_\___|_|\___|\__\___/|_| |_|_/___\___|_| |____/|____/
 */

#include <assert.h>
#include <iostream>
#include <time.h>

#include "../source.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pugg/Kernel.h>

#ifndef PLUGIN_NAME
#define PLUGIN_NAME "skeletonizer3D"
#endif

#define KINECT_AZURE true

#ifdef KINECT_AZURE
  // include Kinect libraries
  #include <k4a/k4a.hpp>
  #include <k4abt.hpp>
#endif

using namespace cv;
using namespace std;
using json = nlohmann::json;

/**
 * @class Skeletonizer3D
 *
 * @brief Skeletonizer3D is a plugin that computes the 3D skeleton of a human
 * body from a depth map.
 *
 */
class Skeletonizer3D : public Source<json> {
public:
  /**
   * @brief Constructor
   *
   */
  Skeletonizer3D() {}

  /**
   * @brief Destructor
   *
   * @author Paolo
   */
  ~Skeletonizer3D() {}

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
  return_type acquire_frame(bool debug = false) {
    // acquire last frame from the camera device
    // if camera device is a Kinect Azure, use the Azure SDK
    // and translate the frame in OpenCV format

    #ifdef KINECT_AZURE


    const clock_t begin_time = clock();
    // acquire and translate into _rgb and _rgbd
    k4a::capture sensor_capture;
    if (_device.get_capture(&sensor_capture, std::chrono::milliseconds(K4A_WAIT_INFINITE)))
    {
      if(debug)
        cout << "Capture time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " s" << endl;
    }
    else
      return return_type::error;

    // acquire and store into _rgb (RGB) and _rgbd (RGBD), if available
    k4a::image colorImage = sensor_capture.get_color_image();
    
    // from k4a::image to cv::Mat --> color image
    if (colorImage != NULL)
    {
      if(debug){
        // you can check the format with this function
        k4a_image_format_t format = colorImage.get_format(); // K4A_IMAGE_FORMAT_COLOR_BGRA32 
        cout << "rgb format: " << format << endl;
      }

      // get raw buffer
      uint8_t* buffer = colorImage.get_buffer();

      // convert the raw buffer to cv::Mat
      int rows = colorImage.get_height_pixels();
      int cols = colorImage.get_width_pixels();
      _rgb = cv::Mat(rows , cols, CV_8UC4, (void*)buffer, cv::Mat::AUTO_STEP);

      if(debug){
        imshow("rgb", _rgb);
        waitKey(0);
      }
    }

    k4a::image depthImage = sensor_capture.get_depth_image();

    // from k4a::image to cv::Mat --> depth image
    if (colorImage != NULL)
    {
      if(debug){
        // you can check the format with this function
        k4a_image_format_t format = depthImage.get_format(); // K4A_IMAGE_FORMAT_COLOR_BGRA32 
        cout << "rgbd format: " << format << endl;
      }

      // get raw buffer
      uint8_t* buffer = depthImage.get_buffer();

      // convert the raw buffer to cv::Mat
      int rows = depthImage.get_height_pixels();
      int cols = depthImage.get_width_pixels();
      _rgbd = cv::Mat(rows , cols, CV_16U, (void*)buffer, cv::Mat::AUTO_STEP);
      
      if(debug){
        imshow("rgbd", _rgbd);
        waitKey(0);
      }
    }
    #endif
    
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
  return_type skeleton_from_depth_compute(bool debug = false) {
#ifdef KINECT_AZURE
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
  return_type point_cloud_filter(bool debug = false) {
#ifdef KINECT_AZURE
    return return_type::success;
#else
    // NOOP
    return return_type::success;
#endif
  }

  /**
   * @brief Transform the 3D skeleton coordinates in the global reference frame
   *
   * Use the extrinsic camera parameters to transorm the 3D skeleton coordinates
   * just before sending them as plugin output.
   *
   * @return return_type
   */
  return_type coordinate_transfrom(bool debug = false) {
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
  return_type skeleton_from_rgb_compute(bool debug = false) {
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
  return_type hessian_compute(bool debug = false) {
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
  return_type consistency_check(bool debug = false) {
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
  void set_params(void *params) override { 
    _params = *(json *)params;

    #ifdef KINECT_AZURE
      cout << "Setting Azure Kinect parameters..." << endl;

      k4a_device_configuration_t device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
      device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32; // <==== For Color image
      device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
      device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // <==== For Depth image

      if(_params.contains("device")){
        _device_id = _params["device"];
          cout << "   Camera id: " << _device_id << endl;
      }
      else {
        cout << "   Camera id (default): " << _device_id << endl;
      }

      _device = k4a::device::open(_device_id);
      _device.start_cameras(&device_config);

      k4a::calibration sensor_calibration = _device.get_calibration(device_config.depth_mode, device_config.color_resolution);
      cout << "   Camera calibrated!" << endl;

      k4abt_tracker_configuration_t trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
      if(_params.contains("CUDA")){
          cout << "   Body tracker CUDA processor enabled: " << _params["CUDA"] << endl;
          if (_params["CUDA"] == true)
            trackerConfig.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
      }
      _tracker = k4abt::tracker::create(sensor_calibration, trackerConfig);
      cout << "   Body Tracker created!" << endl;

    #endif
    cout << "Azure Kinect parameters set!" << endl;
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
  return_type get_output(json *out, vector<unsigned char> *blob) override {
    // call in sequence the methods to compute the skeleton (acquire_frame,
    // skeleton_from_depth_compute, etc.)

    acquire_frame(out->at("debug")["acquire_frame"]);
    /*skeleton_from_depth_compute(out->at("debug")["skeleton_from_depth_compute"]);
    skeleton_from_rgb_compute(out->at("debug")["skeleton_from_rgb_compute"]);
    hessian_compute(out->at("debug")["hessian_compute"]);
    cov3D_compute(out->at("debug")["cov3D_compute"]);
    consistency_check(out->at("debug")["consistency_check"]);
    point_cloud_filter(out->at("debug")["point_cloud_filter"]);
    coordinate_transfrom(out->at("debug")["coordinate_transfrom"]);*/
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
  map<string, string> info() override {
    map<string, string> m{};
    m["device"] = to_string(_device_id);
    return m;
  }

  /**
   * @brief The plugin identifier
   *
   * @author Paolo
   * @return a string with plugin kind
   */
  string kind() override { return PLUGIN_NAME; }

protected:
  int _device_id = 0; /**< the device ID */
  cv::Mat _rgbd; /**< the last RGBD frame */
  cv::Mat _rgb;  /**< the last RGB frame */
  map<string, vector<unsigned char>>
      _skeleton2D; /**< the skeleton from 2D cameras only*/
  map<string, vector<unsigned char>>
      _skeleton3D;       /**< the skeleton from 3D cameras only*/
  vector<Mat> _heatmaps; /**< the joints heatmaps */
  Mat _point_cloud;      /**< the filtered body point cloud */
  Mat _cov2D;            /**< the 2D covariance matrix (18x3)*/
  Mat _cov3D;            /**< the 3D covariance matrix */
  Mat _cov3D_adj;        /**< the adjusted 3D covariance matrix */
  json _params;          /**< the parameters of the plugin */
#ifdef KINECT_AZURE
  k4a_capture_t _k4a_rgbd; /**< the last capture */
  k4a::device _device;
  k4abt::tracker _tracker;
#endif
};

INSTALL_SOURCE_DRIVER(Skeletonizer3D, json);

/*
Example of JSON parameters:
{
    "device": id or relative_video_uri,
    "resolution_rgbd": "lxh",
    "resolution_rgb": "lxh",
    "fps": 30,
    "extrinsic": [[diagonal],[off_diagonal],[translation]]
}
*/

/* Execute only if as standalone program. othrwise, the main is 
  in the plugiung loader  "load_source.cpp"
*/
int main(int argc, char const *argv[]) {
  try {

    cout << "Skeletonizer3D Started!" << endl;

    Skeletonizer3D sk;

    json output;
    vector<unsigned char> blob;

    json params = {
      {"device", 0},
      {"out_res", "800x450"}
      };
    sk.set_params(&params);

    cout << "Params: " << endl;
    for (auto &[k, v] : sk.info()) {
      cout << k << ": " << v << endl;
    }

    sk.get_output(&output, &blob);

  } catch (const std::exception &error) {
    cerr << error.what() << endl;
    return 1;
  }

  return 0;
}