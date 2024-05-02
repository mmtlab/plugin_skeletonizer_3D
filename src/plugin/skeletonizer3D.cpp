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
  #include <BodyTrackingHelpers.h>
  #include <Utilities.h>
  #include <Window3dWrapper.h>

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
    if (_device.get_capture(&_k4a_rgbd, std::chrono::milliseconds(K4A_WAIT_INFINITE)))
    {
      if(debug)
        cout << "Capture time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " s" << endl;
    }
    else
      return return_type::error;

    // acquire and store into _rgb (RGB) and _rgbd (RGBD), if available
    k4a::image colorImage = _k4a_rgbd.get_color_image();
    
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
        cv::Mat rgb_flipped;
        cv::flip(_rgb, rgb_flipped, 1);
        imshow("rgb", rgb_flipped);

        int key = cv::waitKey(1000.0 / 25 /2);
        if (27 == key || 'q' == key || 'Q' == key) { // Esc
          return return_type::error; //
        }

      }
    }

    k4a::image depth_image = _k4a_rgbd.get_depth_image();

    // from k4a::image to cv::Mat --> depth image
    if (colorImage != NULL)
    {
      if(debug){
        // you can check the format with this function
        k4a_image_format_t format = depth_image.get_format(); // K4A_IMAGE_FORMAT_COLOR_BGRA32 
        cout << "rgbd format: " << format << endl;
      }

      // get raw buffer
      uint8_t* buffer = depth_image.get_buffer();

      // convert the raw buffer to cv::Mat
      int rows = depth_image.get_height_pixels();
      int cols = depth_image.get_width_pixels();
      _rgbd = cv::Mat(rows , cols, CV_16U, (void*)buffer, cv::Mat::AUTO_STEP);
      
      if(debug){
        cv::Mat rgbd_flipped;
        cv::flip(_rgbd, rgbd_flipped, 1);

        // Configure the colormap range based on the depth mode
        // Values get from: https://docs.microsoft.com/en-us/azure/kinect-dk/hardware-specification
        int max_depth;
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
        rgbd_flipped.convertTo(rgbd_flipped, CV_8U, 255.0/max_depth); // 2000 is the maximum depth value
        cv::Mat rgbd_flipped_color;
        // Apply the colormap:
        cv::applyColorMap(rgbd_flipped, rgbd_flipped_color, cv::COLORMAP_HSV);
        imshow("rgbd", rgbd_flipped_color);

        int key = cv::waitKey(1000.0 / 25 /2);
        if (27 == key || 'q' == key || 'Q' == key) { // Esc
          return return_type::error; //
        }
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

    cout << "Skeleton from depth compute... STARTED" << endl;

    if (!_tracker.enqueue_capture(_k4a_rgbd))
    {
        // It should never hit timeout when K4A_WAIT_INFINITE is set.
        std::cout << "Error! Add capture to tracker process queue timeout!" << std::endl;
        return return_type::error;
    }

    _body_frame = _tracker.pop_result();
    if (_body_frame != nullptr)
    {
      uint32_t num_bodies = _body_frame.get_num_bodies();
      std::cout << num_bodies << " bodies are detected!" << std::endl;

      if(debug){

        cout << "Skeleton from depth compute... DEBUG MODE" << endl;

        // Print the body information
        for (uint32_t i = 0; i < num_bodies; i++)
        {
          k4abt_body_t body = _body_frame.get_body(i);
          print_body_information(body);
        }

        // Visualize the result on the 3D window
        /*int depth_width = _sensor_calibration.depth_camera_calibration.resolution_width;
        int depth_height = _sensor_calibration.depth_camera_calibration.resolution_height;
        
        VisualizeResult(_body_frame, _window3d, depth_width, depth_height);
            
        _body_frame.reset();

        _window3d.SetLayout3d(Visualization::Layout3d::OnlyMainView);
        _window3d.SetJointFrameVisualization(true);
        _window3d.Render();*/

      }
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

/* UTILITY FUNCTIONS ========================================================*/
// Utility functions for the debug of the Azure Kinect Skeletonizer3D plugin
  void print_body_information(k4abt_body_t body)
{
  std::cout << "Body ID: " << body.id << std::endl;
  for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
  {
      k4a_float3_t position = body.skeleton.joints[i].position;
      k4a_quaternion_t orientation = body.skeleton.joints[i].orientation;
      k4abt_joint_confidence_level_t confidence_level = body.skeleton.joints[i].confidence_level;
      printf("Joint[%d]: Position[mm] ( %f, %f, %f ); Orientation ( %f, %f, %f, %f); Confidence Level (%d)  \n",
          i, position.v[0], position.v[1], position.v[2], orientation.v[0], orientation.v[1], orientation.v[2], orientation.v[3], confidence_level);
  }
}

void print_body_index_map_middle_line(k4a::image body_index_map)
{
  uint8_t* body_index_map_buffer = body_index_map.get_buffer();

  // Given body_index_map pixel type should be uint8, the stride_byte should be the same as width
  // TODO: Since there is no API to query the byte-per-pixel information, we have to compare the width and stride to
  // know the information. We should replace this assert with proper byte-per-pixel query once the API is provided by
  // K4A SDK.
  assert(body_index_map.get_stride_bytes() == body_index_map.get_width_pixels());

  int middle_line_num = body_index_map.get_height_pixels() / 2;
  body_index_map_buffer = body_index_map_buffer + middle_line_num * body_index_map.get_width_pixels();

  std::cout << "body_index_map at Line " << middle_line_num << ":" << std::endl;
  for (int i = 0; i < body_index_map.get_width_pixels(); i++)
  {
      std::cout << (int)*body_index_map_buffer << ", ";
      body_index_map_buffer++;
  }
  std::cout << std::endl;
}

void VisualizeResult(k4abt::frame body_frame, Window3dWrapper& window3d, int depth_width, int depth_height) {

    // Obtain original capture that generates the body tracking result
    k4a::capture original_capture = body_frame.get_capture();
    k4a::image depth_image = original_capture.get_depth_image();

    std::vector<Color> point_cloud_colors(depth_width * depth_height, { 1.f, 1.f, 1.f, 1.f });

    // Read body index map and assign colors
    k4a::image body_index_map = body_frame.get_body_index_map();
    const uint8_t* body_index_map_buffer = body_index_map.get_buffer();
    for (int i = 0; i < depth_width * depth_height; i++)
    {
        uint8_t body_index = body_index_map_buffer[i];
        if (body_index != K4ABT_BODY_INDEX_MAP_BACKGROUND)
        {
            uint32_t bodyId = body_frame.get_body_id(body_index);
            point_cloud_colors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
        }
    }
    body_index_map.reset();

    k4a_image_t depth_image_t = depth_image.handle();

    // Visualize point cloud
    window3d.UpdatePointClouds(depth_image_t, point_cloud_colors);

    // Visualize the skeleton data
    window3d.CleanJointsAndBones();
    uint32_t num_bodies = body_frame.get_num_bodies();
    for (uint32_t i = 0; i < num_bodies; i++)
    {
        k4abt_body_t body;
        body.id = body_frame.get_body_id(i);

        // Assign the correct color based on the body id
        Color color = g_bodyColors[body.id % g_bodyColors.size()];
        color.a = 0.4f;
        Color low_confidence_color = color;
        low_confidence_color.a = 0.1f;

        // Visualize joints
        for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++)
        {
            if (body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                const k4a_float3_t& jointPosition = body.skeleton.joints[joint].position;
                const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;

                window3d.AddJoint(
                    jointPosition,
                    jointOrientation,
                    body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : low_confidence_color);
            }
        }

        // Visualize bones
        for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
        {
            k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
            k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

            if (body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
                body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                bool confidentBone = body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
                    body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM;
                const k4a_float3_t& joint1Position = body.skeleton.joints[joint1].position;
                const k4a_float3_t& joint2Position = body.skeleton.joints[joint2].position;

                window3d.AddBone(joint1Position, joint2Position, confidentBone ? color : low_confidence_color);
            }
        }
    }

    original_capture.reset();
    depth_image.reset();

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

      _device_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
      _device_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32; // <==== For Color image
      _device_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
      _device_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED; // <==== For Depth image

      if(_params.contains("device")){
        _device_id = _params["device"];
          cout << "   Camera id: " << _device_id << endl;
      }
      else {
        cout << "   Camera id (default): " << _device_id << endl;
      }

      _device = k4a::device::open(_device_id);
      _device.start_cameras(&_device_config);

      _sensor_calibration = _device.get_calibration(_device_config.depth_mode, _device_config.color_resolution);
      cout << "   Camera calibrated!" << endl;

      _trackerConfig = K4ABT_TRACKER_CONFIG_DEFAULT;
      if(_params.contains("CUDA")){
          cout << "   Body tracker CUDA processor enabled: " << _params["CUDA"] << endl;
          if (_params["CUDA"] == true)
            _trackerConfig.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
      }
      _tracker = k4abt::tracker::create(_sensor_calibration, _trackerConfig);
      cout << "   Body Tracker created!" << endl;

      if(_params.contains("debug")["skel_from_depth_compute"]){
        if (_params["debug"]["skel_from_depth_compute"] == true)
        {
          // Initialize the 3d window controller
          _window3d.Create("3D Visualization", _sensor_calibration);
          cout << "   3D window created!" << endl;
        }
      }

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
    
    acquire_frame(_params["debug"]["acquire_frame"]);
    skeleton_from_depth_compute(_params.at("debug")["skeleton_from_depth_compute"]);
    /*skeleton_from_rgb_compute(_params.at("debug")["skeleton_from_rgb_compute"]);
    hessian_compute(_params.at("debug")["hessian_compute"]);
    cov3D_compute(_params.at("debug")["cov3D_compute"]);
    consistency_check(_params.at("debug")["consistency_check"]);
    point_cloud_filter(_params.at("debug")["point_cloud_filter"]);
    coordinate_transfrom(_params.at("debug")["coordinate_transfrom"]);*/
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
    m["device"] = to_string(_params["device"]);
    m["debug"] = _params["debug"].dump();
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
  k4a::capture _k4a_rgbd; /**< the last capture */
  k4a_device_configuration_t _device_config; /**< the device configuration */
  k4a::calibration _sensor_calibration; /**< the sensor calibration */
  k4a::device _device; /**< the device */
  k4abt_tracker_configuration_t _trackerConfig; /**< the tracker configuration */
  k4abt::tracker _tracker; /**< the body tracker */

  k4abt::frame _body_frame; /**< the body frame */

  // Debug variables
  Window3dWrapper _window3d; /**< the 3D window controller */

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