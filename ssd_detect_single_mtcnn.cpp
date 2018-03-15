// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>


// Fuck!

#define CPU_ONLY

#ifndef MTCNN_MTCNN_H
#define MTCNN_MTCNN_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;

class MTCNN {

public:

    MTCNN();
    MTCNN(const std::vector<std::string> model_file, const std::vector<std::string> trained_file);
    ~MTCNN();

    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles);
    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence);
    void detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment);
    void detection_TEST(const cv::Mat& img, std::vector<cv::Rect>& rectangles);

    void Preprocess(const cv::Mat &img);
    void P_Net();
    void R_Net();
    void O_Net();
    void detect_net(int i);

    void local_NMS();
    void global_NMS();

    void Predict(const cv::Mat& img, int i);
    void Predict(const std::vector<cv::Mat> imgs, int i);
    void WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i);
    void WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i);

    float IoU(cv::Rect rect1, cv::Rect rect2);
    float IoM(cv::Rect rect1, cv::Rect rect2);
    void resize_img();
    void GenerateBoxs(cv::Mat img);
    void BoxRegress(std::vector<cv::Rect>& bounding_box, std::vector<cv::Rect> regression_box);
    void Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h);
    cv::Mat crop(cv::Mat img, cv::Rect& rect);

    void img_show(cv::Mat img, std::string name);
    void img_show_T(cv::Mat img, std::string name);
    //param for P, R, O, L net
    std::vector<std::shared_ptr<Net<float>>> nets_;
    std::vector<cv::Size> input_geometry_;
    int num_channels_;

    //variable for the image
    cv::Mat img_;
    std::vector<cv::Mat> img_resized_;
    std::vector<double> scale_;

    //variable for the output of the neural network
//    std::vector<cv::Rect> regression_box_;
    std::vector<float> regression_box_temp_;
    std::vector<cv::Rect> bounding_box_;
    std::vector<float> confidence_;
    std::vector<float> confidence_temp_;
    std::vector<std::vector<cv::Point>> alignment_;
    std::vector<float> alignment_temp_;

    //paramter for the threshold
    int minSize_ = 200;
    float factor_ = 0.709;
    float threshold_[3] = {0.5, 0.5, 0.3};
    float threshold_NMS_ = 0.5;
};


#endif //MTCNN_MTCNN_H


// END_FUCK



#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
//#ifdef CPU_ONLY
//  Caffe::set_mode(Caffe::CPU);
//#else
//  Caffe::set_mode(Caffe::GPU);
//#endif

  Caffe::set_mode(Caffe::CPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  std::cout << "1" << std::endl;
  Preprocess(img, &input_channels);
  std::cout << "2" << std::endl;
  net_->Forward();
  std::cout << "3" << std::endl;

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  std::cout << "4" << std::endl;
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);
  sample_normalized *= 0.007843f;

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "127,127,127",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.5,
    "Only store detections with score higher than the threshold.");




const std::string object_classes[] = {
"background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"
};





#define _GNU_SOURCE
#include <pthread.h>
#include <mutex>
std::vector<std::vector<float> > detections;
pthread_mutex_t detections_mutex;
cv::Mat current_img;
pthread_mutex_t img_mutex;


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // IGOR
  const string& model_file = "/home/accendo/data/models/MobileNetSSD/MobileNetSSD_deploy.prototxt";
  const string& weights_file = "/home/accendo/data/models/MobileNetSSD/MobileNetSSD_deploy.caffemodel";
  // Tut toje
  vector<string> face_model_file = {
          "/home/accendo/data/models/MTCNN/det1.prototxt",
          "/home/accendo/data/models/MTCNN/det2.prototxt",
          "/home/accendo/data/models/MTCNN/det3.prototxt"
  };
  // I tut
  vector<string> face_trained_file = {
          "/home/accendo/data/models/MTCNN/det1.caffemodel",
          "/home/accendo/data/models/MTCNN/det2.caffemodel",
          "/home/accendo/data/models/MTCNN/det3.caffemodel"
  };

  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the MTCNN networks.
  MTCNN mtcnn(face_model_file, face_trained_file);


  cv::VideoCapture cap(0);
  cv::Mat img;

  // Creating the threads
  pthread_t main_thread = pthread_self();
  int s = 0;
  if (pthread_mutex_init(&img_mutex, NULL) != 0)
  {
    printf("\n img_mutex init failed\n");
    return 1;
  }

  // Forcing affinity to CPU0 for main thread (3 small neural nets)
  cpu_set_t main_cpuset;
  CPU_ZERO(&main_cpuset);
  CPU_SET(0, &main_cpuset);
  CPU_SET(1, &main_cpuset);
  CPU_SET(2, &main_cpuset);
  CPU_SET(3, &main_cpuset);
  s = pthread_setaffinity_np(main_thread, sizeof(cpu_set_t), &main_cpuset);
  if (s != 0) {
    fprintf(stderr, "Error forcing affinity for main\n");
    return 1;
  }

  int frame_count = 0;
  while (cap.read(img)) {

        CHECK(!img.empty()) << "Error when read frame";
        // cv::copyTo(img, current_img);
        pthread_mutex_lock(&img_mutex);
        img.copyTo(current_img);
        pthread_mutex_unlock(&img_mutex);

        /* Print the SSD detection results. */
        pthread_mutex_lock(&detections_mutex);
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= confidence_threshold) {
            int class_id = static_cast<int>(d[1]);
            int xmin = static_cast<int>(d[3] * img.cols);
            int ymin = static_cast<int>(d[4] * img.rows);
            int xmax = static_cast<int>(d[5] * img.cols);
            int ymax = static_cast<int>(d[6] * img.rows);

            std::string desc = object_classes[class_id] + " " + std::to_string(int(score * 100)) + "%";
            cv::putText(img, desc, cvPoint(xmin - 5, ymin - 8),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
            cv::Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
            cv::rectangle(img, r, cv::Scalar(255, 0, 0), 3);
          }
        }
        pthread_mutex_unlock(&detections_mutex);


        vector<cv::Rect> rectangles;
        vector<float> confidences;
        std::vector<std::vector<cv::Point>> alignment;
        mtcnn.detection(img, rectangles, confidences, alignment);

        /* Print the MTCNN detections on top of it */
        for(int i = 0; i < rectangles.size(); i++)
        {
            int green = confidences[i] * 255;
            int red = (1 - confidences[i]) * 255;
            cv::rectangle(img, rectangles[i], cv::Scalar(0, green, red), 3);
            for(int j = 0; j < alignment[i].size(); j++)
            {
                cv::circle(img, alignment[i][j], 5, cv::Scalar(255, 255, 0), 3);
            }
        }

        ++frame_count;
		cv::putText(img, std::to_string(frame_count), cv::Point(3, 13),
			cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
        cv::imshow("Live", img);
        cv::waitKey(1);
  }
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV

// START_FUCK

MTCNN::MTCNN(){}

MTCNN::MTCNN(const std::vector<std::string> model_file, const std::vector<std::string> trained_file)
{
    //#ifdef CPU_ONLY
    //    Caffe::set_mode(Caffe::CPU);
    //#else
    //    Caffe::set_mode(Caffe::GPU);
    //#endif
    Caffe::set_mode(Caffe::CPU);

    for(int i = 0; i < model_file.size(); i++)
    {
        std::shared_ptr<Net<float>> net;

        cv::Size input_geometry;
        int num_channel;

        net.reset(new Net<float>(model_file[i], TEST));
        net->CopyTrainedLayersFrom(trained_file[i]);

        Blob<float>* input_layer = net->input_blobs()[0];
        num_channel = input_layer->channels();
        input_geometry = cv::Size(input_layer->width(), input_layer->height());

        nets_.push_back(net);
        input_geometry_.push_back(input_geometry);
        if(i == 0)
            num_channels_ = num_channel;
        else if(num_channels_ != num_channel)
            std::cout << "Error: The number channels of the nets are different!" << std::endl;
    }
}

MTCNN::~MTCNN(){}

void MTCNN::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles)
{
    Preprocess(img);
    P_Net();
    local_NMS();
    R_Net();
    local_NMS();
    O_Net();
    global_NMS();


    rectangles.clear();
    for(auto &bounding_box : bounding_box_)
    {
        rectangles.push_back(cv::Rect(bounding_box.y, bounding_box.x, bounding_box.height, bounding_box.width));
    }
}

void MTCNN::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence)
{
    detection(img, rectangles);

    confidence = confidence_;
}

void MTCNN::detection(const cv::Mat& img, std::vector<cv::Rect>& rectangles, std::vector<float>& confidence, std::vector<std::vector<cv::Point>>& alignment)
{
    detection(img, rectangles, confidence);

    alignment.clear();
    for(auto &i : alignment_)
    {
        std::vector<cv::Point> temp_alignment;
        for(auto &j : i)
        {
            temp_alignment.push_back(cv::Point(j.y, j.x));
        }
        alignment.push_back(std::move(temp_alignment));
    }

}

void MTCNN::detection_TEST(const cv::Mat& img, std::vector<cv::Rect>& rectangles)
{
    Preprocess(img);
    P_Net();
    img_show_T(img, "P-Net");
    local_NMS();
    img_show_T(img, "P-Net_nms");
    R_Net();
    img_show_T(img, "R-Net");
    local_NMS();
    img_show_T(img, "R-Net_nms");
    O_Net();
    img_show_T(img, "O-Net");
    global_NMS();
    img_show_T(img, "O-Net_nms");

}

void MTCNN::Preprocess(const cv::Mat &img)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample.convertTo(sample_float, CV_32FC3);
    else
        sample.convertTo(sample_float, CV_32FC1);


    cv::cvtColor(sample_float,sample_float,cv::COLOR_BGR2RGB);
    sample_float = sample_float.t();

    img_ = sample_float;
}

void MTCNN::P_Net()
{
    resize_img();

    for(auto img_resized : img_resized_){
        Predict(img_resized, 0);
        GenerateBoxs(img_resized);
    }
}

void MTCNN::R_Net()
{
    detect_net(1);
}

void MTCNN::O_Net()
{
    detect_net(2);
}

void MTCNN::detect_net(int i)
{
    float thresh = threshold_[i];
    std::vector<cv::Rect> bounding_box;
    std::vector<float> confidence;
    std::vector<cv::Mat> cur_imgs;
    std::vector<std::vector<cv::Point>> alignment;

    if(bounding_box_.size() == 0)
        return;

    for (int j = 0; j < bounding_box_.size(); j++) {
        cv::Mat img = crop(img_, bounding_box_[j]);
        if (img.size() == cv::Size(0,0))
            continue;
        if (img.rows == 0 || img.cols == 0)
            continue;
        if (img.size() != input_geometry_[i])
            cv::resize(img, img, input_geometry_[i]);
        img.convertTo(img, CV_32FC3, 0.0078125,-127.5*0.0078125);
        cur_imgs.push_back(img);
    }

//    std::vector<cv::Mat> cur_imgs_test;
//    cur_imgs_test.push_back(cur_imgs[0]);

    Predict(cur_imgs, i);

    for(int j = 0; j < confidence_temp_.size()/2; j++)
    {
        float conf = confidence_temp_[2*j+1];
        if (conf > thresh) {

            if(conf>1)
                int a = 0;

            //bounding box
            cv::Rect bbox;

            //regression box : y x height width
            bbox.y = bounding_box_[j].y + regression_box_temp_[4*j] * bounding_box_[j].height;
            bbox.x = bounding_box_[j].x + regression_box_temp_[4*j+1] * bounding_box_[j].width ;
            bbox.height = bounding_box_[j].height + regression_box_temp_[4*j+2] * bounding_box_[j].height;
            bbox.width = bounding_box_[j].width + regression_box_temp_[4*j+3] * bounding_box_[j].width;

//            bbox.y = bounding_box_[j].y + regression_box_temp_[4*j] * bounding_box_[j].height - regression_box_temp_[4*j+2] * bounding_box_[j].height *  0.5;
//            bbox.x = bounding_box_[j].x + regression_box_temp_[4*j+1] * bounding_box_[j].width - regression_box_temp_[4*j+3] * bounding_box_[j].width * 0.5;
//            bbox.height = bounding_box_[j].height + regression_box_temp_[4*j+2] * bounding_box_[j].height;
//            bbox.width = bounding_box_[j].width + regression_box_temp_[4*j+3] * bounding_box_[j].width;

            if(i == 2)
            {
                //face alignment
                std::vector<cv::Point> align(5);
                for(int k = 0; k < 5; k++)
                {
//                    align[k].x = bbox.x + bbox.width * alignment_temp_[10*j+5+k] - 1;
//                    align[k].y = bbox.y + bbox.height * alignment_temp_[10*j+k] - 1;

                    align[k].x = bounding_box_[j].x + bounding_box_[j].width * alignment_temp_[10*j+5+k] - 1;
                    align[k].y = bounding_box_[j].y + bounding_box_[j].height * alignment_temp_[10*j+k] - 1;
                }
                alignment.push_back(align);
            }

            confidence.push_back(conf);
            bounding_box.push_back(bbox);

        }
    }

    cur_imgs.clear();

    bounding_box_ = bounding_box;
    confidence_ = confidence;
    alignment_ = alignment;
}


void MTCNN::local_NMS()
{
    std::vector<cv::Rect> cur_rects = bounding_box_;
    std::vector<float> confidence = confidence_;
    float threshold = threshold_NMS_;

    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            if(IoU(cur_rects[i], cur_rects[j]) > threshold)
            {
                float a = IoU(cur_rects[i], cur_rects[j]);
//                if(confidence[i] == confidence[j])
//                {
//                    cur_rects.erase(cur_rects.begin() + j);
//                    confidence.erase(confidence.begin() + j);
//                }
                if(confidence[i] >= confidence[j] && confidence[j] < 0.96)
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else if (confidence[i] < confidence[j] && confidence[i] < 0.96)
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
                else
                {
                    j++;
                }
            }
            else
            {
                j++;
            }

        }
    }

    bounding_box_ = cur_rects;
    confidence_ = confidence;

}

void MTCNN::global_NMS()
{
    std::vector<cv::Rect> cur_rects = bounding_box_;
    std::vector<float> confidence = confidence_;
    std::vector<std::vector<cv::Point>> alignment = alignment_;
    float threshold_IoM = threshold_NMS_;
    float threshold_IoU = threshold_NMS_ - 0.1;


    for(int i = 0; i < cur_rects.size(); i++)
    {
        for(int j = i + 1; j < cur_rects.size(); )
        {
            if(IoU(cur_rects[i], cur_rects[j]) > threshold_IoU || IoM(cur_rects[i], cur_rects[j]) > threshold_IoM)
            {
                if(confidence[i] >= confidence[j])// && confidence[j] < 0.85) //if confidence[i] == confidence[j], it keeps the small one
                {
                    cur_rects.erase(cur_rects.begin() + j);
                    alignment.erase(alignment.begin() + j);
                    confidence.erase(confidence.begin() + j);
                }
                else if(confidence[i] < confidence[j])// && confidence[i] < 0.85)
                {
                    cur_rects.erase(cur_rects.begin() + i);
                    alignment.erase(alignment.begin() + i);
                    confidence.erase(confidence.begin() + i);
                    i--;
                    break;
                }
                else
                {
                    j++;
                }
            }
            else
            {
                j++;
            }
        }
    }

    bounding_box_ = cur_rects;
    confidence_ = confidence;
    alignment_ = alignment;
}


/*
 * Predict function input is a image without crop
 * the reshape of input layer is image's height and width
 */
void MTCNN::Predict(const cv::Mat& img, int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         img.rows, img.cols);
    /* Forward dimension change to all layers. */
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(img, &input_channels, i);
    net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* rect = net->output_blobs()[0];
    Blob<float>* confidence = net->output_blobs()[1];
    int count = confidence->count() / 2;

    const float* rect_begin = rect->cpu_data();
    const float* rect_end = rect_begin + rect->channels() * count;
    regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

    const float* confidence_begin = confidence->cpu_data() + count;
    const float* confidence_end = confidence_begin + count;

    confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
}

/*
 * Predict(const std::vector<cv::Mat> imgs, int i) function
 * used to input is a group of image with crop from original image
 * the reshape of input layer of net is the number, channels, height and width of images.
 */
void MTCNN::Predict(const std::vector<cv::Mat> imgs, int i)
{
    std::shared_ptr<Net<float>> net = nets_[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    input_layer->Reshape(imgs.size(), num_channels_,
                         input_geometry_[i].height, input_geometry_[i].width);
    int num = input_layer->num();
    /* Forward dimension change to all layers. */
    net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(imgs, &input_channels, i);

    net->Forward();

    /* Copy the output layer to a std::vector */
    //You can also try to use the blob_by_name()

    //confidence
    Blob<float>* confidence = net->output_blobs()[i];
    int count = confidence->count() / 2; //the channel of confidence is two
    const float* confidence_begin = confidence->cpu_data();
    const float* confidence_end = confidence_begin + count * 2;
    confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);

    //regression_box
    Blob<float>* rect = net->output_blobs()[0];
    const float* rect_begin = rect->cpu_data();
    const float* rect_end = rect_begin + rect->channels() * count;
    regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

    //landmarks
    if( i == 2){
        Blob<float>* points = net->output_blobs()[1];
        const float* points_begin = points->cpu_data();
        const float* points_end = points_begin + points->channels() * count;
        alignment_temp_ = std::vector<float>(points_begin, points_end);
    }
}

void MTCNN::WrapInputLayer(const cv::Mat& img, std::vector<cv::Mat> *input_channels, int i)
{
    Blob<float>* input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }

    //cv::Mat sample_normalized;
    //cv::subtract(img, mean_[i], img);
    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(img, *input_channels);

}

/*
 * WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i) function
 * used to write the separate BGR planes directly to the input layer of the network
 */
void MTCNN::WrapInputLayer(const vector<cv::Mat> imgs, std::vector<cv::Mat> *input_channels, int i)
{
    Blob<float> *input_layer = nets_[i]->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int num = input_layer->num();
    float *input_data = input_layer->mutable_cpu_data();

    for (int j = 0; j < num; j++) {
        //std::vector<cv::Mat> *input_channels;
        for (int k = 0; k < input_layer->channels(); ++k) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
        cv::Mat img = imgs[j];
        cv::split(img, *input_channels);
        input_channels->clear();
    }
}

float MTCNN::IoU(cv::Rect rect1, cv::Rect rect2)
{
    int x_overlap, y_overlap, intersection, unions;
    x_overlap = std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) - std::max(rect1.x, rect2.x));
    y_overlap = std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) - std::max(rect1.y, rect2.y));
    intersection = x_overlap * y_overlap;
    unions = rect1.width * rect1.height + rect2.width * rect2.height - intersection;
    return float(intersection)/unions;
}

float MTCNN::IoM(cv::Rect rect1, cv::Rect rect2)
{
    int x_overlap, y_overlap, intersection, min_area;
    x_overlap = std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) - std::max(rect1.x, rect2.x));
    y_overlap = std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) - std::max(rect1.y, rect2.y));
    intersection = x_overlap * y_overlap;
    min_area = std::min((rect1.width * rect1.height), (rect2.width * rect2.height));
    return float(intersection)/min_area;
}

void MTCNN::resize_img()
{
    cv::Mat img = img_;
    int height = img.rows;
    int width = img.cols;

    int minSize = minSize_;
    float factor = factor_;
    double scale = 24./minSize;
    int minWH = std::min(height, width) * scale;

    std::vector<cv::Mat> img_resized;

    while(minWH >= 12)
    {
        int resized_h = std::ceil(height*scale);
        int resized_w = std::ceil(width*scale);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_AREA);
        resized.convertTo(resized, CV_32FC3, 0.0078125,-127.5*0.0078125);
        img_resized.push_back(resized);

        minWH *= factor;
        scale *= factor;
    }

    img_resized_ = img_resized;
}

void MTCNN::GenerateBoxs(cv::Mat img)
{
    int stride = 2;
    int cellSize = input_geometry_[0].width;
    int image_h = img.rows;
    int image_w = img.cols;
    double scale = double(image_w) / img_.cols ;
    int feature_map_h = std::ceil((image_h - cellSize)*1.0/stride)+1;
    int feature_map_w = std::ceil((image_w - cellSize)*1.0/stride)+1;
    int width = (cellSize) / scale;
    int count = confidence_temp_.size();
    float thresh = threshold_[0];

    std::vector<cv::Rect> bounding_box;
    std::vector<cv::Rect> regression_box;
//    cv::Rect regression_box;
    std::vector<float> confidence;

    for(int i = 0; i < count; i++)
    {
        if(confidence_temp_[i] < thresh)
            continue;

        confidence.push_back(confidence_temp_[i]);

        int y = i / feature_map_w;
        int x = i - feature_map_w * y;

        //the regression box from the neural network
        //regression box : y x height width
        regression_box.push_back(cv::Rect(regression_box_temp_[i + count] * width, regression_box_temp_[i] * width,
                                          regression_box_temp_[i + count*3] * width, regression_box_temp_[i + count*2] * width));
//        regression_box = cv::Rect(regression_box_temp_[i] * width, regression_box_temp_[i + count] * width,
//                                          regression_box_temp_[i + count*2] * width, regression_box_temp_[i + count*3] * width));
        //the bounding box combined with regression box
        bounding_box.push_back(cv::Rect((x*stride+1)/scale, (y*stride+1)/scale,
                                        width, width));

        if((x*stride+1)/scale < -1000 || (x*stride+1)/scale > 1000)
            int a = 0;

//        bounding_box.push_back(cv::Rect((x*stride+1)/scale + regression_box.x, (y*stride+1)/scale + regression_box.y,
//                                        width + regression_box.width, width + regression_box.height));

    }

    confidence_.insert(confidence_.end(), confidence.begin(), confidence.end());
    BoxRegress(bounding_box, regression_box);
    bounding_box_.insert(bounding_box_.end(), bounding_box.begin(), bounding_box.end());
//    regression_box_.insert(regression_box_.end(), regression_box.begin(), regression_box.end());
}

void MTCNN::BoxRegress(std::vector<cv::Rect>& bounding_box, std::vector<cv::Rect> regression_box)
{

    for(int i=0;i<bounding_box.size();i++)
    {
        bounding_box[i].x += regression_box[i].x;
        bounding_box[i].y += regression_box[i].y;
        bounding_box[i].width += regression_box[i].width;
        bounding_box[i].height += regression_box[i].height;
//        float width = regression_box[i].width;
//        float height = regression_box[i].height;
//        float side = height>width ? height:width;
//        bounding_box[i].x -= side*0.5;
//        bounding_box[i].y -= width-side*0.5;
//        bounding_box[i].width += side*0.5;
//        bounding_box[i].height -= width-side*0.5;

    }
}

void MTCNN::Padding(std::vector<cv::Rect>& bounding_box, int img_w,int img_h)
{
    for(int i=0;i<bounding_box.size();i++)
    {
        bounding_box[i].x = (bounding_box[i].x > 0)? bounding_box[i].x : 0;
        bounding_box[i].y = (bounding_box[i].y > 0)? bounding_box[i].y : 0;
        bounding_box[i].width = (bounding_box[i].x + bounding_box[i].width < img_w) ? bounding_box[i].width : img_w;
        bounding_box[i].height = (bounding_box[i].y + bounding_box[i].height < img_h) ? bounding_box[i].height : img_h;
    }
}

cv::Mat MTCNN::crop(cv::Mat img, cv::Rect& rect)
{
    cv::Rect rect_old = rect;

//    if(rect.width > rect.height)
//    {
//        int change_to_square = rect.width - rect.height;
//        rect.height += change_to_square;
//        rect.y -= change_to_square * 0.5;
//    }
//    else
//    {
//        int change_to_square = rect.height - rect.width;
//        rect.width += change_to_square;
//        rect.x -= change_to_square * 0.5;
//    }

    cv::Rect padding;

    if(rect.x < 0)
    {
        padding.x = -rect.x;
        rect.x = 0;
    }
    if(rect.y < 0)
    {
        padding.y = -rect.y;
        rect.y = 0;
    }
    if(img.cols < (rect.x + rect.width))
    {
        padding.width = rect.x + rect.width - img.cols;
        rect.width = img.cols-rect.x;
    }
    if(img.rows < (rect.y + rect.height))
    {
        padding.height = rect.y + rect.height - img.rows;
        rect.height = img.rows - rect.y;
    }
    if(rect.width<0 || rect.height<0)
    {
        rect = cv::Rect(0,0,0,0);
        padding = cv::Rect(0,0,0,0);
    }
    cv::Mat img_cropped = img(rect);
    if(padding.x||padding.y||padding.width||padding.height)
    {
        cv::copyMakeBorder(img_cropped, img_cropped, padding.y, padding.height, padding.x, padding.width,cv::BORDER_CONSTANT,cv::Scalar(0));
        //here, the rect should be changed
        rect.x -= padding.x;
        rect.y -= padding.y;
        rect.width += padding.width + padding.x;
        rect.width += padding.height + padding.y;
    }

//    cv::imshow("crop", img_cropped);
//    cv::waitKey(0);

    return img_cropped;
}

void MTCNN::img_show(cv::Mat img, std::string name)
{
    cv::Mat img_show;
    img.copyTo(img_show);

    //cv::imwrite("../result/" + name + "test.jpg", img);

    for(int i = 0; i < bounding_box_.size(); i++)
    {
        rectangle(img_show, bounding_box_[i], cv::Scalar(0, 0, 255));
        cv::putText(img_show, std::to_string(confidence_[i]), cvPoint(bounding_box_[i].x + 3, bounding_box_[i].y + 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    }

    for(int i = 0; i < alignment_.size(); i++)
    {
        for(int j = 0; j < alignment_[i].size(); j++)
        {
            cv::circle(img_show, alignment_[i][j], 5, cv::Scalar(0, 255, 0));
        }
    }

    cv::imwrite("../result/" + name + ".jpg", img_show);
    //cv::waitKey(0);
}

void MTCNN::img_show_T(cv::Mat img, std::string name)
{
    cv::Mat img_show;
    img.copyTo(img_show);

    //cv::imwrite("../result/" + name + "test.jpg", img);

    for(int i = 0; i < bounding_box_.size(); i++)
    {
        rectangle(img_show, cv::Rect(bounding_box_[i].y, bounding_box_[i].x, bounding_box_[i].height, bounding_box_[i].width), cv::Scalar(0, 0, 255), 3);
        cv::putText(img_show, std::to_string(confidence_[i]), cvPoint(bounding_box_[i].y + 3, bounding_box_[i].x + 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    }

    for(int i = 0; i < alignment_.size(); i++)
    {
        for(int j = 0; j < alignment_[i].size(); j++)
        {
            cv::circle(img_show, cv::Point(alignment_[i][j].y, alignment_[i][j].x), 5, cv::Scalar(255, 255, 0), 3);
        }
    }

    cv::imwrite("../result/" + name + ".jpg", img_show);
    //cv::waitKey(0);
}
// END_FUCK
