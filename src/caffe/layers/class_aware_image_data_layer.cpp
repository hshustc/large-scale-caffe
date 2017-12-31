#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/class_aware_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ClassAwareImageDataLayer<Dtype>::~ClassAwareImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ClassAwareImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  const int num_classes = this->layer_param_.class_aware_image_data_param().num_classes();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  int cls_nimgs[num_classes] = {0};
  int cls_base_offset[num_classes] = {0};
  lines_id_ = 0;
  //assume that the images of the same classes are listed together in the list
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));

    if(cls_nimgs[label] == 0){
        cls_base_offset[label] = lines_id_;
    }
    cls_nimgs[label] = cls_nimgs[label] + 1;
    lines_id_ = lines_id_ + 1;
  }

  for(int idx = 0; idx < num_classes;idx++){
      cls_list_.push_back(idx);
      cls_nimgs_.push_back(cls_nimgs[idx]);
      cls_base_offset_.push_back(cls_base_offset[idx]);
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleClasses();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << num_classes << " classes.";
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  cls_list_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ClassAwareImageDataLayer<Dtype>::ShuffleClasses() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(cls_list_.begin(), cls_list_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ClassAwareImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  ClassAwareImageDataParameter class_aware_image_data_param = this->layer_param_.class_aware_image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  const int num_classes = class_aware_image_data_param.num_classes();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  int cls_id = 0;
  int cls_base_offset = 0;
  int cls_nimgs = 0;
  Dtype cls_offset = 0;
  Dtype eps = 1e-3;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(num_classes, cls_list_id_);
    
    cls_id = cls_list_[cls_list_id_];
    cls_base_offset = cls_base_offset_[cls_id];
    cls_nimgs = cls_nimgs_[cls_id];
    caffe_rng_uniform(1, Dtype(0), Dtype(cls_nimgs - eps), &cls_offset);

    lines_id_ = cls_base_offset + floor(cls_offset); 
    CHECK_GT(lines_size, lines_id_);
    CHECK_EQ(lines_[lines_id_].second, cls_id)<<" cls_id: "<<cls_id<<" cls_nimgs: "<<cls_nimgs
        <<" cls_base_offset: "<<cls_base_offset<<" cls_offset: "<<cls_offset
        <<" line: "<<lines_[lines_id_].first<<"\t"<<lines_[lines_id_].second;
    DLOG(INFO)<<cls_list_id_<<"\t"<<cls_id<<"\t"<<cls_nimgs<<"\t"<<floor(cls_offset)<<"\t"
        <<lines_[lines_id_].first<<"\t"<<lines_[lines_id_].second;

    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    // lines_id_++;
    // go to the next class
    cls_list_id_++;
    lines_id_ = 0;
    if (cls_list_id_ >= num_classes) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      cls_list_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleClasses();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ClassAwareImageDataLayer);
REGISTER_LAYER_CLASS(ClassAwareImageData);

}  // namespace caffe
#endif  // USE_OPENCV
