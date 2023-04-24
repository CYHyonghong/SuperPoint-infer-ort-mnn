#include "infer_superpoint.hpp"
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <chrono>

using namespace std;


void normalize_keypoints(const vector<vector<int>>& keypoints, vector<vector<double>>& keypoints_norm, int h, int w, int s) {
    for (auto& keypoint : keypoints) {
        vector<double> kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}

int clip(int val, int max) {
    if (val < 0) return 0;
    return min(val, max - 1);
}

void grid_sample(const float* input, vector<vector<double>> &grid, vector<vector<double>>& output, int dim, int h, int w) {
    // descriptors 1, 256, image_height/8, image_width/8
    // keypoints 1, 1, number, 2
    // out 1, 256, 1, number
    for (auto& g : grid) {
        double ix = ((g[0] + 1) / 2) * (w - 1);
        double iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        double nw = (ix_se - ix) * (iy_se - iy);
        double ne = (ix - ix_sw) * (iy_sw - iy);
        double sw = (ix_ne - ix) * (iy - iy_ne);
        double se = (ix - ix_nw) * (iy - iy_nw);

        vector<double> descriptor;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 dhw
            // x * height * depth + y * depth + z
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }
}

template<typename Iter_T>
double vector_normalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}

void normalize_descriptors(vector<vector<double>>& dest_descriptors) {
    for (auto& descriptor : dest_descriptors) {
        double norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
        std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                       std::bind1st(std::multiplies<double>(), norm_inv));
    }
}

cv::Mat vector2mat(vector<float> output, cv::Size size){
    cv::Mat out_result(size.height, size.width, CV_32FC1, cv::Scalar(0));
    memcpy(out_result.data, output.data(), output.size() * sizeof(float));
    return out_result;
}

SPDetector::SPDetector(string model_path){
    model_path_ = model_path;
}

SPDetector::~SPDetector(){
    // if(ort_session_ != nullptr){
    //     ort_session_->release();
    //     ort_session_ = nullptr;
    // }
}

void SPDetector::detect(cv::Mat& img, vector<cv::KeyPoint>& res_keypoints, cv::Mat& res_descriptors){
    if(img.dims != 2){
        printf("input must has 2 dims.\n");
        return;
    }
    
    Ort::Env env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx");
	Ort::SessionOptions sessionOptions_ = Ort::SessionOptions();
    // OrtCUDAProviderOptions cuda_options; 
    Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Session ort_session_(env_, model_path_.c_str(), sessionOptions_);
    keypoints_.clear();
    descriptors_.clear();

    input_dims_.dim3 = img.rows;
    input_dims_.dim4 = img.cols;

    semi_dims_.dim2 = img.rows;
    semi_dims_.dim3 = img.cols;

    desc_dims_.dim2 = 256;
    desc_dims_.dim3 = int(img.rows / 8);
    desc_dims_.dim4 = int(img.cols / 8);

    int input_numel = img.rows * img.cols;
    float* input_data_host = new float[input_numel];
    int64_t input_shape[] = {1, 1, img.rows, img.cols};
    
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            input_data_host[row * img.cols + col] = float(img.at<unsigned char>(row, col)) / 255.0f;
        }
    }
    auto input_tensor = Ort::Value::CreateTensor(mem_, input_data_host, input_numel, input_shape, 4);
    const char* input_names_[] = {"input"};
    const char* output_names_[] = {"scores", "descriptors"};
    Ort::RunOptions options;
    std::vector<Ort::Value> ort_outputs = ort_session_.Run(options, 
        (const char* const*)input_names_, &input_tensor, 1, 
        (const char* const*)output_names_, 2);
    
    delete[] input_data_host;
    float* output_score = ort_outputs[0].GetTensorMutableData<float>();
    float* output_desc = ort_outputs[1].GetTensorMutableData<float>();

    // cout << "ort run end..." << endl;

    bool pFlag = process_output(output_score, output_desc, res_keypoints, res_descriptors);
}

bool SPDetector::process_output(float* output_score, float* output_desc, vector<cv::KeyPoint>& res_keypoints, cv::Mat& res_descriptors){
    // cout << "process_output begin..." << endl;
    int semi_feature_map_h = semi_dims_.dim2;
    int semi_feature_map_w = semi_dims_.dim3;
    std::vector<float> scores_vec(output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    // cout << "scores_vec element : " << endl;
    // cout << "00 scores_vec.size = " << scores_vec.size() << endl;
    // cout << "00 descriptors_.size = " << descriptors_.size() << endl;
    // cout << "00 keypoints_.size = " << keypoints_.size() << endl;
    // for(int i = 0; i < scores_vec.size(); i++){
    //     cout << scores_vec[i] << " ";
    // }
    // cout << endl;
    find_high_score_index(scores_vec, keypoints_, semi_feature_map_h, semi_feature_map_w,
                          keypoint_threshold_);
    // cout << "01 scores_vec.size = " << scores_vec.size() << endl;
    // cout << "01 descriptors_.size = " << descriptors_.size() << endl;
    // cout << "01 keypoints_.size = " << keypoints_.size() << endl;
    remove_borders(keypoints_, scores_vec, remove_borders_, semi_feature_map_h, semi_feature_map_w);
    // cout << "02 scores_vec.size = " << scores_vec.size() << endl;
    // cout << "02 descriptors_.size = " << descriptors_.size() << endl;
    // cout << "02 keypoints_.size = " << keypoints_.size() << endl;
    top_k_keypoints(keypoints_, scores_vec, max_keypoints_);
    // cout << "03 scores_vec.size = " << scores_vec.size() << endl;
    // cout << "03 descriptors_.size = " << descriptors_.size() << endl;
    // cout << "03 keypoints_.size = " << keypoints_.size() << endl;
    int desc_feature_dim = desc_dims_.dim2;
    int desc_feature_map_h = desc_dims_.dim3;
    int desc_feature_map_w = desc_dims_.dim4;
    sample_descriptors(keypoints_, output_desc, descriptors_, desc_feature_dim, desc_feature_map_h, desc_feature_map_w);
    // cout << "04 scores_vec.size = " << scores_vec.size() << endl;
    // cout << "04 descriptors_.size = " << descriptors_.size() << " x " << descriptors_[0].size() << endl;
    // cout << "04 keypoints_.size = " << keypoints_.size() << endl;
    // cout << "get kps and desc..." << endl;

    if(scores_vec.size() == 0){
        return false;
    }
    // cout << "descriptors_ elements: 1 - 10 " << endl; 
    // for(int i = 0; i < keypoints_.size(); i++){
    //     cout << keypoints_[i][0] << " ";
    // }
    // cout << endl;
    res_keypoints.clear();
    res_keypoints.reserve(scores_vec.size());
    for(int i = 0; i < scores_vec.size(); ++i){
        float conf = scores_vec[i];
        float x = keypoints_[i][0];
        float y = keypoints_[i][1];
        res_keypoints.push_back(cv::KeyPoint(cv::Point((int)x, (int)y), 1.0, 0.0, conf));
    }
    // cout << "res_keypoints end..." << endl;
    // cout << "descriptors_.size = " << descriptors_.size() << " x " << descriptors_[0].size() << endl;
    // cout << "scores_vec.size = " << scores_vec.size() << endl;
    // cout << "keypoints_.size = " << keypoints_.size() << " x " << keypoints_[0].size() << endl;
	res_descriptors.create(scores_vec.size(), 256, CV_32FC1);
    // cout << "res_descriptors.rows = " << res_descriptors.rows << endl;
    // cout << "res_descriptors.cols = " << res_descriptors.cols << endl;
    // cout << "fuzhi res_descriptors..." << endl;

    vector<float> desc_temp(scores_vec.size() * 256, 0);
    for(int i = 0; i < descriptors_.size(); ++i){
        for(int j = 0; j < 256; ++j){
            desc_temp[i * 256 + j] = descriptors_[i][j];
        }
    }

    res_descriptors = vector2mat(desc_temp, cv::Size(256, scores_vec.size()));

    // cout << "process_output end..." << endl;
    return true;
}

void SPDetector::remove_borders(vector<vector<int>>& keypoints, vector<float>& scores, int border, int height, int width){
    vector<std::vector<int>> keypoints_selected;
    vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (height - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (width - border));
        if (flag_h && flag_w) {
            keypoints_selected.push_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
            scores_selected.push_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

vector<size_t> SPDetector::sort_indexes(vector<float>& data){
    vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SPDetector::top_k_keypoints(vector<vector<int>>& keypoints, vector<float>& scores, int k){
    if (k < keypoints.size() && k != -1) {
        vector<std::vector<int>> keypoints_top_k;
        vector<float> scores_top_k;
        vector<size_t> indexes = sort_indexes(scores);
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.push_back(keypoints[indexes[i]]);
            scores_top_k.push_back(scores[indexes[i]]);
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}

void SPDetector::find_high_score_index(vector<float>& scores, vector<vector<int>>& keypoints, int h, int w, double threshold){
    vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}

void SPDetector::sample_descriptors(vector<vector<int>>& keypoints, float* descriptors, vector<vector<double>>& dest_descriptors,
    int dim, int h, int w, int s){
    vector<std::vector<double>> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    normalize_descriptors(dest_descriptors);
}


void image_padding(cv::Mat& src_image, cv::Mat& dst_image, int pad_h, int pad_w){
    int h = src_image.rows;
    int w = src_image.cols;
    if(h > w){
        pad_h = int((h + 7) / 8) * 8 - h;
        pad_w = int((h + 7) / 8) * 8 - w;
    }
    else{
        pad_h = int((w + 7) / 8) * 8 - h;
        pad_w = int((w + 7) / 8) * 8 - w;
    }
    cout << "pad_h = " << pad_h << ", pad_w = " << pad_w << endl;
    cv::copyMakeBorder(src_image, dst_image, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

int demo01(){

    string img0_path = "../image/1.jpg";
    string img1_path = "../image/2.jpg";

    auto image0 = cv::imread(img0_path, cv::IMREAD_GRAYSCALE);
    auto image1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);

    // cv::Mat match_Img = cv::Mat::zeros(image1.cols - image0.cols + 1, image1.rows - image0.rows + 1, CV_32FC1);
    // matchTemplate(image1, image0, match_Img, cv::TM_CCOEFF_NORMED);

    // cv::Point min_loc, max_loc;
    // double min_value, max_value;
    // minMaxLoc(match_Img, &min_value, &max_value, &min_loc, &max_loc);

    // cv::Rect roi_rect = cv::Rect(cv::Point(max_loc.x, max_loc.y), cv::Point(max_loc.x + image0.cols, max_loc.y + image0.rows));
    // cv::Point matchPoint = cv::Point(int((roi_rect.x + roi_rect.width / 2.)), (roi_rect.y + roi_rect.height / 2.));

    // cv::Mat src_image = image1(roi_rect).clone();
    // // cv::imshow("src_image", src_image);
    // cv::resize(image0, image0, cv::Size(640, 640));
    // cv::resize(src_image, src_image, cv::Size(640, 640));

    std::vector<cv::KeyPoint> image0_kpts, image1_kpts;
    cv::Mat image0_descriptors, image1_descriptors;

    SPDetector superpoint("../weights/superpoint_v1.onnx");
    superpoint.detect(image0, image0_kpts, image0_descriptors);
    superpoint.detect(image1, image1_kpts, image1_descriptors);

    cv::Mat point_image0;
    cv::drawKeypoints(image0, image0_kpts, point_image0, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::namedWindow("point_image0", cv::WINDOW_NORMAL);
    cv::imshow("point_image0", point_image0);

    cv::Mat point_image1;
    cv::drawKeypoints(image1, image1_kpts, point_image1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::namedWindow("point_image1", cv::WINDOW_NORMAL);
    cv::imshow("point_image1", point_image1);

    cout << "image0_kpts size = " << image0_kpts.size() << endl;
    cout << "image1_kpts size = " << image1_kpts.size() << endl;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	std::vector<cv::DMatch> matches;
	matcher->match(image0_descriptors, image1_descriptors, matches);

    //定义向量距离的最大值与最小值
    double max_dist = 0;
    double min_dist = 1000;
    for (int i = 1; i < image0_descriptors.rows; ++i)
    {
        //通过循环更新距离，距离越小越匹配
        double dist = matches[i].distance;
        // cout << dist << endl;
        if (dist > max_dist)
            max_dist = dist;
        if (dist < min_dist)
            min_dist = dist;
    }
    cout << "min_dist=" << min_dist << endl;
    cout << "max_dist=" << max_dist << endl;
    //匹配结果筛选    
    vector<cv::DMatch> goodMatches;
    for (int i = 0; i < matches.size(); ++i)
    {
        double dist = matches[i].distance;
        // cout << dist << endl;
        if (min_dist != 0 && dist < 3 * min_dist)
            goodMatches.push_back(matches[i]);
        if(min_dist ==0 && dist < 0.02 * max_dist)
            goodMatches.push_back(matches[i]);
    }
    cout << "goodMatches:" << goodMatches.size() << endl;


	cv::Mat imgMatches;
	cv::drawMatches(image0, image0_kpts, image1, image1_kpts, goodMatches, imgMatches); 
	cv::namedWindow("keypoint matches", cv::WINDOW_NORMAL); //可任意改变窗口大小
	cv::imshow("keypoint matches", imgMatches);
	// cv::imwrite("../image/3-4_res.jpg", imgMatches);
	cv::waitKey(0);

    return 0;
}

int demo02(){

    string img0_path = "../image/1.jpg";
    string img1_path = "../image/2.jpg";

    auto image0 = cv::imread(img0_path, cv::IMREAD_GRAYSCALE);
    auto image1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);

    // cv::Mat match_Img = cv::Mat::zeros(image1.cols - image0.cols + 1, image1.rows - image0.rows + 1, CV_32FC1);
    // matchTemplate(image1, image0, match_Img, cv::TM_CCOEFF_NORMED);

    // cv::Point min_loc, max_loc;
    // double min_value, max_value;
    // minMaxLoc(match_Img, &min_value, &max_value, &min_loc, &max_loc);

    // cv::Rect roi_rect = cv::Rect(cv::Point(max_loc.x, max_loc.y), cv::Point(max_loc.x + image0.cols, max_loc.y + image0.rows));
    // cv::Point matchPoint = cv::Point(int((roi_rect.x + roi_rect.width / 2.)), (roi_rect.y + roi_rect.height / 2.));

    // cv::Mat src_image = image1(roi_rect).clone();
    // // cv::imshow("src_image", src_image);
    // cv::resize(image0, image0, cv::Size(640, 640));
    // cv::resize(src_image, src_image, cv::Size(640, 640));

    std::vector<cv::KeyPoint> image0_kpts, image1_kpts;
    cv::Mat image0_descriptors, image1_descriptors;

    auto start = std::chrono::steady_clock::now();
    SPDetector superpoint("../weights/superpoint_v1.onnx");
    superpoint.detect(image0, image0_kpts, image0_descriptors);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> d = end - start;
    cout << "detect need " << d.count() * 1000 << " ms." << endl;

    return 0;
}


int main(int argc, char** argv){
    sleep(5);
    cout << "starting ... " << endl;
    for(int i=0; i < 20; i++){
        cout << "<<---------->>" << endl;
        demo02();
        // sleep(5);
    }
    // sleep(5);
    demo02();
    cout << "ending ... " << endl;
    // demo02();
    // demo03();
    return 0;
}
