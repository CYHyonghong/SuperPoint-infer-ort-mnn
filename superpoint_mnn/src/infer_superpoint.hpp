#ifndef INFER_SUPERPOINT_HPP
#define INFER_SUPERPOINT_HPP

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>


struct TDims
{
    int dim1 = 1;
    int dim2 = 1;
    int dim3 = 0;
    int dim4 = 0;
};


class SPDetector{

public:
    SPDetector(std::string model_path);

    ~SPDetector();
    
    void detect(cv::Mat& img, std::vector<cv::KeyPoint>& _keypoints, cv::Mat &_descriptors);

private:
    // std::string model_path_ = "";

    TDims input_dims_;
    TDims semi_dims_;
    TDims desc_dims_;
    std::vector<std::vector<int>> keypoints_;
    std::vector<std::vector<double>> descriptors_;

    const std::vector<std::string> output_name_ = {"scores", "descriptors"};
    float norm_vals_ = 1 / 255.0f;
    float keypoint_threshold_ = 0.005; 
    int remove_borders_ = 8;
    int max_keypoints_ = 5000;
    
    std::shared_ptr<MNN::Interpreter> SP_interpreter;
    MNN::Session* SP_session = nullptr;
    MNN::Tensor* input_tensor = nullptr;

    bool process_output(float* output_score, float* output_desc, std::vector<cv::KeyPoint>& res_keypoints, cv::Mat& res_descriptors);
    
    void remove_borders(std::vector<std::vector<int>>& keypoints, std::vector<float>& scores, int border, int height,
                        int width);
    
    std::vector<size_t> sort_indexes(std::vector<float>& data);

    void top_k_keypoints(std::vector<std::vector<int>>& keypoints, std::vector<float>& scores, int k);
 
    void find_high_score_index(std::vector<float>& scores, std::vector<std::vector<int>>& keypoints, int h, int w,
                                double threshold);

    void sample_descriptors(std::vector<std::vector<int>>& keypoints, float* descriptors,
                            std::vector<std::vector<double>>& dest_descriptors, int dim, int h, int w, int s = 8);

};


#endif // !INFER_SUPERPOINT_HPP