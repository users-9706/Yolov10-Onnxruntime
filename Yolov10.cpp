#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
std::vector<std::string> readClassNames(const std::string& filename) {
    std::vector<std::string> classNames;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return classNames;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }
    file.close();
    return classNames;
}
int main(int argc, char* argv[])
{
    std::string filename = "coco.names";
    std::vector<std::string> classes_names = readClassNames(filename);
    std::string model_path = "yolov10n.onnx";
    std::string image_path = "bus.jpg";
    float conf_thresold = 0.5f;
    std::wstring modelPath = std::wstring(model_path.begin(), model_path.end());
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov10");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session(env, modelPath.c_str(), session_options);
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        throw std::runtime_error("Could not read the image: " + image_path);
    }
    int img_width = image.cols;
    int img_height = image.rows;
    int64 start = cv::getTickCount();
    int _max = std::max(img_height, img_width);
    cv::Mat resized_image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, img_width, img_height);
    image.copyTo(resized_image(roi));
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
    }
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1];
    output_w = output_dims[2];
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;
    float x_factor = resized_image.cols / static_cast<float>(input_w);
    float y_factor = resized_image.rows / static_cast<float>(input_h);
    cv::Mat blob = cv::dnn::blobFromImage(resized_image, 1 / 255.0, cv::Size(input_h, input_w), cv::Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_h * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    std::vector<Ort::Value> output_tensors;
    try
    {
        output_tensors = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor, 1, outNames.data(), 1);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    const float* pdata = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat detection_outputs(output_h, output_w, CV_32F, (float*)pdata);
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < detection_outputs.rows; ++i) 
    {
        double confidence = detection_outputs.at<float>(i, 4);
        if (confidence > conf_thresold) 
        {
            const float x1 = detection_outputs.at<float>(i, 0);
            const float y1 = detection_outputs.at<float>(i, 1);
            const float x2 = detection_outputs.at<float>(i, 2);
            const float y2 = detection_outputs.at<float>(i, 3);
            int class_id = static_cast<int>(detection_outputs.at<float>(i, 5));
            int width = static_cast<int>((x2 - x1) * x_factor);
            int height = static_cast<int>((y2 - y1) * y_factor);
            int x = static_cast<int>(x1 * x_factor);
            int y = static_cast<int>(y1 * y_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            boxes.push_back(box);
            classIds.push_back(class_id);
            confidences.push_back(confidence);
        }
    }
    for (size_t i = 0; i < boxes.size(); i++) 
    {
        cv::rectangle(image, boxes[i], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(image, cv::Point(boxes[i].tl().x, boxes[i].tl().y - 20),
            cv::Point(boxes[i].br().x, boxes[i].tl().y), cv::Scalar(0, 255, 255), -1);
        cv::putText(image, classes_names[classIds[i]], cv::Point(boxes[i].tl().x, boxes[i].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
    }
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
    cv::imshow("YOLOv10+ONNXRUNTIME", image);
    cv::waitKey(0);
    return 0;
}
