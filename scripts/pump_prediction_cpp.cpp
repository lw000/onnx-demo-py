#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <iomanip>

class PumpFailurePredictor {
private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    std::string input_name_;
    std::string output_label_name_;
    std::string output_proba_name_;
    const char* label_names_[3] = {"normal", "wear", "cavitation"};

public:
    PumpFailurePredictor(const std::wstring& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "test")
        , session_(env_, model_path.c_str(), Ort::SessionOptions{nullptr})
        , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        char* input_name = session_.GetInputName(0, allocator);
        char* output_label_name = session_.GetOutputName(0, allocator);
        char* output_proba_name = session_.GetOutputName(1, allocator);

        input_name_ = input_name;
        output_label_name_ = output_label_name;
        output_proba_name_ = output_proba_name;

        std::cout << "=== 模型加载成功 ===" << std::endl;
        std::cout << "输入节点: " << input_name_ << std::endl;
        std::cout << "输出[0](标签): " << output_label_name_ << std::endl;
        std::cout << "输出[1](概率): " << output_proba_name_ << std::endl << std::endl;

        allocator.Free(input_name);
        allocator.Free(output_label_name);
        allocator.Free(output_proba_name);
    }

    struct PredictionResult {
        int64_t label;
        std::string label_name;
        std::vector<float> probabilities;
        float confidence;
    };

    PredictionResult predict(float flow, float head, float power, float vibration) {
        // 1. 准备输入数据
        std::vector<float> input_values = {flow, head, power, vibration};
        std::vector<int64_t> input_shape = {1, 4};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 2. 运行推理 - 关键：指定 2 个输出
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_label_name_.c_str(), output_proba_name_.c_str()};

        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2  // 必须是 2，获取标签和概率两个输出
        );

        // 3. 获取标签输出 (int64)
        auto& label_tensor = output_tensors[0];
        int64_t label_value = *label_tensor.GetTensorData<int64_t>();

        // 4. 获取概率输出 (float32) - 关键步骤
        auto& proba_tensor = output_tensors[1];
        float* proba_data = proba_tensor.GetTensorMutableData<float>();

        // 获取概率数组形状
        auto proba_shape = proba_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int64_t num_classes = proba_shape[1];

        // 5. 构建结果
        PredictionResult result;
        result.label = label_value;
        result.label_name = label_names_[label_value];
        result.probabilities.assign(proba_data, proba_data + num_classes);
        result.confidence = proba_data[label_value];

        return result;
    }

    void printResult(const PredictionResult& result) {
        std::cout << "预测结果:" << std::endl;
        std::cout << "  标签: " << result.label << " (" << result.label_name << ")" << std::endl;
        std::cout << "  置信度: " << std::fixed << std::setprecision(6) << result.confidence << std::endl;
        std::cout << "  概率分布:" << std::endl;
        for (int i = 0; i < 3; i++) {
            std::cout << "    " << i << " (" << label_names_[i] << "): "
                      << std::fixed << std::setprecision(6)
                      << result.probabilities[i] << std::endl;
        }
        std::cout << std::endl;
    }
};

int main() {
    try {
        // 1. 加载模型
        PumpFailurePredictor predictor(L"pump_failure_classifier.onnx");

        // 2. 测试预测 - 正常状态
        std::cout << "=== 测试 1: 正常状态 ===" << std::endl;
        auto result1 = predictor.predict(110.0f, 55.0f, 50.0f, 1.2f);
        predictor.printResult(result1);

        // 3. 测试预测 - 磨损状态
        std::cout << "=== 测试 2: 磨损状态 ===" << std::endl;
        auto result2 = predictor.predict(100.0f, 50.0f, 55.0f, 2.0f);
        predictor.printResult(result2);

        // 4. 测试预测 - 汽蚀状态
        std::cout << "=== 测试 3: 汽蚀状态 ===" << std::endl;
        auto result3 = predictor.predict(90.0f, 45.0f, 45.0f, 3.5f);
        predictor.printResult(result3);

        std::cout << "=== 所有测试完成 ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
