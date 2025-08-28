#include "mlp.hpp"

#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

MLP::MLP() {
    this->unmarshal();
}

MLP::MLP(float learningRate) {
    std::cout << "MLP CONSTRUCTED" << std::endl;

    this->learningRate = learningRate;

    //Buffers
    hidden.resize(hidden_layer_size, 0.f);
    output.resize(output_layer_size, 0.f);

    //Weights
    weights_input_hidden.resize(hidden_layer_size, std::vector<float>(input_layer_size));
    weights_hidden_output.resize(output_layer_size, std::vector<float>(hidden_layer_size));

    //Randomize Weights
    xavier_init(weights_input_hidden, hidden_layer_size, input_layer_size);
    xavier_init(weights_hidden_output, output_layer_size, hidden_layer_size);

    //Bias
    bias_hidden.resize(hidden_layer_size, 0.f);
    bias_output.resize(output_layer_size, 0.f);
}

//Activations
float MLP::tanhActivation(float x) {
    return std::tanh(x);
}

float MLP::tanhDerivative(float y) {
    return (1.f - y * y);
}

//Inference
void MLP::forward(const std::vector<float>& input) {
    std::fill(hidden.begin(), hidden.end(), 0.f);

    //Compute hidden layer activations
    for(int i = 0; i < hidden_layer_size; i++) {
        float sum = bias_hidden[i];
        const auto& weights_hidden = weights_input_hidden[i];
        for(int j = 0; j < input_layer_size; j++) {
            sum += weights_hidden[j] * input[j];
        }

        hidden[i] = tanhActivation(sum);
    }

    std::fill(output.begin(), output.end(), 0.f);

    //Compute output layer activations
    for(int i = 0; i < output_layer_size; i++) {
        float output_sum = bias_output[i];
        const auto& weights_output = weights_hidden_output[i];
        for(int j = 0; j < hidden_layer_size; j++) {
            output_sum += weights_output[j] * hidden[j];
        }

        output[i] = output_sum;
    }

    softmax_inplace(output);
}

//Training

inline float MLP::train_sample(const std::vector<float>& x, int label) { //Train with single input
    forward(x);
    float loss = cross_entropy(output, label); //Consolidate with backward doing all the backward propogation
    backward(x, label);
    return loss;
}

void MLP::train(const std::vector<std::vector<float>>& X, const std::vector<int>& labels, int epochs) { //Actual training driver
    std::cout << "TRAINING BEGINS" << std::endl;
    
    std::vector<std::pair<std::vector<float>, int>> dataset;
    for(int i = 0; i < labels.size(); i++) {
        dataset.push_back({X[i], labels[i]});
    }

    std::random_device rd;
    std::mt19937 g(rd());

    for(int epoch = 0; epoch < epochs; epoch++) {
        
        std::shuffle(dataset.begin(), dataset.end(), g);
        

        float error = 0.f;
        for(const auto& n : dataset) {
            error += train_sample(n.first, n.second);
        }

        if(epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ": average error = " << (error / labels.size()) << std::endl;
        }
    }

    this->marshal();
}

std::pair<int, float> MLP::predict(const std::vector<float>& input) { //Returns digit and confidence
    std::cout << "PREDICTION BEGINS" << std::endl;

    forward(input);

    int digit = 0;
    float prob = output[0];

    for(int i = 1; i < output_layer_size; i++) {
        if(output[digit] < output[i]) {digit = i; prob = output[i];}
    }

    return {digit, prob};
}

//Helpers
void MLP::xavier_init(std::vector<std::vector<float>>& W, int fan_out, int fan_in) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));  
    std::mt19937 rng(42u); // fixed seed = reproducible training
    std::uniform_real_distribution<float> dist(-limit, limit);            

    for (int r = 0; r < fan_out; ++r)
        for (int c = 0; c < fan_in; ++c)
            W[r][c] = dist(rng);
}


void MLP::softmax_inplace(std::vector<float>& z) { //Coverts vector to stable probability distribution
    float m = *std::max_element(z.begin(), z.end());
    double sum = 0.0;
    for(float&v : z) {
        v = std::exp(v - m);
        sum += v;
    }

    float inv = 1.f / static_cast<float>(sum);
    for(float& v : z) v *= inv;
}

float MLP::cross_entropy(const std::vector<float>& p, int label) {
    const float eps = 1e-12f; //Avoids log(0)
    float q = std::max(p[label], eps);
    return -std::log(q);
}

//Backpropagation
void MLP::backward(const std::vector<float>& x, int label) {
    float delta_out[output_layer_size];

    for(int o = 0; o < output_layer_size; o++) delta_out[o] = output[o];
    delta_out[label] -= 1.f;

    float delta_h[hidden_layer_size] = {0.f};
    for(int o = 0; o < output_layer_size; o++) {
        bias_output[o] -= learningRate * delta_out[o];

        auto& W2o = weights_hidden_output[o]; // 128
        for(int h = 0; h < hidden_layer_size; h++) {
            delta_h[h] += W2o[h] * delta_out[o];
            W2o[h] -= learningRate * (delta_out[o] * hidden[h]);
        }
    }

    for(int h = 0; h < hidden_layer_size; h++) {
        delta_h[h] *= tanhDerivative(hidden[h]);
    }

    for(int h = 0; h < hidden_layer_size; h++) {
        bias_hidden[h] -= learningRate * delta_h[h];

        auto& W1h = weights_input_hidden[h]; // 784
        for(int i = 0; i < input_layer_size; i++) {
            W1h[i] -= learningRate * (delta_h[h] * x[i]);
        }
    }
}

//JSON

void MLP::marshal() {
    static constexpr const char* kPath = "mlp_model.json";

    // Infer dims from current containers (should be 784-128-10 in your code)
    const std::size_t hidden_dim = weights_input_hidden.size();
    const std::size_t input_dim  = hidden_dim ? weights_input_hidden[0].size() : 784;
    const std::size_t output_dim = weights_hidden_output.size();

    // Basic structural check before saving
    if (hidden_dim != 128 || input_dim != 784 || output_dim != 10)
        std::cerr << "[marshal] Warning: dims not 784-128-10 ("
                  << input_dim << "-" << hidden_dim << "-" << output_dim << ")\n";

    json j;
    j["version"] = 1;
    j["arch"]    = { {"input", input_dim}, {"hidden", hidden_dim}, {"output", output_dim} };
    j["hyper"]   = { {"learning_rate", learningRate}, {"activation", "tanh"} };

    // nlohmann/json can serialize std::vector and std::vector<std::vector<float>> directly
    j["params"]["W1"] = weights_input_hidden;   // [hidden][input]
    j["params"]["b1"] = bias_hidden;            // [hidden]
    j["params"]["W2"] = weights_hidden_output;  // [output][hidden]
    j["params"]["b2"] = bias_output;            // [output]

    std::ofstream out(kPath);
    if (!out) throw std::runtime_error(std::string("Failed to open for write: ") + kPath);
    out << j.dump(2) << '\n';  // pretty-printed; use dump() for compact
}

void MLP::unmarshal() {
    static constexpr const char* kPath = "mlp_model.json";

    std::ifstream in(kPath);
    if (!in) throw std::runtime_error(std::string("Failed to open for read: ") + kPath);

    json j;
    in >> j;

    // Validate header
    if (!j.contains("arch") || !j.contains("params"))
        throw std::runtime_error("Model JSON missing 'arch' or 'params'");

    const int in_dim  = j.at("arch").at("input").get<int>();
    const int hid_dim = j.at("arch").at("hidden").get<int>();
    const int out_dim = j.at("arch").at("output").get<int>();

    // Code hardcodes 784-128-10
    if (in_dim != 784 || hid_dim != 128 || out_dim != 10)
        throw std::runtime_error("Model dims mismatch (expected 784-128-10)");

    // Pull parameters
    auto W1 = j.at("params").at("W1").get<std::vector<std::vector<float>>>();
    auto b1 = j.at("params").at("b1").get<std::vector<float>>();
    auto W2 = j.at("params").at("W2").get<std::vector<std::vector<float>>>();
    auto b2 = j.at("params").at("b2").get<std::vector<float>>();

    // Shape checks
    if (static_cast<int>(W1.size()) != hid_dim)  throw std::runtime_error("W1 rows != hidden");
    for (const auto& row : W1)
        if (static_cast<int>(row.size()) != in_dim) throw std::runtime_error("W1 row size != input");

    if (static_cast<int>(W2.size()) != out_dim)  throw std::runtime_error("W2 rows != output");
    for (const auto& row : W2)
        if (static_cast<int>(row.size()) != hid_dim) throw std::runtime_error("W2 row size != hidden");

    if (static_cast<int>(b1.size()) != hid_dim)  throw std::runtime_error("b1 size != hidden");
    if (static_cast<int>(b2.size()) != out_dim)  throw std::runtime_error("b2 size != output");

    // Commit into the model
    weights_input_hidden = std::move(W1);
    bias_hidden          = std::move(b1);
    weights_hidden_output= std::move(W2);
    bias_output          = std::move(b2);

    // Make sure runtime buffers exist with the right sizes
    hidden.assign(hid_dim, 0.f);
    output.assign(out_dim, 0.f);

    // Optional: restore LR if present
    if (j.contains("hyper") && j["hyper"].contains("learning_rate")) {
        learningRate = j["hyper"]["learning_rate"].get<float>();
    }
}
