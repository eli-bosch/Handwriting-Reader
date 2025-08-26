#include "mlp.hpp"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <iostream>

MLP::MLP(float learningRate) {

    this->learningRate = learningRate;

    //Buffers
    hidden.resize(128, 0.f);
    output.resize(10, 0.f);

    //Weights
    weights_input_hidden.resize(128, std::vector<float>(784));
    weights_hidden_output.resize(10, std::vector<float>(128));

    //Randomize Weights
    xavier_init(weights_input_hidden, 128, 784);
    xavier_init(weights_hidden_output, 10, 128);

    //Bias
    bias_hidden.resize(128, 0.f);
    bias_output.resize(10, 0.f);
}

//Activations
float MLP::tanhActivation(float x) {
    return std::tanh(x);
}

float MLP::tanhDerivative(float y) {
    return (1.f - y * y);
}

//Inference
std::vector<float> MLP::forward(const std::vector<float>& input) {
    std::fill(hidden.begin(), hidden.end(), 0.f);

    //Compute hidden layer activations
    for(int i = 0; i < 128; i++) {
        float sum = bias_hidden[i];
        const auto& weights_hidden = weights_input_hidden[i];
        for(int j = 0; j < 784; j++) {
            sum += weights_hidden[j] * input[i];
        }

        hidden[i] = tanhActivation(sum);
    }

    std::fill(output.begin(), output.end(), 0.f);

    //Compute output layer activations
    for(int i = 0; i < 10; i++) {
        float output_sum = bias_output[i];
        const auto& weights_output = weights_hidden_output[i];
        for(int j = 0; j < 128; j++) {
            output_sum += weights_output[j] * hidden[j];
        }

        output[i] = output_sum;
    }

    softmax_inplace(output);

    return output;
}

//Training

float MLP::train_sample(const std::vector<float>& x, int label) { //Train with single input
    std::vector<float> output = forward(x);
    float loss = cross_entropy(output, label); //Consolidate with backward doing all the backward propogation
    backward(x, label);
    return loss;
}

void MLP::train(const std::vector<std::vector<float>>& X, const std::vector<int>& labels, int epochs) { //Actual training driver
    std::vector<std::pair<std::vector<float>, int>> dataset;
    for(int i = 0; i < X.size(); i++) {
        dataset.push_back({X[i], labels[i]});
    }

    for(int epoch = 0; epoch < epochs; epoch++) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(dataset.begin(), dataset.end(), g);
        

        float error = 0.f;
        for(const auto& n : dataset) {
            error += train_sample(n.first, n.second);
        }

        if(epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ": total error = " << error << std::endl;
        }
    }
}

std::pair<int, float> MLP::predict(const std::vector<float>& input) { //Returns digit and confidence
    const auto& output = forward(input);
    int digit = 0;
    float prob = input[0];

    for(int i = 1; i < 10; i++) {
        if(output[digit] < output[i]) {digit = i; prob = input[i];}
    }

    return {digit, prob};
}

//Helpers
void MLP::xavier_init(std::vector<std::vector<float>>& W, int fan_out, int fan_in) { //Optimizes random weight range for better training
    //Initialize radnom seed in range of fan
    int limit = std::sqrt(6 / (fan_in + fan_out));
    std::mt19937 rng(static_cast<uint32_t>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for(int r = 0; r < fan_out; r++) {
        for(int c = 0; c < fan_in; c++) {
            W[r][c] = dist(rng); 
        }
    }
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
float MLP::backward(const std::vector<float>& x, int label) {
    float delta_out[10];

    for(int o = 0; o < 10; o++) delta_out[o] = output[o];
    delta_out[label] -= 1.f;

    float delta_h[128] = {0.f};
    for(int o = 0; o < 10; o++) {
        bias_output[o] -= learningRate * delta_out[o];

        auto& W2o = weights_hidden_output[o]; // 128
        for(int h = 0; h < 128; h++) {
            delta_h[h] += W2o[h] * delta_out[o];
            W2o[h] -= learningRate * (delta_out[o] * hidden[h]);
        }
    }

    for(int h = 0; h < 128; h++) {
        delta_h[h] *= tanhDerivative(hidden[h]);
    }

    for(int h = 0; h < 128; h++) {
        bias_hidden[h] -= learningRate * delta_h[h];

        auto& W1h = weights_input_hidden[h]; // 784
        for(int i = 0; i < 784; i++) {
            W1h[i] -= learningRate * (delta_h[h] * x[i]);
        }
    }
}