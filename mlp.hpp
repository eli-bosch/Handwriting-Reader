#ifndef MLP_HPP
#define MLP_HPP

#include <vector>

class MLP {
    public:
        explicit MLP();
        explicit MLP(float learningRate);

        //Activations
        static float tanhActivation(float x);
        static float tanhDerivative(float y);

        //Inference
        void forward(const std::vector<float>& input);
        std::pair<int, float> predict(const std::vector<float>& input);

        //Training
        inline float train_sample(const std::vector<float>& x, int label);
        void train(const std::vector<std::vector<float>>& X, const std::vector<int>& labels, int epochs);

    private:
        float learningRate;

        //Layer sizes
        int input_layer_size = 784; //28*28
        int hidden_layer_size = 128;
        int output_layer_size = 10;

        //Layer buffers
        std::vector<float> hidden; // 128
        std::vector<float> output; // 10

        //Weights
        std::vector<std::vector<float>> weights_input_hidden; //128x784
        std::vector<std::vector<float>> weights_hidden_output; //10x128

        //Bias
        std::vector<float> bias_hidden; //128
        std::vector<float> bias_output; //10

        //Helpers
        void xavier_init(std::vector<std::vector<float>>& W, int fan_out, int fan_in); //Used for both weights matrices
        static void softmax_inplace(std::vector<float>& z); //max-trick
        static float cross_entropy(const std::vector<float>& p, int label);

        //Backpropagation
        void backward(const std::vector<float> & x, int label);

        //JSON
        void marshal();
        void unmarshal();
};

#endif