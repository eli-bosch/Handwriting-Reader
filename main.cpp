#include <iostream>
#include <cstring>

#include "mlp.hpp"
#include "image_processing.hpp"

using namespace std;

void training_data(vector<vector<float>>& data, vector<int>& labels, int data_amt) {
    Image_Processing ip;
    const int W = 28 * 28;

    data.assign(10 * data_amt, {});  
    labels.assign(10 * data_amt, 0);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < data_amt; j++) {      
            const int row = i * data_amt + j;

            std::string path = "training/" + to_string(i) + "/" + to_string(j + 1) + ip.raster;

            auto img = ip.processImage(path, /*invert=*/false);
            if (img.size() != W) {
                std::cerr << "BAD SIZE at row " << row << " got "
                          << img.size() << " expected " << W << "\n";

                if(row != 0) {
                    data[row] = data[row - 1];
                    labels[row] = i;
                    continue;
                }
            }
            data[row] = std::move(img);
            labels[row] = i;
        }
    }
}

void train() {
    MLP mlp(0.01);

    vector<vector<float>> data;
    vector<int> labels;
    int data_amt = 1000;
    int epochs = 3000;

    cout << "MAIN START, data_amt = " << data_amt*10 << ", epochs = " << epochs << endl;

    training_data(data, labels, data_amt);
    
    mlp.train(data, labels, epochs);
}

void predict() {
    MLP mlp;
    Image_Processing ip;
    string path = "testing/9_3" + ip.raster;

    auto prediction = mlp.predict(ip.processImage(path, true));

    cout << "The MLP predicts an " << prediction.first << " with " << prediction.second << " confidence" << endl;
}

int main() {
   
    //train();
    predict();

    return 0;
}