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
            }
            data[row] = std::move(img);
            labels[row] = i;
        }
    }
}

int main() {
    cout << "MAIN START" << endl;

    //MLP mlp(0.01);
    MLP mlp;

    vector<vector<float>> data;
    vector<int> labels;
    int data_amt = 100;
    int epochs = 5000;

    //training_data(data, labels, data_amt);
    
    //mlp.train(data, labels, epochs);
    Image_Processing ip;
    string path = "testing/9_0_test" + ip.raster;

    auto prediction = mlp.predict(ip.processImage(path, true));

    cout << "The MLP predicts an " << prediction.first << " with " << prediction.second << " confidence" << endl;

    return 0;
}