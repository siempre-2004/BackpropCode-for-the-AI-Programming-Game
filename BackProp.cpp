/*
 *  Backpropagation Algorithm
 *
 *  mtj@cogitollc.com
 *
 */

#include "BackProp.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

BackProp::BackProp(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
    if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Layer sizes must be positive.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    // Xavier-style initialization range for more stable training.
    double limitIH = std::sqrt(6.0 / (inputSize + hiddenSize));
    double limitHO = std::sqrt(6.0 / (hiddenSize + outputSize));

    std::uniform_real_distribution<double> distIH(-limitIH, limitIH);
    std::uniform_real_distribution<double> distHO(-limitHO, limitHO);

    weightsInputHidden.assign(inputSize, std::vector<double>(hiddenSize));
    weightsHiddenOutput.assign(hiddenSize, std::vector<double>(outputSize));
    biasHidden.assign(hiddenSize, 0.0);
    biasOutput.assign(outputSize, 0.0);

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] = distIH(gen);
        }
    }

    for (int j = 0; j < hiddenSize; ++j) {
        for (int k = 0; k < outputSize; ++k) {
            weightsHiddenOutput[j][k] = distHO(gen);
        }
    }
}

double BackProp::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double BackProp::sigmoidDerivative(double activatedValue) {
    return activatedValue * (1.0 - activatedValue);
}

std::vector<double> BackProp::forward(const std::vector<double>& input,
    std::vector<double>* hiddenOutputs) const {
    if (static_cast<int>(input.size()) != inputSize) {
        throw std::invalid_argument("Input size does not match network input size.");
    }

    std::vector<double> hidden(hiddenSize, 0.0);
    for (int j = 0; j < hiddenSize; ++j) {
        double sum = biasHidden[j];
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weightsInputHidden[i][j];
        }
        hidden[j] = sigmoid(sum);
    }

    std::vector<double> output(outputSize, 0.0);
    for (int k = 0; k < outputSize; ++k) {
        double sum = biasOutput[k];
        for (int j = 0; j < hiddenSize; ++j) {
            sum += hidden[j] * weightsHiddenOutput[j][k];
        }
        output[k] = sigmoid(sum);
    }

    if (hiddenOutputs != nullptr) {
        *hiddenOutputs = hidden;
    }

    return output;
}

void BackProp::backward(const std::vector<double>& input,
    const std::vector<double>& target,
    const std::vector<double>& hiddenOutputs,
    const std::vector<double>& finalOutputs,
    double learningRate) {
    std::vector<double> outputErrors(outputSize, 0.0);
    for (int k = 0; k < outputSize; ++k) {
        outputErrors[k] =
            (target[k] - finalOutputs[k]) * sigmoidDerivative(finalOutputs[k]);
    }

    std::vector<double> hiddenErrors(hiddenSize, 0.0);
    for (int j = 0; j < hiddenSize; ++j) {
        double propagatedError = 0.0;
        for (int k = 0; k < outputSize; ++k) {
            propagatedError += outputErrors[k] * weightsHiddenOutput[j][k];
        }
        hiddenErrors[j] = propagatedError * sigmoidDerivative(hiddenOutputs[j]);
    }

    for (int j = 0; j < hiddenSize; ++j) {
        for (int k = 0; k < outputSize; ++k) {
            weightsHiddenOutput[j][k] += learningRate * outputErrors[k] * hiddenOutputs[j];
        }
    }

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * input[i];
        }
    }

    for (int k = 0; k < outputSize; ++k) {
        biasOutput[k] += learningRate * outputErrors[k];
    }

    for (int j = 0; j < hiddenSize; ++j) {
        biasHidden[j] += learningRate * hiddenErrors[j];
    }
}

int BackProp::argmax(const std::vector<double>& values) {
    return static_cast<int>(
        std::distance(values.begin(),
            std::max_element(values.begin(), values.end())));
}

std::string BackProp::actionName(int index) {
    switch (index) {
    case 0: return "Hunt";
    case 1: return "Eat";
    case 2: return "TakeCover";
    case 3: return "BeVigilant";
    default: return "Unknown";
    }
}

void BackProp::train(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets,
    int epochs,
    double learningRate) {
    if (inputs.empty() || targets.empty() || inputs.size() != targets.size()) {
        throw std::invalid_argument("Training inputs and targets must be non-empty and have equal size.");
    }

    std::vector<int> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);

        double totalLoss = 0.0;

        for (int idx : indices) {
            std::vector<double> hiddenOutputs;
            std::vector<double> finalOutputs = forward(inputs[idx], &hiddenOutputs);

            for (int k = 0; k < outputSize; ++k) {
                double error = targets[idx][k] - finalOutputs[k];
                totalLoss += error * error;
            }

            backward(inputs[idx], targets[idx], hiddenOutputs, finalOutputs, learningRate);
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            double averageLoss = totalLoss / static_cast<double>(inputs.size());
            double accuracy = evaluateAccuracy(inputs, targets);

            std::cout << "Epoch " << std::setw(4) << epoch
                << " | Average Loss: " << std::fixed << std::setprecision(6) << averageLoss
                << " | Training Accuracy: " << std::setprecision(2) << accuracy << "%\n";
        }
    }
}

double BackProp::evaluateAccuracy(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets) const {
    int correct = 0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        int predicted = argmax(predict(inputs[i]));
        int actual = argmax(targets[i]);

        if (predicted == actual) {
            ++correct;
        }
    }

    return 100.0 * static_cast<double>(correct) / static_cast<double>(inputs.size());
}

std::vector<double> BackProp::predict(const std::vector<double>& input) const {
    return forward(input, nullptr);
}

void BackProp::test(const std::vector<std::vector<double>>& testInputs,
    const std::vector<std::vector<double>>& testTargets) const {
    if (testInputs.empty() || testTargets.empty() || testInputs.size() != testTargets.size()) {
        throw std::invalid_argument("Test inputs and targets must be non-empty and have equal size.");
    }

    int correct = 0;

    std::cout << "\nTesting neural network:\n";
    std::cout << "Input\t\t\tPredicted\tActual\t\tCorrect?\n";

    for (size_t i = 0; i < testInputs.size(); ++i) {
        std::vector<double> output = predict(testInputs[i]);

        int predictedIndex = argmax(output);
        int actualIndex = argmax(testTargets[i]);

        std::cout << "[";
        for (size_t j = 0; j < testInputs[i].size(); ++j) {
            std::cout << testInputs[i][j];
            if (j + 1 < testInputs[i].size()) {
                std::cout << ", ";
            }
        }
        std::cout << "]\t";

        std::cout << actionName(predictedIndex) << "\t";
        if (actionName(predictedIndex).length() < 8) {
            std::cout << "\t";
        }

        std::cout << actionName(actualIndex) << "\t";
        if (actionName(actualIndex).length() < 8) {
            std::cout << "\t";
        }

        bool isCorrect = (predictedIndex == actualIndex);
        std::cout << (isCorrect ? "Yes" : "No") << "\n";

        if (isCorrect) {
            ++correct;
        }
    }

    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(testInputs.size());

    std::cout << "\nTest Accuracy: " << std::fixed << std::setprecision(2)
        << accuracy << "% (" << correct << "/" << testInputs.size() << ")\n";
}
/*
 *  Copyright (c) 2003-2005 Charles River Media.  All rights reserved.
 * 
 *  Redistribution and use in source and binary forms, with or 
 *  without modification, is hereby granted without fee provided 
 *  that the following conditions are met:
 * 
 *    1.  Redistributions of source code must retain the above 
 *        copyright notice, this list of conditions and the 
 *        following disclaimer.
 *    2.  Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the 
 *        following disclaimer in the documentation and/or other 
 *        materials provided with the distribution.
 *    3.  Neither the name of Charles River Media nor the names of 
 *        its contributors may be used to endorse or promote 
 *        products derived from this software without specific 
 *        prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY CHARLES RIVER MEDIA AND CONTRIBUTORS 
 * 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTIBILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CHARLES
 * RIVER MEDIA OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS 
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

