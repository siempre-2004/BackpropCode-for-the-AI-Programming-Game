/*
 *  Backpropagation Algorithm
 *
 *  mtj@cogitollc.com
 *
 */

#ifndef BACKPROP_H
#define BACKPROP_H

#include <vector>
#include <string>

class BackProp {
public:
    BackProp(int inputSize, int hiddenSize, int outputSize);

    void train(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets,
        int epochs,
        double learningRate);

    std::vector<double> predict(const std::vector<double>& input) const;

    void test(const std::vector<std::vector<double>>& testInputs,
        const std::vector<std::vector<double>>& testTargets) const;

    double evaluateAccuracy(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets) const;

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasHidden;
    std::vector<double> biasOutput;

    static double sigmoid(double x);
    static double sigmoidDerivative(double activatedValue);

    std::vector<double> forward(const std::vector<double>& input,
        std::vector<double>* hiddenOutputs = nullptr) const;

    void backward(const std::vector<double>& input,
        const std::vector<double>& target,
        const std::vector<double>& hiddenOutputs,
        const std::vector<double>& finalOutputs,
        double learningRate);

    static int argmax(const std::vector<double>& values);
    static std::string actionName(int index);
};

#endif
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

