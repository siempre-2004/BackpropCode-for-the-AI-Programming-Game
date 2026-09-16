#include "BackProp.h"

#include <iostream>
#include <vector>

int main() {
    // NPC Definition:
    // Adaptive Wilderness Survival Hunter

    // Input layer (6 neurons), all normalized to [0.0, 1.0]:
    // 0: Hunger           (0.0 = full, 1.0 = starving)
    // 1: Energy           (0.0 = exhausted, 1.0 = fully rested)
    // 2: Prey Availability(0.0 = none, 1.0 = abundant)
    // 3: Threat Level     (0.0 = safe, 1.0 = severe danger)
    // 4: Weather Severity (0.0 = clear, 1.0 = storm)
    // 5: Shelter Proximity(0.0 = far away, 1.0 = very near)

    // Output layer (4 neurons, one-hot encoded):
    // 0: Hunt
    // 1: Eat
    // 2: TakeCover
    // 3: BeVigilant

    // Training set:
    // Each sample represents a game-state pattern that the network should learn.

    std::vector<std::vector<double>> trainingInputs = {
        // Hunger, Energy, Prey, Threat, Weather, Shelter
        {0.95, 0.70, 0.80, 0.10, 0.10, 0.20}, // very hungry, prey available, low danger
        {0.90, 0.30, 0.10, 0.20, 0.20, 0.30}, // very hungry, low energy, little prey
        {0.20, 0.80, 0.90, 0.10, 0.10, 0.20}, // not hungry, high energy, rich prey
        {0.15, 0.60, 0.20, 0.75, 0.30, 0.40}, // low hunger, rising threat
        {0.40, 0.70, 0.60, 0.20, 0.15, 0.20}, // moderate hunger, favorable hunt state
        {0.80, 0.20, 0.20, 0.15, 0.20, 0.80}, // hungry, tired, shelter near
        {0.35, 0.60, 0.30, 0.85, 0.90, 0.95}, // severe threat and severe weather
        {0.25, 0.75, 0.40, 0.55, 0.50, 0.50}, // uncertain conditions
        {0.60, 0.85, 0.85, 0.15, 0.10, 0.20}, // hungry enough, strong hunting opportunity
        {0.75, 0.40, 0.70, 0.65, 0.35, 0.30}, // hunger and prey, but danger exists
        {0.10, 0.90, 0.10, 0.20, 0.10, 0.30}, // safe, not hungry, little reason to act
        {0.55, 0.50, 0.15, 0.80, 0.70, 0.90}, // danger dominates, shelter near
        {0.85, 0.80, 0.95, 0.05, 0.05, 0.10}, // ideal hunt conditions
        {0.92, 0.25, 0.05, 0.10, 0.20, 0.85}, // starving, tired, shelter close
        {0.30, 0.70, 0.50, 0.60, 0.20, 0.30}, // medium tension, may stay alert
        {0.45, 0.65, 0.75, 0.30, 0.20, 0.25}  // balanced but prey-rich situation
    };

    std::vector<std::vector<double>> trainingTargets = {
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 1.0, 0.0, 0.0}, // Eat
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 0.0, 0.0, 1.0}, // BeVigilant
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 1.0, 0.0, 0.0}, // Eat
        {0.0, 0.0, 1.0, 0.0}, // TakeCover
        {0.0, 0.0, 0.0, 1.0}, // BeVigilant
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 0.0, 0.0, 1.0}, // BeVigilant
        {0.0, 0.0, 0.0, 1.0}, // BeVigilant
        {0.0, 0.0, 1.0, 0.0}, // TakeCover
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 1.0, 0.0, 0.0}, // Eat
        {0.0, 0.0, 0.0, 1.0}, // BeVigilant
        {1.0, 0.0, 0.0, 0.0}  // Hunt
    };

    // Test samples:
    // These are not copied from the training set and are used to examine generalization.

    std::vector<std::vector<double>> testInputs = {
        {0.88, 0.72, 0.85, 0.10, 0.12, 0.15}, // should lean Hunt
        {0.93, 0.22, 0.12, 0.18, 0.25, 0.88}, // should lean Eat
        {0.28, 0.70, 0.30, 0.82, 0.75, 0.92}, // should lean TakeCover
        {0.35, 0.78, 0.45, 0.58, 0.30, 0.35}, // should lean BeVigilant
        {0.70, 0.82, 0.90, 0.12, 0.08, 0.12}, // should lean Hunt
        {0.18, 0.88, 0.15, 0.18, 0.12, 0.25}  // should lean BeVigilant
    };

    std::vector<std::vector<double>> testTargets = {
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 1.0, 0.0, 0.0}, // Eat
        {0.0, 0.0, 1.0, 0.0}, // TakeCover
        {0.0, 0.0, 0.0, 1.0}, // BeVigilant
        {1.0, 0.0, 0.0, 0.0}, // Hunt
        {0.0, 0.0, 0.0, 1.0}  // BeVigilant
    };

    int inputSize = 6;
    int hiddenSize = 6;
    int outputSize = 4;

    BackProp neuralNet(inputSize, hiddenSize, outputSize);

    int epochs = 2500;
    double learningRate = 0.15;

    std::cout << "Training neural network...\n";
    neuralNet.train(trainingInputs, trainingTargets, epochs, learningRate);
    std::cout << "Training completed.\n";

    std::cout << "\nFinal training accuracy: "
        << neuralNet.evaluateAccuracy(trainingInputs, trainingTargets)
        << "%\n";

    neuralNet.test(testInputs, testTargets);

    return 0;
}