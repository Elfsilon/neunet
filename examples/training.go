package examples

import (
	"fmt"

	"github.com/Elfsilon/neunet/lib/core"
)

// Training ...
func Training() {
	config := core.NetConfig{
		LayersConfig:   []int{3, 8, 2}, // 3 layers, 3 neuron on input, 8 on hidden, 2 on output
		ActivationFunc: "Sigmoid",      // At this moment only
		CostFunc:       "MSE",          // sigmoid and MSE available
		LearningRate:   0.015,
	}

	// Creating network object
	net, err := core.NewNet(&config)
	if err != nil {
		fmt.Println(err)
	}

	// Training dataset
	X := [][]float64{
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 0},
		{0, 1, 1},
		{1, 1, 0},
	}

	// Training answers
	Y := [][]float64{
		{1, 0},
		{0, 1},
		{1, 0},
		{1, 0},
		{1, 0},
	}

	// Fit with X, Y, epoch=10000, batch_size=0 (0 = all data)
	err = net.Fit(X, Y, 10000, 0)
	if err != nil {
		fmt.Println(err)
	}

	// Saving net configuration in the json file (Only json available now)
	err = net.Save("config.json")
	if err != nil {
		fmt.Println(err)
	}
}
