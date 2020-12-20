package examples

import (
	"fmt"

	"github.com/Elfsilon/neunet/lib/core"
)

// Loading ...
func Loading() {
	net, err := core.Load("config.json")
	if err != nil {
		fmt.Println(err)
	}

	// Shows current net configuration
	// fmt.Println(net.String())

	X := []float64{1, 1, 0}

	// guess - type of Prediction{Value, Index}
	// Value - value of activated output neuron
	// Index - it's index
	guess, err := net.Predict(X)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(guess.Index, guess.Value)
}
