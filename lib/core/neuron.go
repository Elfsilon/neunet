package core

import (
	"fmt"

	"github.com/Elfsilon/neunet/lib/core/fun"
)

// Neuron ...
type Neuron struct {
	Value float64 `json:"value"`
	Err   float64 `json:"error"`
}

func (n *Neuron) activate(f fun.ActivationFunction) {
	n.Value = f.Val(n.Value)
}

func (n *Neuron) String() string {
	return fmt.Sprintf("Neuron: {value: %v, error: %v}\n", n.Value, n.Err)
}
