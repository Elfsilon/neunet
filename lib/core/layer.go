package core

import (
	"fmt"
)

// Layer ...
type Layer struct {
	Weights Matrix `json:"weights"`
	Biases  Vector `json:"biases"`
	Neurons []Neuron `json:"neurons"`
}

func (l *Layer) String() string {
	var neurons string
	for _, n := range l.Neurons {
		neurons += fmt.Sprintf("    %v", n.String())
	}
	return fmt.Sprintf("Layer config:\n  Weights:\n    %v  Biases:\n    %v  Neurons:\n%v\n", l.Weights.String(), l.Biases.String(), neurons)
}
