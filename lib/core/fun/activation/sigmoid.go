package activation

import (
	"math"
)

// NewSigmoid ...
func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

// Sigmoid ...
type Sigmoid struct{}

// Val ...
func (s *Sigmoid) Val(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

// Der ...
func (s *Sigmoid) Der(x float64) float64 {
	return x * (1 - x)
}
