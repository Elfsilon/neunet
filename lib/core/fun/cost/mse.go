package cost

import (
	"math"
)

// NewMSE ...
func NewMSE() *MSE {
	return &MSE{}
}

// MSE ...
type MSE struct{}

// Val ...
func (s *MSE) Val(target, value float64) float64 {
	return math.Pow(target-value, 2)
}

// Der ...
func (s *MSE) Der(target, value float64) float64 {
	return target - value
}
