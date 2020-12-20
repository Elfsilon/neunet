package utils

import (
	"math/rand"
)

// RandFloat ...
func RandFloat(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}
