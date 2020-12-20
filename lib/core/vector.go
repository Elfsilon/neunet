package core

import (
	"fmt"
)

// Vector ...
type Vector []float64

func (v *Vector) add(v2 Vector) Vector {
	res := make(Vector, len(v2))
	for i, value := range *v {
		res[i] = value + v2[i]
	}
	return res
}

func (v *Vector) String() string {
	return fmt.Sprintf("Vector: %v\n", *v)
}
