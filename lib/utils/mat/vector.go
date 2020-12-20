package mat

import (
	"fmt"
)

// Vector ...
type Vector []float64

func (v *Vector) String() string {
	return fmt.Sprintf("Vector: %v\n", *v)
}
