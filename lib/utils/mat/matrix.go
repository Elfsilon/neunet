package mat

import (
	"fmt"

	"github.com/Elfsilon/neunet/lib/core"
)

// Matrix ...
type Matrix [][]float64

func (m *Matrix) dotNeurons(n []core.Neuron) Vector {

}

func (m *Matrix) String() string {
	var s string = "Matrix:\n"
	for _, row := range *m {
		s += fmt.Sprintf("    %v\n", row)
	}
	return s
}
