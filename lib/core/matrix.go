package core

import (
	"fmt"
)

// NewMatrix ...
func NewMatrix(n, m int) Matrix {
	mat := make(Matrix, n)
	for i := range mat {
		mat[i] = make(Vector, m)
	}
	return mat
}

// Matrix ...
type Matrix [][]float64

func (m *Matrix) dotNeurons(n []Neuron) Vector {
	res := make(Vector, len(*m))
	for i, row := range *m {
		for j, w := range row {
			res[i] += w * n[j].Value
		}
	}
	return res
}

func (m *Matrix) String() string {
	var s string = "Matrix:\n"
	for _, row := range *m {
		s += fmt.Sprintf("    %v\n", row)
	}
	return s
}
