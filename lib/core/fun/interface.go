package fun

// ActivationFunction ...
type ActivationFunction interface {
	Val(x float64) float64
	Der(x float64) float64
}

// CostFunction ...
type CostFunction interface {
	Val(target, value float64) float64
	Der(target, value float64) float64
}
