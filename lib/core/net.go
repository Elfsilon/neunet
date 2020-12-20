package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/Elfsilon/neunet/lib/core/fun"
	"github.com/Elfsilon/neunet/lib/core/fun/activation"
	"github.com/Elfsilon/neunet/lib/core/fun/cost"
	"github.com/Elfsilon/neunet/lib/utils"
)

func selectActivationFunc(name string) fun.ActivationFunction {
	switch name {
	case "sigmoid":
		return activation.NewSigmoid()
	}
	return activation.NewSigmoid()
}

func selectCostFunc(name string) fun.CostFunction {
	switch name {
	case "mse":
		return cost.NewMSE()
	}
	return cost.NewMSE()
}

// NewNet ...
func NewNet(config *NetConfig) (*Net, error) {
	err := config.validate()
	if err != nil {
		return nil, err
	}

	fmt.Println("Start configuring the net with parameters:")
	fmt.Printf("Activation function: %v\n", config.ActivationFunc)
	fmt.Printf("Loss function: %v\n", config.CostFunc)
	fmt.Printf("Learning rate: %v\n", config.LearningRate)
	fmt.Printf("Layers configuration: %v\n\n", config.LayersConfig)

	net := Net{
		LearningRate:       config.LearningRate,
		ActivationFunc:     selectActivationFunc(strings.ToLower(config.ActivationFunc)),
		CostFunc:           selectCostFunc(strings.ToLower(config.CostFunc)),
		ActivationFuncName: config.ActivationFunc,
		CostFuncName:       config.CostFunc,
	}
	net.init(config.LayersConfig)

	return &net, nil
}

// Net ...
type Net struct {
	Layers             []Layer                `json:"layers"`
	LearningRate       float64                `json:"learning-rate"`
	ActivationFuncName string                 `json:"activation-function-name"`
	CostFuncName       string                 `json:"loss-function-name"`
	ActivationFunc     fun.ActivationFunction `json:"-"`
	CostFunc           fun.CostFunction       `json:"-"`
	Loss               float64                `json:"loss"`
	// ActivationFunc     fun.ActivationFunction `json:"activation-function"`
	// CostFunc           fun.CostFunction       `json:"loss-function"`
}

func (n *Net) init(lconf []int) {
	for curLayerIndex := 0; curLayerIndex < len(lconf); curLayerIndex++ {
		l := Layer{}

		// If not output layer
		if curLayerIndex < len(lconf)-1 {
			// Init weights if not output layer
			l.Weights = make([][]float64, lconf[curLayerIndex+1])
			for i := 0; i < len(l.Weights); i++ {
				l.Weights[i] = make([]float64, lconf[curLayerIndex])
				// Randomize weight values
				for j := 0; j < len(l.Weights[i]); j++ {
					l.Weights[i][j] = utils.RandFloat(0, 1)
				}
			}

			// Init biases if not output layer
			l.Biases = make([]float64, lconf[curLayerIndex+1])
			for i := 0; i < len(l.Biases); i++ {
				// Randomize biases
				l.Biases[i] = utils.RandFloat(0, 1)
			}
		}

		// Init neurons
		for i := 0; i < lconf[curLayerIndex]; i++ {
			l.Neurons = append(l.Neurons, Neuron{})
		}

		n.Layers = append(n.Layers, l)
	}
}

func (n *Net) initInputs(X []float64) {
	for i := range n.Layers[0].Neurons {
		n.Layers[0].Neurons[i].Value = X[i]
	}
}

func (n *Net) forwards(X []float64) {
	n.initInputs(X)
	for i := 1; i < len(n.Layers); i++ {
		mul := n.Layers[i-1].Weights.dotNeurons(n.Layers[i-1].Neurons)
		fin := mul.add(n.Layers[i-1].Biases)
		for j, calculated := range fin {
			n.Layers[i].Neurons[j].Value = calculated
			n.Layers[i].Neurons[j].activate(n.ActivationFunc)
		}
	}
}

func (n *Net) computeErrors(targets []float64, activationFunc fun.ActivationFunction, costFunc fun.CostFunction) {
	var totalErr float64 = 0
	// Error derivatives of the output neurons
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		err := costFunc.Der(targets[i], neuron.Value) * activationFunc.Der(neuron.Value)
		n.Layers[len(n.Layers)-1].Neurons[i].Err = err

		totalErr += costFunc.Val(targets[i], neuron.Value)
	}
	totalErr /= float64(2)
	n.Loss = totalErr
}

func (n *Net) backwards(learningRate float64, activationFunc fun.ActivationFunction) {
	for i := len(n.Layers) - 1; i > 0; i-- {
		for j, curLayerNeuron := range n.Layers[i].Neurons {
			// Error computation
			var computedErr float64 = 0
			if i == len(n.Layers)-1 {
				// Output error
				computedErr = curLayerNeuron.Err
			} else {
				// Hidden error
				for k, nextLayerNeuron := range n.Layers[i+1].Neurons {
					computedErr += nextLayerNeuron.Err * n.Layers[i].Weights[k][j]
				}
				computedErr *= activationFunc.Der(curLayerNeuron.Value)
				n.Layers[i].Neurons[j].Err = computedErr
			}

			// Adjusting
			// Computes delta bias
			db := learningRate * computedErr
			for k, prevLayerNeuron := range n.Layers[i-1].Neurons {
				// Computes delta weight
				dw := db * prevLayerNeuron.Value
				// Adjust weights
				n.Layers[i-1].Weights[j][k] += dw
			}
			// Adjust bias
			n.Layers[i-1].Biases[j] += db
		}
	}
}

func (n *Net) computeAccuracy(X [][]float64, Y [][]float64) float64 {
	// Init confusion matrix
	out := len(n.Layers[len(n.Layers)-1].Neurons)
	confusionMatrix := make([][]int, out)
	for i := range confusionMatrix {
		confusionMatrix[i] = make([]int, out)
	}

	// Computes confusion matrix
	for i, input := range X {
		p, _ := n.Predict(input)

		var targetIndex int = -1
		var targetMax float64 = -99999
		for j, target := range Y[i] {
			if target > targetMax {
				targetMax = target
				targetIndex = j
			}
		}

		confusionMatrix[p.Index][targetIndex]++
	}

	// Computes prescision
	var precision float64 = 0
	for i, row := range confusionMatrix {
		sum := 0
		for _, val := range row {
			sum += val
		}
		if sum != 0 {
			precision += float64(confusionMatrix[i][i]) / float64(sum)
		}
	}
	precision /= float64(len(confusionMatrix))

	// Computes recall
	var recall float64 = 0
	for j := 0; j < out; j++ {
		sum := 0
		for i := 0; i < out; i++ {
			sum += confusionMatrix[i][j]
		}
		if sum != 0 {
			recall += float64(confusionMatrix[j][j]) / float64(sum)
		}
	}
	recall /= float64(len(confusionMatrix))

	// Computes F-measure
	Fmeasure := (2 * precision * recall) / (precision + recall)
	return Fmeasure
}

// Fit ...
func (n *Net) Fit(X [][]float64, Y [][]float64, epoch, batch int) error {
	if len(X[0]) != len(n.Layers[0].Neurons) {
		return errors.New("Shapes of the input layer and passed X doesn't match")
	}
	if len(Y[0]) != len(n.Layers[len(n.Layers)-1].Neurons) {
		return errors.New("Shapes of the output layer and passed Y doesn't match")
	}
	if len(X) != len(Y) {
		return errors.New("Lengths of the passed X and Y doesn't match")
	}

	rand.Seed(time.Now().UnixNano())
	if batch == 0 {
		batch = len(X)
	}

	fmt.Println("Fit starting")
	for e := 0; e < epoch; e++ {
		for i := 0; i < len(Y); i++ {
			setIndex := rand.Intn(batch)
			n.forwards(X[setIndex])
			n.computeErrors(Y[setIndex], n.ActivationFunc, n.CostFunc)
			n.backwards(n.LearningRate, n.ActivationFunc)
		}

		if (e+1)%50 == 0 {
			fmt.Printf("Epoch %v/%v, Loss: %v\n", e+1, epoch, n.Loss)
		}
	}

	accuracy := n.computeAccuracy(X, Y)
	fmt.Println("\nFit finished")
	fmt.Printf("F-measure of model (accuracy): %.4v%%\n", accuracy*100)

	return nil
}

// Predicted ...
type Predicted struct {
	Value float64
	Index int
}

// Predict returns the number of activated output neuron and it's value
func (n *Net) Predict(X []float64) (Predicted, error) {
	p := Predicted{}
	if len(X) != len(n.Layers[0].Neurons) {
		return p, errors.New("Shapes of the input layer and passed X doesn't match")
	}

	n.forwards(X)
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		if neuron.Value > p.Value {
			p.Value = neuron.Value
			p.Index = i
		}
	}
	return p, nil
}

// GetLayer ...
func (n *Net) GetLayer(index int) Layer {
	return n.Layers[index]
}

// GetNeuron ...
func (n *Net) GetNeuron(layer, index int) Neuron {
	return n.Layers[layer].Neurons[index]
}

// GetLayerNeurons ...
func (n *Net) GetLayerNeurons(index int) []Neuron {
	return n.Layers[index].Neurons
}

// Save ...
func (n *Net) Save(path string) error {
	netConfig, err := json.MarshalIndent(n, "", " ")
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(path, netConfig, 0644)
	if err != nil {
		return err
	}

	return nil
}

// Load ...
func Load(path string) (*Net, error) {
	configJSON, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer configJSON.Close()

	bytesJSON, err := ioutil.ReadAll(configJSON)
	if err != nil {
		return nil, err
	}

	net := Net{}
	json.Unmarshal(bytesJSON, &net)

	net.ActivationFunc = selectActivationFunc(strings.ToLower(net.ActivationFuncName))
	net.CostFunc = selectCostFunc(strings.ToLower(net.CostFuncName))

	return &net, nil
}

func (n *Net) String() string {
	var s string = "Neural net config:%v\n"
	for _, l := range n.Layers {
		s += l.String()
	}
	return s
}
