package core

import (
	"errors"
)

// NetConfig ...
type NetConfig struct {
	LayersConfig   []int
	ActivationFunc string
	CostFunc       string
	LearningRate   float64
}

func (c *NetConfig) validate() error {
	if c.LearningRate == 0 {
		return errors.New("Learning rate must be set and greater then 0")
	}
	if len(c.LayersConfig) < 2 {
		return errors.New("There must be 2 layers at least in the network")
	}

	return nil
}
