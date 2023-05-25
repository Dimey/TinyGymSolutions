# TinyGymSolutions

Welcome to TinyGymSolutions! This repository is a collection of minimalist, single-file reinforcement learning solutions for various [OpenAI Gym](https://gymnasium.farama.org) environments. 

The aim is to explore less orthodox solutions and paradigms, all while keeping the codebase small and understandable. Each solution file is a standalone script (~100 loc) that you can run to train and evaluate an agent in a specific Gym environment. NumPy only!

## Prerequisites

All you need to run these solutions is Python and Swig (install via `brew install swig` in your terminal), and the following Python libraries:
- numpy
- gym

You can install these prerequisites using pip:
`pip install numpy 'gymnasium[all]'`

## Project Structure

Each Gym environment has its own directory, and within each environment's directory, there are subdirectories for each algorithm applied to that environment.

## Usage

To run a solution, navigate to the solution's directory and run the .py file:
`python lunarlander_mlp_pgpe.py`

Happy Exploring!
