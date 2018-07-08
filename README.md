# Board games AI

The goal of this project is to create on agent that can, given a board game rules, learn to play it
on expert level by cycles of self playing and model retraining, i.e. replicating an AlphaZero algorithm.

## Dependencies

* Python 3.5
* Tensorflow
* Keras

## Usage

The program can run in two modes: learning and custom play. Learning mode is for training the model
while custom play mode is for testing the model.

1. To run existing example choose 'Learn' build configuration or run `$ python3 main.py`. It will
start training a new model to play the 'four' game. Learning process will continue indefinitely
until manually interrupted.
2. To test the model choose 'Custom play' build configuration or run `$ python3 main.py 1`. This
will run an interactive playing session between players defined in `play_custom()` function.
Each player can be either human or AI.

Things worth noting about training process:
* A memory cache is saved each cycle and can be reused along with the model in future training
sessions
* A new version of the model is saved each time it succeeds its predecessor and can be reused in
future training sessions
* Model hyper-parameters and training parameters can be customized in `config.py`
* You can find the progress graph, logs, saved memory cache and model in `temp/` directory

## Adding a new game

Use `games/four/` directory as a reference. You'll have to implement `IGame` and `IGameState`
interfaces for the game rules, `INeuralNet` interface for the model, and `IPlayer` for the agent.
Let's go through each of them briefly.

* `IGame` is a static utility class. It contains a method for retrieving an initial game state and
a method for building symmetries of the board state. If there are no symmetries then it must return
a list containing a single (board state, pi) tuple. It also contain static game data like board
size and action space size.
* `IGameState` represents a current state of the game and contains methods responsible for making
turns and retrieving various state data.
* `INeuralNet` is responsible for creating the model, training it on given data, and inferencing.
* `IPlayer` represents an agent. Add your agent class to `players.py` and use `player_builder()`
factory for creating instances.

## Exporting a model

Configure the `utils/export.py` by setting the variables at the top of the file then run it. The
`MODEL_VERSION` and `DIR_NUMBER` use the same conventions as described in `utils/paths.py`.

## License

MIT License

Copyright (c) 2018 Alex Baryzhikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
