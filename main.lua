locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("models.lstm_classifier")
require("models.rnn_lstm_wrapper")
require("tasks.doom_clock")

local dc = DoomClock()

local X, T, F, L = dc:updateBatch()

local lstm  = LSTMWrapper(X[1][1]:nElement(),X[1][1]:nElement(),{})

local inp = X[1][1]:view(X[1][1]:nElement(), 1)
lstm:forward(inp:reshape(inp:nElement()))
