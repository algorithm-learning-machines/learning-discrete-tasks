locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")
require("models.lstm_classifier")
require("tasks.doom_clock")

local dc = DoomClock()
local lc = LSTMClassifier(dc)

local X, T, F, L = dc:updateBatch()

lc:forward(X, L[1])
