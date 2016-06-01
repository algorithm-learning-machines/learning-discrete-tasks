require("nn")
require("nngraph")
require("string.color")

--------------------------------------------------------------------------------
-- Creates a LSTM cell with an internal vector of size @stateSize.
-- External output has size @extSize.
-- Specific transfer functions might be defined for gates.
--------------------------------------------------------------------------------

local LSTM = torch.class("LSTM")

function LSTM:__init(extSize, stateSize, opt)

   opt = opt or {}

   self.noAsserts = opt.noAsserts or false
   self.verbose = opt.verbose or false

   -- Use --noAsserts not to check size
   if not self.noAsserts then
      assert(tonumber(stateSize) > 0)
      assert(tonumber(extSize) > 0)
   end

   -- Should this LSTM cell be prepared to process batches
   local batchSize = opt.batchSize == nil and 1 or opt.batchSize
   if batchSize >= 1 then
      self.fsDim = 2
   else
      self.fsDim = 1
   end

   self.stateSize = stateSize
   self.extSize = extSize

   self.fGate = opt.gateTransfer or nn.Sigmoid
   self.fInput = opt.inputTransfer or nn.Tanh
   self.fState = opt.stateTransfer or nn.Tanh
   self.fOutput = opt.outputTransfer or nn.Tanh

end

function LSTM:buildUnit(params, gradParams)
   local stateSize = self.stateSize
   local extSize = self.extSize
   local fsDim = self.fsDim

   local fGate = self.fGate
   local fInput = self.fInput
   local fState = self.fState
   local fOutput = self.fOutput

   local inputSize = 2 * stateSize + extSize

   local prevState = nn.Identity()()
   local prevOutput = nn.Identity()()
   local extInput = nn.Identity()()

   local allInputs = nn.JoinTable(fsDim)({prevState, prevOutput, extInput})
   local allLinear = nn.Linear(inputSize, 3 * stateSize)(allInputs)

   local twoGates = fGate()(nn.Narrow(fsDim, 1, 2 * stateSize)(allLinear))
   local forgetGate = nn.Narrow(fsDim, 1, stateSize)(twoGates)
   local writeGate = nn.Narrow(fsDim, stateSize + 1, stateSize)(twoGates)

   local input = fInput()(nn.Narrow(fsDim, 2*stateSize+1, stateSize)(allLinear))
   local state = nn.CAddTable()({
         nn.CMulTable()({forgetGate, prevState}),
         nn.CMulTable()({writeGate, input})
   })
   local outInputs = nn.JoinTable(fsDim)({state, prevOutput, extInput})
   local outGate = fGate()(nn.Linear(inputSize, stateSize)(outInputs))
   local output = fOutput()(nn.CMulTable()({outGate, fState()(state)}))

   local lstm = nn.gModule({prevState, prevOutput, extInput}, {state, output})

   -- Tie parameters if params and gradParams are given
   if params and gradParams then
      local crtParams, crtGradParams = lstm:parameters()
      if not self.noAsserts then
         assert((#crtParams == #params) and (#crtGradParams == #gradParams))
      end
      for i = 1, #params do
         crtParams[i]:set(params[i])
      end
      for i = 1, #gradParams do
         crtGradParams[i]:set(gradParams[i])
      end
   end

   if self.verbose then
      local message = "New LSTM cell of size " .. stateSize
      if fsDim == 1 then
         message = message .. " for individual examples"
      else
         message = message .. " for batches"
      end
      if params then
         message = message .. " with given parameters."
      else
         message = message .. " with free parameters."
      end

      print(string.format("[LSTM] "):color("blue") ..
               string.format(message):color("green"))
   end

   return lstm
end


function LSTM:showParameters(unit, zoom)
   if not image then require("image") end
   zoom = zoom == nil and 100 or zoom

   local crtParams, _ = unit:parameters()

   local forgetGateParam = crtParams[1][{{1, self.stateSize},{}}]
   local writeGateParam = crtParams[1][{{self.stateSize+1,2*self.stateSize},{}}]
   local inputParam = crtParams[1][{{2*self.stateSize+1, 3*self.stateSize},{}}]
   local outGateParam = crtParams[3][{{1, self.stateSize},{}}]

   self.forgetGateParamsWin = image.display{
      image = forgetGateParam, win = self.forgetGateParamsWin,
      zoom = zoom, legend = 'Forget gate'
   }

   self.writeGateParamsWin = image.display{
      image = writeGateParam, win = self.writeGateParamsWin,
      zoom = zoom, legend = 'Write gate'
   }

   self.inputParamsWin = image.display{
      image = inputParam, win = self.inputParamsWin,
      zoom = zoom, legend = 'Input parameters'
   }

   self.outGateParamsWin = image.display{
      image = inputParam, win = self.outGateParamsWin,
      zoom = zoom, legend = 'Out gate'
   }
end
