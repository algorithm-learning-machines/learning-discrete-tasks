--------------------------------------------------------------------------------
-- File containing model definition --------------------------------------------
--------------------------------------------------------------------------------
require "nn"
require "rnn"
require "nngraph"
shift_learn = require("models.shift_learn")
custom_sharpeners = require("models.custom_sharpeners")
local memoryModel = {}

--------------------------------------------------------------------------------
-- !! Order of modules at end
-- [initialMem, input, prevAddrWrite] -> [finMem, addrCalc, p, pNRAM]
--------------------------------------------------------------------------------
function memoryModel.create(opt, addressReader, addressWriter, valueWriter)
   local vectorSize = tonumber(opt.vectorSize)
   local memSize = tonumber(opt.memorySize)
   local inputSize = 0
   if not opt.noInput then
      inputSize = tonumber(opt.inputSize)
   end
   local dummyInput = nn.Identity()()
   local RNN_steps = 5 --TODO add command line param
   local flatMemSize = memSize * vectorSize

   -----------------------------------------------------------------------------
   -- All input is given as one flat Tensor, we shall split it with Narrow
   -----------------------------------------------------------------------------
   local allInput = nn.Identity()()

   local flatMem
   local initialMem

   if not opt.noInput then
      --------------------------------------------------------------------------
      -- Extract flattened memory
      --------------------------------------------------------------------------
      flatMem = nn.Narrow(2,1,flatMemSize)(allInput)
   else
      flatMem = allInput
   end


   -----------------------------------------------------------------------------
   --  HACK: To avoid rewriting, left extra reshaper here for memory
   -----------------------------------------------------------------------------
   initialMem = nn.Reshape(memSize, vectorSize)(flatMem)
   --------------------------------------------------------------------------


   -----------------------------------------------------------------------------
   -- Input
   -----------------------------------------------------------------------------
   local input
   if not opt.noInput then
      input = nn.Narrow(2, flatMemSize + 1, inputSize)(allInput)
   end
   ----------------------------------------------------------------------------
  
   -----------------------------------------------------------------------------
   -- Previous write address
   -----------------------------------------------------------------------------
   local prevWriteAddress = nn.Identity()()

   ----------------------------------------------------------------------------
   --  Address Encoder
   ----------------------------------------------------------------------------
   local reshapedMem = nil
   if not addressReader then
      reshapedMem = nn.Reshape(memSize * vectorSize)(initialMem)
   end

   local AR = nn.GRU
   params = {memSize * vectorSize, memSize, RNN_steps}
   linkedNode = reshapedMem

   if addressReader then
      AR = addressReader
      params = {memSize}
      linkedNode = {prevWriteAddress}
   end

   local enc = AR(unpack(params))(linkedNode)
   --local address = nn.MulSoftMax()(enc)
    local address = nn.Identity()(enc)
   -----------------------------------------------------------------------------


   -----------------------------------------------------------------------------
   -- Value Extractor
   -----------------------------------------------------------------------------

   local addressTransp = nn.Reshape(1, memSize)(address)
   local value = nn.MM()({addressTransp, initialMem})     -- extract memory line
   -----------------------------------------------------------------------------

   -----------------------------------------------------------------------------
   -- Next address calculator
   -----------------------------------------------------------------------------
   local reshapedValue = nn.Squeeze(1)(value)             -- a line should be 1D
   local inputValAddr
   local inputVal
   local inputAddr
   if opt.separateValAddr then                -- separate value and adress paths
      if opt.noInput then
         inputVal = reshapedValue
         inputAddr = address
      else
         inputVal = nn.JoinTable(2)({input, reshapedValue})
         inputAddr = nn.JoinTable(2)({input, address})
      end
   else                                          -- cross value and adress paths
      if not opt.noInput then                           -- TODO: invert if cases
         local auxJoin = nn.JoinTable(2)({input, address})
         inputVal = nn.JoinTable(2)({auxJoin, reshapedValue})
         inputAddr = inputVal
      else
         inputVal = nn.JoinTable(2)({address, reshapedValue})
         inputAddr = inputVal
      end
   end

   local AW = nn.GRU

   params = {inputSize + memSize, memSize, RNN_steps}
   if not opt.separateValAddr then
      params = {inputSize + memSize + vectorSize, memSize, RNN_steps}
   end

   linkedNode = inputAddr
   if addressWriter then
      AW = addressWriter
      params = {memSize}
   end

   local addrCalc = AW(unpack(params))(linkedNode)
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   -- Next value calculator
   ----------------------------------------------------------------------------

   local VW = nn.GRU
   params = {inputSize + vectorSize, vectorSize, RNN_steps}

   if not opt.separateValAddr then
      params = {inputSize + memSize + vectorSize, vectorSize, RNN_steps}
   end

   linkedNode = inputVal
   if valueWriter then
      VW = valueWriter
      params = {}
   end

   local valueCalc = VW(unpack(params))(linkedNode)
   ----------------------------------------------------------------------------

   ----------------------------------------------------------------------------
   -- Memory Calculator
   ----------------------------------------------------------------------------

   -- adder
   local resizeValueCalc = nn.Reshape(1, vectorSize)(valueCalc)
   local resizeAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
   local adder = nn.MM()({resizeAddrCalc, resizeValueCalc})

   -- eraser
   local addrCalcTransp = nn.Reshape(1, memSize)(addrCalc)
   local AT_M_t_1 =  nn.MM()({addrCalcTransp, initialMem})
   local resAddrCalc = nn.Reshape(memSize, 1)(addrCalc)
   local AAT_M_t_1 = nn.MM()({resAddrCalc, AT_M_t_1})

   -- memory update
   local memEraser = nn.CSubTable()({initialMem, AAT_M_t_1})
   local finMem = nn.CAddTable()({memEraser, adder})
   ----------------------------------------------------------------------------

   in_dict = {}
   out_dict = {}
   in_dict[#in_dict + 1] = allInput 
   out_dict[#out_dict + 1] = finMem

   if addressReader then -- add back address to input
      in_dict[#in_dict + 1] = prevWriteAddress
      out_dict[#out_dict + 1] = addrCalc
   end
   
   if opt.outputLine then
      out_dict[#out_dict + 1] = valueCalc -- add an output line to the network
   end
   
   if opt.noProb then
      return nn.gModule(in_dict, out_dict)
   end

   -----------------------------------------------------------------------------
   -- Probability calculator
   -----------------------------------------------------------------------------
   local addrValCalc = nn.JoinTable(1)({addrCalc, valueCalc})
   local allInOne = nn.JoinTable(1)({addrValCalc, reshapedMem})

   -- TODO maybe this could be generalized as well
   local h1 = nn.Linear(vectorSize + memSize + memSize * vectorSize, 10)
   (allInOne) -- hidden layer

   local p = nn.Sigmoid()(nn.Linear(10, 1)(nn.Sigmoid()(nn.Linear(10, 10)(
   nn.Sigmoid()(h1)))))

   out_dict[#out_dict + 1] = p
   -----------------------------------------------------------------------------
   -- NRAM probability calculator
   -----------------------------------------------------------------------------

   if opt.NRAMProb then
      local prevDelta = nn.Identity()()
      local pNRAM = nn.MM()({nn.Reshape(1,1)(p), nn.Reshape(1,1)(prevDelta)})
      out_dict[#out_dict + 1] = pNRAM
      in_dict[#in_dict + 1] = prevDelta
   end
   ----------------------------------------------------------------------------

   return nn.gModule(in_dict, out_dict)
end

--------------------------------------------------------------------------------
-- Save model to file
-- Specify overWrite = true if you wish to overwrite an existent file
--------------------------------------------------------------------------------
function memoryModel.saveModel(model, fileName, overWrite)

   --TODO remove hardcoding
   if fileName == nil then
      fileName = "autosave.model"
   end
   if (path.exists(fileName) and overWrite == false) then
      print("file "..fileName.." already exists, overWrite option not specified. aborting.")
      return false
   end
   torch.save(fileName, model)
   print("Saved model!")

   return true
end

--------------------------------------------------------------------------------
-- Load a model from a file
--------------------------------------------------------------------------------
function memoryModel.loadModel(fileName)

   if not path.exists(fileName) then
      print("file "..fileName.." does not exist. Create it first before loading something from it")
      return nil
   end
   model = torch.load(fileName)

   return model
end

function cloneModel(model)
   local params, gradParams
   if model.parameters then
      params, gradParams = model:parameters()
      if params == nil then
         params = {}
      end
   end
   local paramsNoGrad
   if model.parametersNoGrad then
      paramsNoGrad = model:parametersNoGrad()
   end
   local mem = torch.MemoryFile("w"):binary()
   mem:writeObject(model)

   local reader = torch.MemoryFile(mem:storage(), "r"):binary()
   local clone = reader:readObject()
   reader:close()

   if model.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
         --Sets reference to model's parameters
         cloneParams[i]:set(params[i])
         cloneGradParams[i]:set(gradParams[i])

      end
      if paramsNoGrad then
         cloneParamsNoGrad = clone:parametersNoGrad()
         for i =1,#paramsNoGrad do
            ---- Sets reference to model's parameters
            cloneParamsNoGrad[i]:set(paramsNoGrad[i])
         end
      end
   end
   collectgarbage()
   mem:close()
   return clone
end

function memoryModel:cloneModel()
   return cloneModel(self)
end


function memoryModel.createMyModel(task, opt)
  addressReader = shift_learn.createWrapper
  addressWriter = shift_learn.createWrapper
  valueWriter = nn.Identity
  opt.separateValAddr = true 
  opt.noInput = true
  opt.noProb = true
  opt.simplified = true 
  opt.supervised = true
  return memoryModel.create(opt, addressReader, addressWriter, valueWriter)

end

return memoryModel
