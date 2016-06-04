--------------------------------------------------------------------------------
-- This files defines an abstract class that should be inherited by all classes
-- that implement a specific task.
--
-- When implementing a new task the following functions should be implemented:
--  * __init(opt)
--
--     i. It should first call Parent (this class)'s __init(opt)
--    ii. It should set up task specific values:
--            * inputsInfo
--            * outputsInfo
--            * targetAtEachStep / targetAtTheEnd
--            * overwrite trainMaxLength, testMaxLength, fixedLength if needed
--   iii. It might call Parent's __initTensors() to prepare batch tensors or to
--        generate the data sets
--    iv. Call Parent's __initCriterions() to prepare criterions for evaluation.
--
--  * __generateBatch(inputs, targets, flags, lengths, isTraining)
--
--------------------------------------------------------------------------------

require("torch")
require("nn")
require("string.color")

local Task = torch.class("Task")

function Task:__init(opt)

   opt = opt or {}

   self.verbose = opt.verbose or false
   self.noAsserts = opt.noAsserts or false

   self.onTheFly = opt.onTheFly or false

   self.trainMaxLength = opt.trainMaxLength or 20
   self.testMaxLength = opt.testMaxLength or (self.trainMaxLength * 2)

   self.fixedLength = opt.fixedLength or false
   self.batchSize = opt.batchSize == nil and 1 or opt.batchSize

   self.positive = opt.positive or 1
   self.negative = opt.negative or -1

   if not self.onTheFly then
      self.trainSize = opt.trainSize or 10000
      self.testSize = opt.testSize or 1000
      self.trainSize = self.trainSize - (self.trainSize % self.batchSize)
      self.testSize = self.testSize - (self.testSize % self.batchSize)
      self.trainIdx = 1
      self.testIdx = 1
   end

   self:message("Parsed generic task options.")
end

function Task:__initTensors()
   if not self.noAsserts then
      assert((self.inputsInfo ~= nil) and (self.outputsInfo ~= nil), "Ooops")
   end

   local bs = self.batchSize
   local trl = self.trainMaxLength
   local tsl = self.testMaxLength

   self.trainInputsBatch = {}
   self.testInputsBatch = {}
   self.trainTargetsBatch = {}
   self.trainTargetFlagsBatch = {}
   self.testTargetsBatch = {}
   self.testTargetFlagsBatch = {}

   if not self.onTheFly then

      --------------------------------------------------------------------------
      -- Tensors for train and test data sets

      local trs = self.trainSize
      local tss = self.testSize

      self.trainInputs = {}
      self.testInputs = {}

      for k, v in pairs(self.inputsInfo) do
         local ins = v.size
         self.trainInputs[k] = torch.Tensor(trl, trs, ins)       -- train inputs
         self.testInputs[k] = torch.Tensor(tsl, tss, ins)         -- test inputs
      end

      self.trainTargets = {}
      self.trainTargetFlags = {}
      self.testTargets = {}
      self.testTargetFlags = {}

      for k, v in pairs(self.outputsInfo) do
         local outs
         if v.type == "regression" or v.type == "binary" then
            outs = v.size
         elseif v.type == "one-hot" then
            outs = 1
         end

         if self.targetAtTheEnd then                                  -- targets
            self.trainTargets[k] = torch.Tensor(1, trs, outs)
            self.testTargets[k] = torch.Tensor(1, tss, outs)
         else
            self.trainTargets[k] = torch.Tensor(trl, trs, outs)
            self.testTargets[k] = torch.Tensor(tsl, tss, outs)
         end

         if not (self.targetAtTheEnd or self.targetAtEachStep) then     -- flags
            self.trainTargetFlags[k] = torch.ByteTensor(trl, trs)
            self.trainTargetFlags[k] = torch.ByteTensor(tsl, tss)
         end
      end

      if not self.fixedLength then                                    -- lengths
         self.trainLengths = torch.LongTensor(trs)
         self.testLengths = torch.LongTensor(tss)
      end

      for i = 1, (trs / bs) do
         local start = (i-1) * bs + 1
         local X, T, F = {}, {}, {}
         for k, _ in pairs(self.inputsInfo) do
            X[k] = self.trainInputs[k]:narrow(2, start, bs)
         end
         for k, _ in pairs(self.outputsInfo) do
            T[k] = self.trainTargets[k]:narrow(2, start, bs)
            if self.trainTargetFlagsBatch[k] then
               F[k] = self.trainTargetFlags[k]:narrow(2, start, bs)
            end
         end
         local L
         if self.trainLengths then
            L = self.trainLengths:narrow(1, start, bs)
         end
         self:__generateBatch(X, T, F, L, true)
      end

      for i = 1, (tss / bs) do
         local start = (i-1) * bs + 1
         local X, T, F = {}, {}, {}
         for k, v in pairs(self.inputsInfo) do
            X[k] = self.testInputs[k]:narrow(2, start, bs)
         end
         for k, v in pairs(self.outputsInfo) do
            T[k] = self.testTargets[k]:narrow(2, start, bs)
            if self.testTargetFlagsBatch[k] then
               F[k] = self.testTargetFlags[k]:narrow(2, start, bs)
            end
         end
         local L
         if self.testLengths then
            L = self.testLengths:narrow(1, start, bs)
         end
         self:__generateBatch(X, T, F, L, false)
      end


      self:message("Created a " .. self.trainSize .. " train set.")
      self:message("Created a " .. self.testSize .. " test set.")

   else

      --------------------------------------------------------------------------
      -- Tensors for train and test batches


      for k, v in pairs(self.inputsInfo) do
         self.trainInputsBatch[k] = torch.Tensor(trl, bs, v.size)
         self.testInputsBatch[k] = torch.Tensor(tsl, bs, v.size)
      end

      for k, v in pairs(self.outputsInfo) do
         if self.targetAtTheEnd then                                  -- targets
            self.trainTargetsBatch[k] = torch.Tensor(1, bs, v.size)
            self.testTargetsBatch[k] = torch.Tensor(1, bs, v.size)
         else
            self.trainTargetsBatch[k] = torch.Tensor(trl, bs, v.size)
            self.testTargetsBatch[k] = torch.Tensor(tsl, bs, v.size)
         end
         if not (self.targetAtTheEnd or self.targetAtEachStep) then     -- flags
            self.trainTargetFlagsBatch[k] = torch.ByteTensor(trl, bs)
            self.testTargetFlagsBatch[k] = torch.ByteTensor(tsl, bs)
         end
      end

      if not self.fixedLength then                                    -- lengths
         self.trainLengthsBatch = torch.LongTensor(bs)
         self.testLengthsBatch = torch.LongTensor(bs)
      end

   end

   self:message("Initialized tensors.")
end

function Task:__initCriterions()
   self.criterions = {}
   for k, v in pairs(self.outputsInfo) do
      if v.type == "regression" then
         self.criterions[k] = nn.MSECriterion()
      elseif v.type == "one-hot" then
         self.criterions[k] = nn.ClassNLLCriterion()
      elseif v.type == "binary" then
         self.criterions[k] = nn.BCECriterion()
      else
         assert(false, "Unknown output type")
      end
   end
end

function Task:__generateBatch(inputs, targets, flags, lengths, isTraining)
   assert(false, "This is a virtual method :)")
end

function Task:updateBatch(split)
   split = split == nil and "train" or split
   if split == "train" then
      if self.onTheFly then
         self:__generateBatch(
            self.trainInputsBatch,
            self.trainTargetsBatch,
            self.trainTargetFlagsBatch,
            self.trainLengthsBatch,
            true
         )
      else
         if self.trainIdx + self.batchSize > self.trainSize + 1 then
            self.trainIdx = 1
            self:message("Restarting training set.")
         end

         local i = self.trainIdx
         local bs = self.batchSize

         for k, _ in pairs(self.inputsInfo) do
            self.trainInputsBatch[k] = self.trainInputs[k]:narrow(2, i, bs)
         end
         for k, _ in pairs(self.outputsInfo) do
            self.trainTargetsBatch[k] = self.trainTargets[k]:narrow(2, i, bs)
            if not (self.targetAtEachStep or self.targetAtTheEnd) then
               self.trainTargetFlagsBatch[k] =
                  self.trainTargetFlags[k]:narrow(2, i, bs)
            end
         end
         if not self.fixedLength then
            self.trainLengthsBatch = self.trainLengths:narrow(1, i, bs)
         end

         self.trainIdx = self.trainIdx + self.batchSize
      end

      return self.trainInputsBatch,
             self.trainTargetsBatch,
             self.trainTargetFlagsBatch,
             self.trainLengthsBatch

   elseif split == "test" then
      if self.onTheFly then
         self:__generateBatch(
            self.testInputsBatch,
            self.testTargetsBatch,
            self.testTargetFlagsBatch,
            self.testLengthsBatch,
            false
         )
      else
         if self.testIdx + self.batchSize > self.testSize + 1 then
            self.testIdx = 1
            self:message("Restarting test set.")
         end

         local i = self.testIdx
         local bs = self.batchSize

         for k, _ in pairs(self.inputsInfo) do
            self.testInputsBatch[k] = self.testInputs[k]:narrow(2, i, bs)
         end
         for k, _ in pairs(self.outputsInfo) do
            self.testTargetsBatch = self.testTargets[k]:narrow(2, i, bs)
            if not (self.targetAtEachStep or self.targetAtTheEnd) then
               self.testTargetFlagsBatch[k] =
                  self.testTargetFlags[k]:narrow(2, i, bs)
            end
         end

         if not self.fixedLength then
            self.testLengthsBatch = self.testLengths:narrow(1, i, bs)
         end

         self.testIdx = self.testIdx + self.batchSize
      end

      return self.testInputsBatch,
             self.testTargetsBatch,
             self.testTargetFlagsBatch,
             self.testLengthsBatch

   else
      assert(false, "unknown split: " .. split)
   end
end

function Task:isEpochOver(split)
   split = split == nil and "train" or split
   if self.onTheFly then return false end
   if split == "train" then
      return self.trainIdx > self.trainSize
   elseif split == "test" then
      return self.testIdx > self.testSize
   else
      assert(false, "unknown split: " .. split)
   end
end

function Task:resetIndex(split)
   split = split == nil and "train" or split
   if split == "train" then
      self.trainIdx = 1
   elseif split == "test" then
      self.testIdx = 1
   else
      assert(false, "unknown split: " .. split)
   end
end

function Task:getInputsInfo()
   if not self.noAsserts then
      assert(self.inputsInfo ~= nil, "Missing inputs info.")
   end
   return self.inputsInfo
end

function Task:getOutputsInfo()
   if not self.noAsserts then
      assert(self.outputsInfo ~= nil, "Missing outputs info.")
   end
   return self.outputsInfo
end

function Task:__getTotalInputSize()
   local s = 0
   for _, v in pairs(self.inputsInfo) do
      s = s + v.size
   end
   return s
end

function Task:__getTotalOutputSize()
   local s = 0
   for _, v in pairs(self.outputsInfo) do
      if v.type ~= "one-hot" then
         s = s + v.size
      else
         s = s + 1
      end
   end
   return s
end

--------------------------------------------------------------------------------
-- Function used to evaluate a given batch against the target values
--------------------------------------------------------------------------------


function Task:evaluateBatch(output, targets, err)
   local threshold = self.negative + (self.positive - self.negative) / 2
   local toBinary = function(x)
      if x >= threshold then return 1 else return 0 end
   end
   err = err or {}
   if not self.noAsserts then
      for k, v in pairs(self.outputsInfo) do
         assert(output[k] ~= nil and targets[k] ~= nil, k .. " output missing.")
      end
   end
   for k, v in pairs(self.outputsInfo) do              -- go through all outputs
      err[k] = err[k] or {}
      local errInfo = err[k]
      if v.type == "regression" then
         errInfo.loss = (errInfo.loss or 0) +
            self.criterions[1]:forward(output[k], targets[k])
      elseif v.type == "one-hot" then
         local T = targets[k]
         local O = output[k]
         local _, Y = O:max(2)
         if not self.noAsserts then
            assert(T:size(1) == Y:size(1), "output size very bad!")
         end
         errInfo.correct = (errInfo.correct or 0) + Y:eq(T):sum()
         errInfo.n = (errInfo.n or 0) + Y:nElement()
         errInfo.loss = (errInfo.loss or 0) + self.criterions[k]:forward(O, T)
         errInfo.confMatrix = errInfo.confMatrix or
            torch.zeros(v.size, v.size):long()
         for i = 1, Y:size(1) do
            errInfo.confMatrix[{{T[i]},{Y[i]}}]:add(1)
         end
      elseif v.type == "binary" then
         local O = output[k]:clone():apply(toBinary)
         local T = lables[k]:clone():apply(toBinary)
         errInfo.correct = (errInfo.correct or 0) + O:eq(T):sum()
         errInfo.n = (errInfo.n or 0) + O:nElement()
         errInfo.loss = (errInfo.loss or 0) + self.criterions[k]:forward(O, T)
      else
         assert(false, "Unknown output type")
      end
   end
   return err
end

--------------------------------------------------------------------------------
-- Information about avaialble labels
--------------------------------------------------------------------------------

function Task:hasTargetAtEachStep()
   return self.targetAtEachStep or false
end

function Task:hasTargetAtTheEnd()
   return self.targetAtTheEnd or false
end


--------------------------------------------------------------------------------
-- Move task on CUDA
--------------------------------------------------------------------------------

function Task:cuda()
   require("cutorch")
   require("cunn")
   if self.onTheFly then
      -- Move the batch tensors to CUDA
      for k, _ in pairs(self.inputsInfo) do
         self.trainInputsBatch[k] = self.trainInputsBatch[k]:cuda()
         self.testInputsBatch[k] = self.testInputsBatch[k]:cuda()
      end
      for k, _ in pairs(self.outputsInfo) do
         self.trainTargetsBatch[k] = self.trainTargetsBatch[k]:cuda()
         self.testTargetsBatch[k] = self.testTargetsBatch[k]:cuda()
         if not (self.targetAtTheEnd or self.targetAtEachStep) then
            self.trainTargetFlagsBatch[k] = self.trainTargetFlagsBatch[k]:cuda()
            self.testTargetFlagsBatch[k] = self.testTargetFlagsBatch[k]:cuda()
         end
      end
      if not self.fixedLength then
         self.trainLengthsBatch = self.trainLengthsBatch:cuda()
         self.testLengthsBatch = self.testLengthsBatch:cuda()
      end
   else
      -- Move the whole dataset to CUDA
      for k,_ in pairs(self.inputsInfo) do
         self.trainInputs[k] = self.trainInputs[k]:cuda()
         self.testInputs[k] = self.testInputs[k]:cuda()
      end
      for k,_ in pairs(self.outputsInfo) do
         self.trainTargets = self.trainTargets:cuda()
         self.testTargets = self.testTargets:cuda()
         if not (self.targetAtTheEnd or self.targetAtEachStep) then
            self.trainTargetFlags = self.trainTargetFlags:cuda()
            self.testTargetFlags = self.testTargetFlags:cuda()
         end
      end

      if not self.fixedLength then
         self.trainLengths = self.trainLengths:cuda()
         self.testLengths = self.testLengths:cuda()
      end
   end


   for _, criterion in pairs(self.criterions) do
      criterion:cuda()
   end

   self.onCuda = true
   self:message("Moved to GPU using CUDA.")
end

--------------------------------------------------------------------------------
-- Generic function that displays the current batch
-- Use `qlua` when calling this function.
-- It concatenates in the dumbest way possible all inputs.
--------------------------------------------------------------------------------

function Task:displayCurrentBatch(split, zoom)
   if not image then
      require("image")
   end

   split = split == nil and "train" or split
   zoom = zoom or 50

   local bs = self.batchSize
   local seqLength, inputBatch, targetBatch

   if split == "train" then
      seqLength = self.trainMaxLength
      if not self.fixedLength then
         seqLength = self.trainLengthsBatch[1]
      end
      inputBatch = self.trainInputsBatch
      targetBatch = self.trainTargetsBatch
   elseif split == "test" then
      seqLength = self.testMaxLength
      if not self.fixedLength then
         seqLength = self.testLengthsBatch[1]
      end
      inputBatch = self.testInputsBatch
      targetBatch = self.testTargetsBatch
   else
      assert(false, "unknown split: " .. split)
   end

   local iSize = self:__getTotalInputSize()
   local inputVals = torch.zeros(3, bs * iSize, seqLength)

   local start = 1
   for i = 1, bs do
      for k, v in pairs(self.inputsInfo) do
         inputVals[{1+(i%2), {start,start+v.size-1},{}}]:copy(
            inputBatch[k]:select(2,i):narrow(1,1,seqLength):t()
                                                    )
         start = start + v.size
      end
   end

   if not self.noAsserts then
      assert((start - 1) == (iSize * bs), "This is strange...")
   end

   self.trainInputsWindow = image.display{
      image = inputVals,
      zoom = zoom,
      win = self.trainInputsWindow,
      legend = "Inputs"
   }

   local oSize = self:__getTotalOutputSize()
   local outputVals = torch.zeros(3, bs * oSize, seqLength)

   start = 1
   for i = 1, bs do
      for k, v in pairs(self.outputsInfo) do
         outputVals[{1+(i%2), {start, start+v.size-1},{}}]:copy(
            targetBatch[k]:select(2,i):narrow(1,1,seqLength):t()
                                                      )

         start = start + v.size
      end
   end

   if not self.noAsserts then
      assert((start - 1) == (oSize * bs), "This is strange...")
   end

   self.trainTargetsWindow = image.display{
      image = outputVals,
      zoom = zoom,
      win = self.trainTargetsWindow,
      legend = "Targets"
   }
end

--------------------------------------------------------------------------------
-- Function to be used when being verbose
--------------------------------------------------------------------------------

function Task:message(m)
   local name = self.name or "TASK"
   if self.verbose then
      print(string.format("[" .. name .. "] "):color("green") ..
               string.format(m):color("blue"))
   end
end
