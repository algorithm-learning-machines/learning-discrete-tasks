--------------------------------------------------------------------------------
-- This files defines an abstract class that should be inherited by all classes
-- that implement a specific task.
--
-- When implementing a new task the following functions should be implemented:
--  * __init(opt)
--
--     i. It should first call Parent (this class)'s __init(opt)
--    ii. It should set up task specific values:
--            * inputSize
--            * outputSize
--            * targetAtEachStep / targetAtTheEnd
--            * isClassification / isRegression
--            * overwrite trainMaxLength, testMaxLength, fixedLength if needed
--   iii. It might call Parent's initTensors() to prepare batch tensors or to
--        generate the data sets
--
--  * generateBatch(inputs, targets, flags, lengths, isTraining)
--------------------------------------------------------------------------------

require("torch")
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

function Task:initTensors()
   if not self.noAsserts then
      assert(self.inputSize > 0 and self.outputSize > 0)
   end

   local bs = self.batchSize
   local ins = self.inputSize
   local outs = self.outputSize
   local trl = self.trainMaxLength
   local tsl = self.testMaxLength

   if not self.onTheFly then

      --------------------------------------------------------------------------
      -- Tensors for train data set

      local trs = self.trainSize

      self.trainInputs = torch.Tensor(trl, trs, ins)                   -- inputs

      if self.targetAtTheEnd then                                     -- targets
         self.trainTargets = torch.Tensor(1, trs, outs)
      else
         self.trainTargets = torch.Tensor(trl, trs, outs)
      end

      if not (self.targetAtTheEnd or self.targetAtEachStep) then        -- flags
         self.trainTargetFlags = torch.ByteTensor(trl, trs)
      end

      if not self.fixedLength then                                    -- lengths
         self.trainLengths = torch.LongTensor(trs)
      end

      for i = 1, (trs / bs) do
         local start = (i-1) * bs + 1
         local X = self.trainInputs:narrow(2, start, bs)
         local T = self.trainTargets:narrow(2, start, bs)
         local F
         if self.trainTargetFlagsBatch then
            F = self.trainTargetFlags:narrow(2, start, bs)
         end
         local L
         if self.trainLengths then
            L = self.trainLengths:narrow(1, start, bs)
         end
         self:generateBatch(X, T, F, L, true)
      end

      --------------------------------------------------------------------------
      -- Tensors for test data set

      local tss = self.testSize

      self.testInputs = torch.Tensor(tsl, tss, ins)                    -- inputs

      if self.targetAtTheEnd then                                     -- targets
         self.testTargets = torch.Tensor(1, tss, outs)
      else
         self.testTargets = torch.Tensor(tsl, tss, outs)
      end

      if not (self.targetAtTheEnd or self.targetAtEachStep) then        -- flags
         self.testTargetFlags = torch.ByteTensor(tsl, tss)
      end

      if not self.fixedLength then                                    -- lengths
         self.testLengths = torch.LongTensor(tss)
      end

      for i = 1, (tss / bs) do
         local start = (i-1) * bs + 1
         local X = self.testInputs:narrow(2, start, bs)
         local T = self.testTargets:narrow(2, start, bs)
         local F
         if self.testTargetFlagsBatch then
            F = self.testTargetFlags:narrow(2, start, bs)
         end
         local L
         if self.testLengths then
            L = self.testLengths:narrow(1, start, bs)
         end
         self:generateBatch(X, T, F, L, false)
      end

   else

      --------------------------------------------------------------------------
      -- Tensors for train batch

      local trs = self.trainSize

      self.trainInputsBatch = torch.Tensor(trl, bs, ins)               -- inputs

      if self.targetAtTheEnd then                                     -- targets
         self.trainTargetsBatch = torch.Tensor(1, bs, outs)
      else
         self.trainTargetsBatch = torch.Tensor(trl, bs, outs)
      end

      if not (self.targetAtTheEnd or self.targetAtEachStep) then        -- flags
         self.trainTargetFlagsBatch = torch.ByteTensor(trl, bs)
      end

      if not self.fixedLength then                                    -- lengths
         self.trainLengthsBatch = torch.LongTensor(bs)
      end

      --------------------------------------------------------------------------
      -- Tensors for test batch

      self.testInputsBatch = torch.Tensor(tsl, bs, ins)                -- inputs

      if self.targetAtTheEnd then                                     -- targets
         self.testTargetsBatch = torch.Tensor(1, bs, outs)
      else
         self.testTargetsBatch = torch.Tensor(tsl, bs, outs)
      end

      if not (self.targetAtTheEnd or self.targetAtEachStep) then        -- flags
         self.testTargetFlags = torch.ByteTensor(tsl, bs)
      end

      if not self.fixedLength then                                    -- lengths
         self.testLengthsBatch = torch.LongTensor(bs)
      end

   end

   self:message("Initialized tensors.")
end


function Task:generateBatch(inputs, targets, flags, lengths, isTraining)
   assert(false, "This is a virtual method :)")
end

function Task:updateBatch(split)
   split = split == nil and "train" or split
   if split == "train" then
      if self.onTheFly then
         self:generateBatch(
            self.trainInputsBatch,
            self.trainTargetsBatch,
            self.trainTargetFlagsBatch,
            self.trainLengthsBatch,
            true
         )
      else
         if self.trainIdx + self.batchSize > self.trainSize then
            self.trainIdx = 1
            self:message("Restarting training set.")
         end

         local i = self.trainIdx
         local bs = self.batchSize

         self.trainInputsBatch = self.trainInputs:narrow(2, i, bs)
         self.trainTargetsBatch = self.trainTargets:narrow(2, i, bs)
         if not (self.targetAtEachStep or self.targetAtTheEnd) then
            self.trainTargetFlagsBatch = self.trainTargetFlags:narrow(2, i, bs)
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
         self:generateBatch(
            self.testInputsBatch,
            self.testTargetsBatch,
            self.testTargetFlagsBatch,
            self.testLengthsBatch,
            false
         )
      else
         if self.testIdx + self.batchSize > self.testSize then
            self.testIdx = 1
            self:message("Restarting test set.")
         end

         local i = self.testIdx
         local bs = self.batchSize

         self.testInputsBatch = self.testInputs:narrow(2, i, bs)
         self.testTargetsBatch = self.testTargets:narrow(2, i, bs)
         if not (self.targetAtEachStep or self.targetAtTheEnd) then
            self.testTargetFlagsBatch = self.testTargetFlags:narrow(2, i, bs)
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

function Task:getType()
   if self.isClassification then
      return "classification"
   elseif self.isRegression then
      return "regression"
   else
      assert(false, "unknown task type")
   end
end

function Task:getInputSize()
   if not self.noAsserts then
      assert(self.inputSize ~= nil and self.inputSize > 0, "No input size")
   end
   return self.inputSize
end

function Task:getOutputSize()
   if not self.noAsserts then
      assert(self.outputSize ~= nil and self.outputSize > 0, "No output size")
   end
   return self.outputSize
end

function Task:getOutputsNo()
   return 1
end


function Task:hasTargetAtEachStep()
   return self.targetAtEachStep or false
end

function Task:hasTargetAtTheEnd()
   return self.targetAtTheEnd or false
end

function Task:cuda()
   require("cutorch")
   if self.onTheFly then
      -- Move the batch tensors to CUDA
      self.trainInputsBatch = self.trainInputsBatch:cuda()
      self.trainTargetsBatch = self.trainTargetsBatch:cuda()
      if not (self.targetAtTheEnd or self.targetAtEachStep) then
         self.trainTargetFlagsBatch = self.trainTargetFlagsBatch:cuda()
      end
      if not self.fixedLength then
         self.trainLengthsBatch = self.trainLengthsBatch:cuda()
      end

      self.testInputsBatch = self.testInputsBatch:cuda()
      self.testTargetsBatch = self.testTargetsBatch:cuda()
      if not (self.targetAtTheEnd or self.targetAtEachStep) then
         self.testTargetFlagsBatch = self.testTargetFlagsBatch:cuda()
      end
      if not self.fixedLength then
         self.testLengthsBatch = self.testLengthsBatch:cuda()
      end

   else
      -- Move the whole dataset to CUDA
      self.trainInputs = self.trainInputs:cuda()
      self.trainTargets = self.trainTargets:cuda()
      if not (self.targetAtTheEnd or self.targetAtEachStep) then
         self.trainTargetFlags = self.trainTargetFlags:cuda()
      end
      if not self.fixedLength then
         self.trainLengths = self.trainLengths:cuda()
      end

      self.testInputs = self.testInputs:cuda()
      self.testTargets = self.testTargets:cuda()
      if not (self.targetAtTheEnd or self.targetAtEachStep) then
         self.testTargetFlags = self.testTargetFlags:cuda()
      end
      if not self.fixedLength then
         self.testLengths = self.testLengths:cuda()
      end
   end
   self.onCuda = true
   self:message("Moved to GPU using CUDA.")
end

function Task:displayCurrentBatch(split, zoom)
   if not image then
      require("image")
   end

   split = split == nil and "train" or split
   zoom = zoom or 50

   if split == "train" then
      local seqLength = self.trainMaxLength
      if not self.fixedLength then
         seqLength = self.trainLengthsBatch[1]
      end
      local rowsNo = self.batchSize * self.inputSize

      self.trainInputsWindow = image.display{
         image = self.trainInputsBatch
            :narrow(1, 1, seqLength)
            :contiguous()
            :view(seqLength, rowsNo)
            :transpose(1, 2),
         zoom = zoom,
         win = self.trainInputsWindow,
         legend = "Inputs"
      }

      self.trainTargetsWindow = image.display{
         image = self.trainTargetsBatch
            :narrow(1, 1, seqLength)
            :contiguous()
            :view(seqLength, rowsNo)
            :transpose(1, 2),
         zoom = zoom,
         win = self.trainTargetsWindow,
         legend = "Targets"
      }
   elseif split == "test" then
      local seqLength = self.testMaxLength
      if not self.fixedLength then
         seqLength = self.testLengthsBatch[1]
      end
      local rowsNo = self.batchSize * self.inputSize

      self.testInputsWindow = image.display{
         image = self.testInputsBatch
            :narrow(1, 1, seqLength)
            :contiguous()
            :view(seqLength, rowsNo)
            :transpose(1, 2),
         zoom = zoom,
         win = self.testInputsWindow,
         legend = "Inputs"
      }

      self.testTargetsWindow = image.display{
         image = self.testTargetsBatch
            :narrow(1, 1, seqLength)
            :contiguous()
            :view(seqLength, rowsNo)
            :transpose(1, 2),
         zoom = zoom,
         win = self.testTargetsWindow,
         legend = "Targets"
      }
   end

end

function Task:message(m)
   if self.verbose then
      print(string.format("[TASK] "):color("green") ..
               string.format(m):color("blue"))
   end
end

return Task
