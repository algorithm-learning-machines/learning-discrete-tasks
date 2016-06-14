--------------------------------------------------------------------------------
-- This class implements a generic trainer.
--------------------------------------------------------------------------------

require("torch")
require("optim")

local Trainer = torch.class("Trainer")

function Trainer:__init(task, model, evaluator, opt)

   opt = opt or {}

   self.verbose = opt.verbose or false
   self.noAsserts = opt.noAsserts or false

   self.task = task
   self.model = model
   self.evaluator = evaluator

   self.epochsNo = opt.epochsNo or 1000
end

function Trainer:train()

   local model = self.model
   local task = self.task

   for epoch = 1, self.epochsNo do
      self:message("Starting epoch " .. epoch .. ".")
      task:reset("train")
      while task:isEpochOver("train") do
         local X, T, F, L, = task:updateBatch("train")
         local Y = model:forward(X, L)
      end
   end

end

function Trainer:message(m)
   if self.verbose then
      print(string.format("[TRAIN] "):color("green") ..
               string.format(m):color("blue"))
   end
end
