locales = {'en_US.UTF-8'}
os.setlocale(locales[1])

require("torch")


require("rnn")
require("tasks.doom_clock")
require("tasks.all_tasks")


function print_r ( t )  
   local print_r_cache={}
   local function sub_print_r(t,indent)
      if (print_r_cache[tostring(t)]) then
         print(indent.."*"..tostring(t))
      else
         print_r_cache[tostring(t)]=true
         if (type(t)=="table") then
            for pos,val in pairs(t) do
               if (type(val)=="table") then
                  print(indent.."["..pos.."] => "..tostring(t).." {")
                  sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                  print(indent..string.rep(" ",string.len(pos)+6).."}")
               elseif (type(val)=="string") then
                  print(indent.."["..pos..'] => "'..val..'"')
               else
                  print(indent.."["..pos.."] => "..tostring(val))
               end
            end
         else
            print(indent..tostring(t))
         end
      end
   end
   if (type(t)=="table") then
      print(tostring(t).." {")
      sub_print_r(t,"  ")
      print("}")
   else
      sub_print_r(t,"  ")
   end
end


local opts = {}
opts.fixedLength = true

local tasks = allTasks()
--print(tasks)

for k,v in ipairs(tasks) do
   local T = getTask(v) 
   local t = T()
end

--local dc = DoomClock()

--local X, T, F, L = dc:updateBatch()

--local backSteps = 3 -- arbitrary value

--local seqLSTM = nn.Sequencer(nn.FastLSTM(dc.inputSize, dc.outputSize, 
   --backSteps))

--local X, T, F, L = dc:updateBatch()

--local seq = {}
--local input = X[1]
--for i=1,dc.trainMaxLength do
   --seq[#seq + 1] = input[i]
--end




--print(seqLSTM:forward(seq))

