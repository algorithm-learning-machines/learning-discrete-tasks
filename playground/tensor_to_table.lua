require("torch")
require("nn")

-- Create some random 3D tensor
x = torch.rand(2, 2, 9)   -- 2 x 2 x (3 + 4 + 2)

print(x)

print("------------------------------")

-- Create model that splits in a table
unflattener = nn.ConcatTable()

unflattener:add(nn.Narrow(3, 1, 3))   -- 1-3
unflattener:add(nn.Narrow(3, 4, 4))   -- 4-7
unflattener:add(nn.Narrow(3, 8, 2))   -- 8-9

print(unflattener)

print("------------------------------")

-- Split the tensor
z = unflattener:forward(x)

print(z)
for k, v in pairs(z) do print("Output " .. k); print(v); end

print("------------------------------")

