require("torch")
require("nn")

-- Create some random 3D tensor
x = torch.rand(2, 2, 9)   -- 2 x 2 x (3 + 4 + 2)

print(x)

print("------------------------------")

-- Create model that splits in a table
splitter = nn.ConcatTable()

splitter:add(nn.Narrow(3, 1, 3))   -- 1-3
splitter:add(nn.Narrow(3, 4, 4))   -- 4-7
splitter:add(nn.Narrow(3, 8, 2))   -- 8-9

print(splitter)

print("------------------------------")

-- Split the tensor
z = splitter:forward(x)

print(z)
for k, v in pairs(z) do print("Output " .. k); print(v); end

print("------------------------------")

-- Create some criterion and fake targets

c = nn.ParallelCriterion()

c:add(nn.MSECriterion())
c:add(nn.MSECriterion())
c:add(nn.MSECriterion())

t = {}
t[1] = torch.rand(2, 2, 3)
t[2] = torch.rand(2, 2, 4)
t[3] = torch.rand(2, 2, 2)

-- Forward through criterion

loss = c:forward(z, t)

print("Loss = " .. loss)
print("------------------------------")

-- Backward through criterion

dz = c:backward(z, t)

print("dz (grad output as table)")
for k, v in pairs(dz) do print("Grad output " .. k); print(v); end
print("------------------------------")

-- Backward through splitter

dx = splitter:backward(x, dz)

print("dx (grad output as tensor")
print(dx)
