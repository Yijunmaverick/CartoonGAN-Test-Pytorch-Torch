require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'nngraph'
require 'paths'
require 'src/InstanceNormalization'
util = paths.dofile('src/util.lua')

local cmd = torch.CmdLine()

cmd:option('-input_dir', 'test_img');
cmd:option('-output_dir', 'test_output', 'Path to save stylized image.')
cmd:option('-load_size', 450)
cmd:option('-gpu', 0, '-1 for CPU mode')
cmd:option('-model_path', './pretrained_model/')
cmd:option('-style', 'Hosoda')

opt = cmd:parse(arg)

if paths.dirp(opt.output_dir) then
else
    paths.mkdir(opt.output_dir)
end

if opt.gpu > -1 then
  cutorch.setDevice(opt.gpu+1)
end

-- Define model
local model = torch.load(paths.concat(opt.model_path .. opt.style .. '_net_G_float.t7'))
model:evaluate()
if opt.gpu > -1 then
  print('GPU mode')
  model:cuda()
else
  print('CPU mode')
  model:float()
end

contentPaths = {}
if opt.input_dir ~= '' then 
  contentPaths = util.extractImageNamesRecursive(opt.input_dir)
else
  print('Please specify the input dierectory')
end

for i=1, #contentPaths do
  local contentPath = contentPaths[i]
  local contentExt = paths.extname(contentPath)
  local contentName = paths.basename(contentPath, contentExt)
  -- load image
	local img = image.load(contentPath, 3, 'float')
  -- resize image, keep aspect ratio
	img = image.scale(img, opt.load_size, 'bilinear')
	sg = img:size()
	local input = nil
  if opt.gpu > -1 then
    input = torch.zeros(1, sg[1], sg[2], sg[3]):cuda()
    input[1] = img
  else
    input = torch.zeros(1, sg[1], sg[2], sg[3]):float()
    input[1] = img
  end
  -- forward
	local out = util.deprocess_batch(model:forward(util.preprocess_batch(input)))
  -- save
	local savePath = paths.concat(opt.output_dir, contentName .. '_' .. opt.style .. '.' .. contentExt)
  image.save(savePath, out[1])
	collectgarbage()
end
print('Done!')




