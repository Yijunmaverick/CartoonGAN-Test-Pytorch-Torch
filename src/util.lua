--
-- code derived from https://github.com/soumith/dcgan.torch
--

local util = {}

require 'torch'
require 'nn'
require 'lfs'

-- Preprocesses an image before passing it to a net
-- Converts from RGB to BGR and rescales from [0,1] to [-1,1]
function util.preprocess(img)
    -- RGB to BGR
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    
    -- [0,1] to [-1,1]
    img = img:mul(2):add(-1)
    
    -- check that input is in expected range
    assert(img:max()<=1,"badly scaled inputs")
    assert(img:min()>=-1,"badly scaled inputs")
    
    return img
end

-- Undo the above preprocessing.
function util.deprocess(img)
    -- BGR to RGB
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    
    -- [-1,1] to [0,1]
    
    img = img:add(1):div(2)
    
    return img
end

function util.preprocess_batch(batch)
  for i = 1, batch:size(1) do
    batch[i] = util.preprocess(batch[i]:squeeze())
  end
  return batch
end

function util.deprocess_batch(batch)
  for i = 1, batch:size(1) do
   batch[i] = util.deprocess(batch[i]:squeeze())
  end
return batch
end

--
-- code derived from AdaIN https://github.com/xunhuang1995/AdaIN-style
--

function util.extractImageNamesRecursive(dir)
    local files = {}
    print("Extracting image paths: " .. dir)
  
    local function browseFolder(root, pathTable)
        for entity in lfs.dir(root) do
            if entity~="." and entity~=".." then
                local fullPath=root..'/'..entity
                local mode=lfs.attributes(fullPath,"mode")
                if mode=="file" then
                    local filepath = paths.concat(root, entity)
  
                    if string.find(filepath, 'jpg$')
                    or string.find(filepath, 'png$')
                    or string.find(filepath, 'jpeg$')
                    or string.find(filepath, 'JPEG$')
                    or string.find(filepath, 'ppm$') then
                        table.insert(pathTable, filepath)
                    end
                elseif mode=="directory" then
                    browseFolder(fullPath, pathTable);
                end
            end
        end
    end

    browseFolder(dir, files)
    return files
end


return util