function sysCall_init()
    -- do some initialization here
    L, W = 10, 2
    generate_floor(L, W)
    lookdown_viewport(L, W)
    -- visualize_path({0,0,0.5,1,1,0.5,2,4,0.5})
    delete_path()
end

function sysCall_nonSimulation()
    -- is executed when simulation is not running
end

function sysCall_beforeSimulation()
    -- is executed before a simulation starts
    
end

function sysCall_afterSimulation()
    -- is executed before a simulation ends
end

function sysCall_cleanup()
    -- do some clean-up here
end

function lookdown_viewport(L, W)
    -- L: length of the floor
    -- W: width of the floor
    default_camera = sim.getObjectHandle('DefaultCamera')
    sim.setObjectOrientation(default_camera, -1, {-math.pi, 0, math.pi})
    hL = 4.327*math.ceil(L/5)
    hW = 6.2*math.ceil(W/5)
    -- sim.addStatusbarMessage(hL .. ' ' .. hW)
    h = math.max(hL, hW) + 0.5
    sim.setObjectPosition(default_camera, -1, {0, 0, h})
end

function generate_obstacles()
    generate_circle = function()
        
    end
   
    generate_prism = function()
    
    end
    
    generate_prism_tree = function()
    
    end
end

function visualize_path(path, color)
    -- path
    -- color
    if not color then
        color = {0.2, 0.2, 0.2}
    end
    
    if not _lineContainer then
        _lineContainer=sim.addDrawingObject(sim.drawing_lines,3,0,-1,99999, color)
    end
    
    sim.addDrawingObjectItem(_lineContainer,nil)
    if path then
        local pc=#path/3
        for i=1,pc-1,1 do
            lineDat={path[(i-1)*3+1],path[(i-1)*3+2],path[(i-1)*3+3],path[i*3+1],path[i*3+2],path[i*3+3]}
            sim.addDrawingObjectItem(_lineContainer,lineDat)
        end
    end
end

function delete_path()
    if _lineContainer then
        sim.removeDrawingObject(_lineContainer)
        sim.addStatusbarMessage("The line was removed")
    end
end

function generate_floor(L, W)
    -- generate the floor
    -- L: length > 0
    -- W: width > 0
    model = sim.getObjectHandle('ResizableFloor_5_25')
    e1=sim.getObjectHandle('ResizableFloor_5_25_element')
    e2=sim.getObjectHandle('ResizableFloor_5_25_visibleElement')
    if (L <= 0) or (W <= 0) then
        return
    end
    local sx=math.ceil(L/5)
    local sy=math.ceil(W/5)
    sim.addStatusbarMessage(sx .. " " .. sy)
    local sizeFact=sim.getObjectSizeFactor(model)
    sim.setObjectParent(e1,-1,true)
    local child=sim.getObjectChild(model,0)
    while child~=-1 do
        sim.removeObject(child)
        child=sim.getObjectChild(model,0)
    end
    local xPosInit=(sx-1)*-2.5*sizeFact
    local yPosInit=(sy-1)*-2.5*sizeFact
    local f1,f2
    for x=1,sx,1 do
        for y=1,sy,1 do
            if (x==1)and(y==1) then
                sim.setObjectParent(e1,model,true)
                f1=e1
            else
                f1=sim.copyPasteObjects({e1},0)[1]
                f2=sim.copyPasteObjects({e2},0)[1]
                sim.setObjectParent(f1,model,true)
                sim.setObjectParent(f2,f1,true)
            end
            local p=sim.getObjectPosition(f1, sim.handle_parent)
            p[1]=xPosInit+(x-1)*5*sizeFact
            p[2]=yPosInit+(y-1)*5*sizeFact
            sim.setObjectPosition(f1,sim.handle_parent,p)
        end
    end
end
