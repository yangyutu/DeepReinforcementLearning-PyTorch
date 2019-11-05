clear all
close all

jsonData = loadjson('config_RBC.json');
centers = [];
orient = [];
scales = [];

for i = 1:jsonData.numObstacles
    centers = [centers; getfield(jsonData, ['obs' num2str(i-1)], 'center')];
    orient = [orient; getfield(jsonData, ['obs' num2str(i-1)], 'orient')];
    scales = [scales; getfield(jsonData, ['obs' num2str(i-1)], 'scale')];
end

figure(1)
hold on
for i = 1: size(centers, 1)

    plotRBCFunc(centers(i,:), scales(i), orient(i,:));
    
end
material('dull');
    xlabel('x')
    ylabel('y')
    zlabel('z')
    
    % Fix the axes scaling, and set a nice view angle
    axis('image');
    view([39 15]);
    camlight('left')
    
plotCurvedVesselFunc(500, 50)
view(3)
ylim([-100, 100])
xlim([-100, 100])
pbaspect([1, 2, 10])

axis on
print(gcf,'RBC.png','-dpng','-r600');