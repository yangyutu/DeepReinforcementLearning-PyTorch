clear all
close all

z_c = linspace(0, 500, 200);
k1 = 0.05;
R1 = 15;
R0 = 10;
k2 = 0.02;
midRadius = 25;

y_c = R1 * cos(k1 * z_c);
x_c = zeros(size(z_c));


R_c = R0 * cos(k2 * z_c) + midRadius;


theta = linspace(0, 2 * pi, 200);
[Theta, ZC] = meshgrid(theta, z_c);

Theta = Theta(:);
ZC = ZC(:);

R = R0 * cos(k2 * ZC) + midRadius;

YC = R1 * cos(k1 * ZC);
XC = zeros(size(ZC));


figure(1)

X = XC + R .* cos(Theta);
Y = YC + R .* sin(Theta);
scatter3(ZC, X, Y, 2, [0.8, 0.8, 0.8],'filled')
pbaspect([5 ,1, 1])
ylim([-50, 50])
zlim([-50, 50])
hold on
plot3(z_c, x_c, y_c);

X = reshape(X, 200, 200);
Y = reshape(Y, 200, 200);
Z = reshape(ZC, 200, 200);

figure(5)
% Draw vessle
[TRI,v]= surf2patch(X,Y,Z,'triangle'); 
patch('Vertices',v,'Faces',TRI,'facecolor',[1 0 0 ],'facealpha',0.2, 'EdgeColor',       'none');
view(3)
pbaspect([1,1, 5])
ylim([-50, 50])
xlim([-50, 50])
