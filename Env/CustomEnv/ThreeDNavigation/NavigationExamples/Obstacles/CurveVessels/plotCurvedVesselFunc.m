function plotCurvedVesselFunc(height, radius)

z_c = linspace(0, height, 200);
k1 = 0.05;
R1 = radius;
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




X = XC + R .* cos(Theta);
Y = YC + R .* sin(Theta);


X = reshape(X, 200, 200);
Y = reshape(Y, 200, 200);
Z = reshape(ZC, 200, 200);

% Draw vessle
[TRI,v]= surf2patch(X,Y,Z,'triangle'); 
patch('Vertices',v,'Faces',TRI,'facecolor',[1 0 0 ],'facealpha',0.2, 'EdgeColor',       'none');

area = sum(pi^2 * R * z_c(2) - z_c(1))

