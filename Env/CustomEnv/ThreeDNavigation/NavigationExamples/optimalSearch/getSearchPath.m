clear all
close all

figure(1)
k = 3;
t = 0:0.01:100;

z = k*t;
r = 5;

x = r .* cos(t * 6);
y = r .* sin(t * 6);


plot3(x, y, z);
pbaspect([1, 1, 1]);


figure(2)
k = 3;
t = 0:0.1:100;

z = k*t;
r = cos(t * 3);
%r = floor(t / 10);

x = r .* cos(t * 6);
y = r .* sin(t * 6) * 1.2;


plot3(x, y, z);
pbaspect([1, 1, 1]);

figure(3)
k = 5;
t = 0:0.04:100;

z = k*t;
r = cos(t * 4) * 45;
%r = floor(t / 10);

x = r .* cos(t * 6);
y = r .* sin(t * 6);


scatter3(x, y, z,[], z, 'filled');
hold on
plot3(x, y, z);

pbaspect([1, 1, 1]);

figure(4)
k = 5;
t = 0:0.04:100;

z = k*t;
r = cos(t * 5) * 45;
%r = floor(t / 10);

x = r .* cos(t * 7);
y = r .* sin(t * 7);


scatter3(x, y, z,[], z, 'filled');
hold on
plot3(x, y, z);

pbaspect([1, 1, 1]);



