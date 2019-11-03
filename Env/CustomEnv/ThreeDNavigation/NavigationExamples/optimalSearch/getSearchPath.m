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
t = 0:0.001:40;

z = k*t;
r = cos(t * 4) * 45;
%r = floor(t / 10);

x = r .* cos(t * 6);
y = r .* sin(t * 6);

colormap(jet)
scatter3(x, y, z, 2, z, 'filled');
hold on
%plot3(x, y, z);
plotCylinderFunc(50, 200)
pbaspect([1, 1, 1]);

xlim([-50 50])
ylim([-50 50])
zlim([0 200])
%set(gca,'box','off')
%set(gca,'visible','off')
set(gca,'linewidth',2,'fontsize',16,'fontweight','bold','plotboxaspectratiomode','manual','xminortick','on','yminortick','on');
set(gca,'TickLength',[0.04;0.02]);
pbaspect([1 1 1])
%saveas(gcf,'traj.png')
axis off
print(gcf,'traj.png','-dpng','-r600');



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



