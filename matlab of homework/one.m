t = 0:0.5:2.5;  
y = zeros(size(t)); 
y(t>=0 & t<1) = t(t>=0 & t<1).^2;              
y(t>=1 & t<2) = t(t>=1 & t<2).^2 - 1;          
y(t>=2 & t<3) = t(t>=2 & t<3).^2 - 2*t(t>=2 & t<3) + 1; 
disp('t的值：');  disp(t);
disp('对应的y值：');  disp(y);