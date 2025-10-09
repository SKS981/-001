syms  a11 a12 a21 a22
A = [a11 a12; a21 a22];  
det_A = det(A);   
inv_A = inv(A);   
eig_A = eig(A);   

disp('矩阵A：');  disp(A);
disp('行列式值：');  disp(det_A);
disp('逆矩阵：');  disp(inv_A);
disp('特征根：');  disp(eig_A);