syms  a11 a12 a21 a22
A = [a11 a12; a21 a22];  
det_A = det(A);   
inv_A = inv(A);   
eig_A = eig(A);   

disp('����A��');  disp(A);
disp('����ʽֵ��');  disp(det_A);
disp('�����');  disp(inv_A);
disp('��������');  disp(eig_A);