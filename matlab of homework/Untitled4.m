A = [1 2 3; 4 5 6; 7 8 9];  
[idx_row, idx_col] = find(A >= 5);  
disp('����A��');  disp(A);
disp('���ڵ���5��Ԫ�ص����У���, �У���');
for i = 1:length(idx_row)
    disp([num2str(idx_row(i)), ', ', num2str(idx_col(i))]);
end