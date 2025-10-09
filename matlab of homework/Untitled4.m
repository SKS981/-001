A = [1 2 3; 4 5 6; 7 8 9];  
[idx_row, idx_col] = find(A >= 5);  
disp('矩阵A：');  disp(A);
disp('大于等于5的元素的行列（行, 列）：');
for i = 1:length(idx_row)
    disp([num2str(idx_row(i)), ', ', num2str(idx_col(i))]);
end