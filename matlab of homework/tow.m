numbers = 100:999;
isDivisible = mod(numbers, 30) == 0;
count = sum(isDivisible);
disp(['��100��999�ܱ�30���������ĸ���Ϊ��', num2str(count)]);