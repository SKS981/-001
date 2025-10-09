numbers = 100:999;
isDivisible = mod(numbers, 30) == 0;
count = sum(isDivisible);
disp(['在100到999能被30整除的数的个数为：', num2str(count)]);