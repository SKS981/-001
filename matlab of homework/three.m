randStr = char(randi([65, 122], 1, 10));
filteredStr = randStr(~(randStr >= 'A' & randStr <= 'Z'));
finalStr = [filteredStr, 'ZQD'];
disp(['�������ַ�����', finalStr]);