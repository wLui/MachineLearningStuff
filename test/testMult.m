% Want to try multiplying by each row corresponding column

col = [1; 2; 3]
mat = [1 2; 3 4; 5 6]

% want result: [1 2; 6 8; 15 18];
col .* mat

% Now try multiplying a row by a specific column
row = [1, 2, 3]
newMat = [1, 2, 3; 4, 5, 6;7, 8, 9]
extract = newMat(:, 2)'

% Want result: [2, 10, 24]
extract .* row 
