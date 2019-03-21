function data = readNPZ(filename)
% Function to read NPZ files into matlab.
% *** Only reads a subset of all possible NPY files, specifically N-D arrays of certain data types.
% See https://github.com/kwikteam/npy-matlab/blob/master/tests/npy.ipynb for
% more.
%

dname = '/tmp/unzip-npz-reader';
[~,~,~] = mkdir(dname); % Ignore if has already been made
unzip(filename, dname);
map = containers.Map;
% read the data
npyfiles = dir([ dname '/*npy' ]);
for i = 1:length(npyfiles)
  key = npyfiles(i).name(1:end-length('.npy'));
  value = readNPY([ dname '/' npyfiles(i).name ]);
  map(key) = value;
end
delete([ dname '/*' ])
data = map;
