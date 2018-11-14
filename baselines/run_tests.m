function run_tests(mat_files)

v = 3;
p = 80;
n = p*v;
dimGroups = [ p, p, p ];
params015.maxiter = 15;
params100.maxiter = 100;
params200.maxiter = 200;

test_fns = { ...
 'MatchALS400Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 400); ...
 'MatchALS100Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 100); ...
 'MatchALS015Iter', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 15); ...
 'Spectral', @(W)  myspectral(W, p); ...
 'PGDDS200Iter', @(W) PGDDS(W, dimGroups, p, params200); ...
 'PGDDS100Iter', @(W) PGDDS(W, dimGroups, p, params100); ...
 'PGDDS015Iter', @(W) PGDDS(W, dimGroups, p, params015); ...
};

% disp_string =  [ '%06d Errors: ' ...
%                  'L1: %.03e, L2: %.03e, BCE: %.03e\n' ];
disp_string =  [ '%06d Errors: ' ...
                 'L1: %.03e, L2: %.03e, BCE: %.03e, ' ...
                 'Same sim: %.03e +/- %.03e, ' ...
                 'Diff sim: %.03e +/- %.03e, ' ...
                 '\n' ];

for test_fn_index = 1:size(test_fns,1)
  l1s = [];
  l2s = [];
  bces = [];
  ssame_m = [];
  ssame_s = [];
  sdiff_m = [];
  sdiff_s = [];
  test_fn = test_fns{test_fn_index,2};
  fid = fopen(sprintf('%sTestErrors.log', test_fns{test_fn_index,1}), 'w');
  fprintf('%s Method:\n', test_fns{test_fn_index,1})
  test_index = 0;
  for mat_index = 1:length(mat_files)
    test_mats = load(mat_files{mat_index});
    mat_length = size(test_mats.AdjMat, 1);
    l1_ = zeros(mat_length,1);
    l2_ = zeros(mat_length,1);
    bce_ = zeros(mat_length,1);
    ssame_m_ = zeros(mat_length,1);
    ssame_s_ = zeros(mat_length,1);
    sdiff_m_ = zeros(mat_length,1);
    sdiff_s_ = zeros(mat_length,1);
    for index = 1:mat_length
      fprintf('Matrix %03d of %03d, file %05d of %05d\r', mat_index, length(mat_files), index, mat_length)
      W = squeeze(test_mats.AdjMat(index,:,:)) + eye(n);
      Xgt = squeeze(test_mats.TrueEmbedding(index,:,:));
      Agt = Xgt*Xgt';
      Ah = max(0,min(1,test_fn(W)));
      [l1, l2, bce] = testOutput_soft(Ah,Agt);
      [ssame, ssame_std, sdiff, sdiff_std] = testOutputhist(Ah,Agt);
      l1_(index) = l1;
      l2_(index) = l2;
      bce_(index) = bce;
      ssame_m_(index) = ssame;
      ssame_s_(index) = ssame_std;
      sdiff_m_(index) = sdiff;
      sdiff_s_(index) = sdiff_std;
      fprintf(fid, disp_string, test_index, l1, l2, bce, ssame, ssame_std, sdiff, sdiff_std);
      % fprintf(disp_string, test_index, l1, l2, bce, ssame, ssame_std, sdiff, sdiff_std)
      test_index = test_index + 1;
    end
    l1s = [l1s; l1_];
    l2s = [l2s; l2_];
    bces = [bces; bce_];
    ssame_m = [ssame_m; ssame_m_];
    ssame_s = [ssame_s; ssame_s_];
    sdiff_m = [sdiff_m; sdiff_m_];
    sdiff_s = [sdiff_s; sdiff_s_];
  end
  fprintf('\n')
  fclose(fid);
  l1s = mean(l1s);
  l2s = mean(l2s);
  bces = mean(bces);
  ssame_m = mean(ssame_m);
  ssame_s = sqrt(mean(ssame_s.^2));
  sdiff_m = mean(sdiff_m);
  sdiff_s = sqrt(mean(sdiff_s.^2));
  fprintf(disp_string, 0, l1s, l2s, bces, ssame_m, ssame_s, sdiff_m, sdiff_s)
end


disp('Finished');

end

function [ssame, ssame_std, sdiff, sdiff_std] = testOutputhist(Ah,Agt,p)

N = sum(sum(Agt));
M = sum(sum(1-Agt));
ssame = sum(sum(Ah.*Agt)) / N;
ssame_std = sqrt(sum(sum((Ah.*Agt).^2)) / N - ssame^2);
sdiff = sum(sum(Ah.*(1-Agt))) / M;
sdiff_std = sqrt(sum(sum((Ah.*(1-Agt)).^2)) / M  - sdiff^2);

end

function [l1, l2, bce] = testOutput_soft(Ah,Agt)

l1  = mean2(abs(Ah-Agt));
l2  = mean2((Ah-Agt).^2);
bce = -mean2(Agt.*log2(eps+Ah) + (1-Agt).*log2(eps+1-Ah));

end

function [overlap, precision, recall, l1, l2, bce] = testOutput_full(Ah,Agt,p)

[overlap, precision, recall] = evalMMatch(triu(Ah,1),triu(Agt,1),false);
l1  = mean2(abs(Ah-Agt));
l2  = mean2((Ah-Agt).^2);
bce = mean2(Agt.*log2(eps+Ah) + (1-Agt).*log2(eps+1-Ah));

end

function testOutput_withX_full(Ah,X,Agt,Xgt,p,verbose)

[ U, S, V ] = svd(Xgt'*X);
Q = U*diag([ ones(p-1,1) ; det(U*V') ])*V';

Xr = X*Q';
if verbose
imagesc(cat(2, [ Xr, ones(n,30), Xgt ]));
axis equal;
end

[~, mi] = max(Xr,[],2);
Xh = eye(p);
Xh = Xh(mi,:);

[overlap, precision, recall] = evalMMatch(Xh,Xgt,verbose);
[overlap, precision, recall] = evalMMatch(triu(Ah,1),triu(Agt,1),verbose);

end

function [overlap, precision, recall] = evalMMatch(A,B,verbose)
if nargin < 3
  verbose = false;
end
s1 = A > 0;
s2 = B > 0;
overlap = nnz(s1&s2)/(nnz(s1|s2)+eps);
precision = nnz(s1&s2)/(nnz(s1)+eps);
recall = nnz(s1&s2)/(nnz(s2)+eps);

if verbose
  fprintf('Overlap: %f%%, Precision: %f%%, Recall: %f%%\n', 100*overlap, 100*precision, 100*recall)
end
end


