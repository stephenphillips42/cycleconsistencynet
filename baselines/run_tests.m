function run_tests(mat_files)

v = 3;
p = 80;
n = p*v;
dimGroups = [ p, p, p ];

%  'MatchALSLimited', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 15); ...
test_fns = { ...
 'MatchALS', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 400); ...
 'Spectral', @(W)  myspectral(W, p); ...
 'PGDDS', @(W) PGDDS(W, dimGroups, p); ...
};

disp_string =  [ '%06d Errors: ' ...
                 'L1: %.03e, L2: %.03e, BCE: %.03e\n' ];

for test_fn_index = 1:size(test_fns,1)
  test_fn = test_fns{test_fn_index,2};
  fid = fopen(sprintf('%sTestErrors.log', test_fns{test_fn_index,1}), 'w');
  fprintf('%s Method:\n', test_fns{test_fn_index,1})
  test_index = 0;
  for mat_index = 1:length(mat_files)
    test_mats = load(mat_files{mat_index});
    mat_length = size(test_mats.AdjMat, 1);
    for index = 1:mat_length
      fprintf('Matrix %03d of %03d, file %05d of %05d\r', mat_index, length(mat_files), index, mat_length)
      W = squeeze(test_mats.AdjMat(index,:,:)) + eye(n);
      Xgt = squeeze(test_mats.TrueEmbedding(index,:,:));
      Agt = Xgt*Xgt';
      Ah = max(0,min(1,test_fn(W)));
      [l1, l2, bce] = testOutput_soft(Ah,Agt,p);
      fprintf(fid, disp_string, test_index, l1, l2, bce);
      % fprintf(disp_string, test_index, l1, l2, bce)
      test_index = test_index + 1;
    end
  end
  fprintf('\n')
  fclose(fid);
end


disp('Finished');

end

function [l1, l2, bce] = testOutput_soft(Ah,Agt,p)

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

function run_tests_old(mat_files)
v = 3;
p = 80;
n = p*v;
dimGroups = [ p, p, p ];
test_fns = { ...
 'MatchALS', @(W)  mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', 400); ...
 'Spectral', @(W)  mmatch_spectral(W, dimGroups, p); ...
 'PGDDS', @(W) PGDDS(W, dimGroups, p); ...
};
disp_string =  [ '%06d Errors: ' ...
                 'Overlap: %.03e, Precision: %.03e, Recall: %.03e, ' ...
                 'L1: %.03e,  L2: %.03e, BCE: %.03e\n' ];
for test_fn_index = 1:size(test_fns,1)
  test_fn = test_fns{test_fn_index,2};
  fid = fopen(sprintf('%sTestErrors.txt', test_fns{test_fn_index,1}), 'w');
  for mat_index = 1:length(mat_files)
    test_mats = load(mat_files{mat_index});
    for index = 1:size(test_mats.AdjMat, 1)
      fprintf('Matrix %03d of %03d, file %05d of %05d\r', mat_index, length(mat_files), index, size(test_mats.AdjMat, 1))
      W = squeeze(test_mats.AdjMat(index,:,:)) + eye(n);
      Xgt = squeeze(test_mats.TrueEmbedding(index,:,:));
      Agt = Xgt*Xgt';
      Ah = max(0,min(1,test_fn(W)));
      [overlap, precision, recall, l1, l2, bce] = testOutput_full(Ah,Agt,p);
      fprintf(fid, disp_string, index, overlap, precision, recall, l1, l2, bce);
    end
  end
  fprintf('\n')
  fclose(fid);
end
disp('Finished');
end



