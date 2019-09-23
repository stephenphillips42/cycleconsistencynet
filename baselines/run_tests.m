function run_tests(npz_files, views, save_out)

if nargin < 2
  views = 3;
end
if nargin < 3
  save_out = false;
end

% Random setup
[~,~,~] = mkdir('/tmp/mymatlab');
npzmatlab_path = 'npz-matlab';
pathCell = regexp(path, pathsep, 'split');
onPath = any(strcmp(npzmatlab_path, pathCell));
if ~onPath
  addpath(npzmatlab_path);
end
if save_out
  [~,~,~] = mkdir('Adjmats');
end

v = views;
p = 80;
n = p*v;
dimGroups = ones(v,1)*p;
params015 = pgdds_params(15);
params025.maxiter = 25;
params050.maxiter = 50;
params100.maxiter = 100;
params200.maxiter = 200;

metric_info = { ...
    { 'l1',      'l1: %.03e, '         , @mean                 }, ...
    { 'l2',      'l2: %.03e, '         , @mean                 }, ...
    { 'ssame_m', 'ssame: { m: %.03e, ' , @mean                 }, ...
    { 'ssame_s', 'std: %.03e }, '      , @(x) sqrt(mean(x.^2)) }, ...
    { 'sdiff_m', 'sdiff: { m: %.03e, ' , @mean                 }, ...
    { 'sdiff_s', 'std: %.03e }, '      , @(x) sqrt(mean(x.^2)) }, ...
    { 'roc',     'roc: %.03e, '        , @mean                 }, ...
    { 'pr',      'p_r: %.03e, '        , @mean                 }, ...
};
metrics = cell(length(metric_info),1);
for i = 1:length(metric_info)
  metrics{i} = zeros(length(npz_files),1);
end

% matchals_iters = [ 15, 25, 50, 100 ];
% pgdds_iters = [ 15, 25, 50 ];
% matchals_iters = [ 10, 15, 20, 25, 30, 35, 40, 45 ];
matchals_iters = [  ];
pgdds_iters = [ 10, 15, 20, 25, 30, 35, 40, 45 ];
test_fns = cell(2 + length(matchals_iters) + length(pgdds_iters),2);
test_fns{1,1} = 'Spectral';
test_fns{1,2} = @(W) myspectral(W, p);
test_fns{2,1} = 'Random';
test_fns{2,2} = @(W) random_adjmat(W, p);
offset = 3;
for i = 1:length(matchals_iters)
  niters = matchals_iters(i);
  test_fns{offset,1} = sprintf('MatchALS%03dIter', niters);
  test_fns{offset,2} = @(W) mmatch_CVX_ALS(W, dimGroups, 'maxrank', p, 'maxiter', niters);
  offset = offset + 1;
end
for i = 1:length(pgdds_iters)
  niters = pgdds_iters(i);
  test_fns{offset,1} = sprintf('PGDDS%03dIter', niters);
  test_fns{offset,2} = @(W) PGDDS(W, dimGroups, p, pgdds_params(niters));
  offset = offset + 1;
end

saveout_str = '%s%02dOutputs/%04d.npy';
for test_fn_index = 1:size(test_fns,1)
  test_fn_tic = tic;
  test_fn = test_fns{test_fn_index,2};
  fid = fopen(sprintf('%s%02dViewTestErrors.yaml', test_fns{test_fn_index,1}, views), 'w');
  if save_out
    [~,~,~] = mkdir(sprintf('%s%02dOutputs', test_fns{test_fn_index,1}, views))
  end
  fprintf('%s Method:\n', test_fns{test_fn_index,1})
  test_index = 0;
  for npz_index = 1:length(npz_files)
    fprintf('Matrix %03d of %03d\r', npz_index, length(npz_files))
    [ W, Agt ] = load_adjmat(npz_files{npz_index});
    tic;
    A_output = test_fn(W);
    Ah = max(0,min(1,A_output));
    run_time = toc;
    values = evaluate_tests(Ah, Agt);
    for metric_idx = 1:length(metrics)
      metrics{metric_idx}(npz_index) = values(metric_idx);
    end
    disp_values(metric_info, fid, npz_index, values, run_time);
    test_index = test_index + 1;
    if save_out
      output_name = sprintf(saveout_str, test_fns{test_fn_index,1}, views, npz_index);
      writeNPY(single(Ah), output_name);
      adjmat_name = sprintf('Adjmats/%04d.npy', npz_index);
      if ~exist(adjmat_name)
        writeNPY(Agt, adjmat_name)
      end
    end
  end
  fprintf('\n')
  fclose(fid);
  means = zeros(length(metrics),1);
  for metric_idx = 1:length(metrics)
    means(metric_idx) = metric_info{metric_idx}{3}(metrics{metric_idx});
  end
  disp_values(metric_info, 1, test_fn_index, means, run_time);
  fprintf(1, 'Total time: %.03f seconds\n', toc(test_fn_tic));
end

disp('Finished');

end

%%% Display functions
function disp_values(metric_info, fid, idx, values, time)
  fprintf(fid, '%06d: {', idx);
  fprintf(fid, 'time: %.03e, ', time);
  for i = 1:length(values)
    fprintf(fid, metric_info{i}{2}, values(i));
  end
  fprintf(fid, ' }\n', time);
end

function [ means ] = get_metric_means(metrics)
end

%%% Evaulation functions
function [ values ] = evaluate_tests(Ah, Agt)
  [l1, l2] = testOutput_soft(Ah,Agt);
  [ssame, ssame_std, sdiff, sdiff_std] = testOutputhist(Ah,Agt);
  [roc, pr] = testOutput_roc_pr(Ah,Agt);
  values = [ l1, l2, ssame, ssame_std, sdiff, sdiff_std, roc, pr ];
end


function [l1, l2] = testOutput_soft(Ah,Agt)

l1  = mean2(abs(Ah-Agt));
l2  = mean2((Ah-Agt).^2);

end

function [ssame, ssame_std, sdiff, sdiff_std] = testOutputhist(Ah,Agt)

N = sum(sum(Agt));
M = sum(sum(1-Agt));
ssame = sum(sum(Ah.*Agt)) / N;
ssame_std = sqrt(sum(sum((Ah.*Agt).^2)) / N - ssame^2);
sdiff = sum(sum(Ah.*(1-Agt))) / M;
sdiff_std = sqrt(sum(sum((Ah.*(1-Agt)).^2)) / M  - sdiff^2);

end

function [roc, p_r] = testOutput_roc_pr_old(y_pred, y_true)

writeNPY(y_true(:), '/tmp/mymatlab/y_true.npy')
writeNPY(y_pred(:), '/tmp/mymatlab/y_pred.npy')
% Get python version
[status,cmdout] = system('python3 roc_test.py /tmp/mymatlab');
vals = strsplit(cmdout);
roc = str2double(vals{1});
p_r = str2double(vals{2});
% fprintf('ROC: %.6e, P-R: %.6e\n', roc, p_r)

end

function [roc, p_r] = testOutput_roc_pr(y_pred, y_true)
y_pred = y_pred(:);
y_true = y_true(:);
% Get standard ROC curve
[ y_pred_, sort_inds ] = sort(y_pred, 'descend');
y_true_ = y_true(sort_inds);
threshold_inds = [ find(diff(y_pred_)); length(y_pred) ];
tpr = cumsum(y_true_);
tpr = tpr(threshold_inds);
fpr = threshold_inds - tpr;

% Get precision recall curve
prec_ = tpr ./ (tpr + fpr);
prec_(isnan(prec_)) = 0;
reca_ = tpr / tpr(end);
last_ind = max(find(tpr < tpr(end)))+1;
% prec = prec_(last_ind:1:-1);
% reca = reca_(last_ind:1:-1);
prec = [ prec_(last_ind:-1:1); 1 ];
reca = [ reca_(last_ind:-1:1); 0 ];

fpr = [0; fpr/max(fpr)];
tpr = [0; tpr/max(tpr)];

% Calculate areas
roc = abs(trapz(fpr,tpr));
p_r = abs(trapz(reca,prec));

end

%%% Utility functions
function params = pgdds_params(niters)
  params.maxiter = 15;
end

function [W, Agt] = load_adjmat(npz_file)
  map = readNPZ(npz_file);
  idx = map('adj_mat_idx');
  val = map('adj_mat_val');
  W = full(sparse(double(idx(:,1))+1, double(idx(:,2))+1, double(val)));
  W = W + eye(size(W));
  idx = map('true_adj_mat_idx');
  val = map('true_adj_mat_val');
  Agt = full(sparse(double(idx(:,1))+1, double(idx(:,2))+1, double(val)));
end

function [ Ah ] = random_adjmat(W, p)
  Ah_emb = normr(randn(max(size(W)),p));
  Ah = Ah_emb*Ah_emb';
end


