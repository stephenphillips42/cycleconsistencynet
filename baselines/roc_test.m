function test(verbose, plot_vals)
if nargin < 1
  verbose = false;
end
if nargin < 2
  plot_vals = false;
end
% This is a test script

[~,~,~] = mkdir('/tmp/mymatlab');
npzmatlab_path = 'npz-matlab';
pathCell = regexp(path, pathsep, 'split');
onPath = any(strcmp(npzmatlab_path, pathCell));
if ~onPath
  addpath(npzmatlab_path);
end

niters = 20;
n = 10;
nbins = 14;
bins = linspace(0,1,nbins);
sz = 300;
if plot_vals
  fig = figure('Position', [floor(1.5*sz) 4*sz 3*sz 3*sz]); 
end
svals = [ 0.8 1.3 2.1 3.0 ];
% Storage array
ml_rocs = zeros(niters, length(svals));
py_rocs = zeros(niters, length(svals));
my_rocs = zeros(niters, length(svals));
ml_p_rs = zeros(niters, length(svals));
py_p_rs = zeros(niters, length(svals));
my_p_rs = zeros(niters, length(svals));

for iter = 1:niters
for si = 1:length(svals)
  %%% Calculate things
  % Build score data
  [ y_true, y_pred ] = build_data(n,svals(si));
  % Build classification curves
  [mlfpr,mltpr,~,mlroc] = perfcurve(y_true, y_pred, 1);
  [mlreca,mlprec,~,mlp_r] = perfcurve(y_true, y_pred, 1, 'XCrit', 'prec', 'YCrit', 'reca');
  mlprec(end) = 1;
  % Python version of classification curves
  [pyfpr, pytpr, pyprec, pyreca, pyroc, pyp_r] = run_python(y_true, y_pred);
  % My version of classification curves
  [myfpr, mytpr, myprec, myreca, myroc, myp_r] = ...
      binary_clf_curve(y_true, y_pred);
  %% Save for iteration
  ml_rocs(iter,si) = mlroc;
  ml_p_rs(iter,si) = mlp_r;
  py_rocs(iter,si) = pyroc;
  py_p_rs(iter,si) = pyp_r;
  my_rocs(iter,si) = myroc;
  my_p_rs(iter,si) = myp_r;
  %%% Print things
  if verbose
    fprintf('Values for s=%.04f\n',svals(si))
    fprintf('   MATLAB: ROC: %.6e, P-R: %.6e\n', mlroc, mlp_r)
    fprintf('   Python: ROC: %.6e, P-R: %.6e\n', pyroc, pyp_r)
    fprintf('My MATLAB: ROC: %.6e, P-R: %.6e\n', myroc, myp_r)
    disp(' orig     py  mine  <- fpr/tpr')
    disp([ length(mlfpr), length(pyfpr), length(myfpr) ])
    disp(' orig     py  mine  <- prec/reca')
    disp([ length(mlprec), length(pyprec), length(myprec) ])
    disp([ norm(myprec - pyprec), norm(myreca - pyreca) ])
  end
  %%% Plot things
  % Histogram things
  if plot_vals
    tvals = find(y_true);
    fvals = find(1-y_true);
    subplot(3,length(svals),si);
    histogram(y_pred(tvals), bins);
    hold on;
    histogram(y_pred(fvals), bins);
    title(sprintf('Plots for s=%.04f\n',svals(si)));
    subplot(3,length(svals),length(svals) + si);
    plot(fpr0,tpr0);
    xlim([ 0,1 ]);
    ylim([ 0,1 ]);
    hold on;
    plot(pyfpr,pytpr)
    plot(myfpr,mytpr)
    hold off;
    subplot(3,length(svals),2*length(svals) + si);
    plot(mlreca,mlprec);
    xlim([ 0,1 ])
    ylim([ 0,1 ])
    hold on
    plot(pyreca,pyprec);
    plot(myreca,myprec);
    hold off
  end
end
end

for si = 1:length(svals)
  ml_rocerr = abs(ml_rocs(:,si) - my_rocs(:,si));
  py_rocerr = abs(py_rocs(:,si) - my_rocs(:,si));
  ml_p_rerr = abs(ml_p_rs(:,si) - my_p_rs(:,si));
  py_p_rerr = abs(py_p_rs(:,si) - my_p_rs(:,si));
  mlpy_rocerr = abs(ml_rocs(:,si) - py_rocs(:,si));
  mlpy_p_rerr = abs(ml_p_rs(:,si) - py_p_rs(:,si));
  fprintf('\nErrs for mine vs x (s=%.04f)\n',svals(si))
  fprintf('  Mine-MATLAB: ROC: %.6e +/- %.06e, P-R: %.6e +/- %.06e\n', ...
          mean(ml_rocerr), std(ml_rocerr), mean(ml_p_rerr), std(ml_p_rerr));
  fprintf('  Mine-Python: ROC: %.6e +/- %.06e, P-R: %.6e +/- %.06e\n', ...
          mean(py_rocerr), std(py_rocerr), mean(py_p_rerr), std(py_p_rerr));
  fprintf('MATLAB-Python: ROC: %.6e +/- %.06e, P-R: %.6e +/- %.06e\n', ...
          mean(mlpy_rocerr), std(mlpy_rocerr), mean(mlpy_p_rerr), std(mlpy_p_rerr));
end

end

function [ y_true, y_pred ] = build_data(n,s)

E = eye(n);
M = cat(1,randperm(n)',randperm(n)',randperm(n)');
P = E(M,:);
A = P*P';
Anoise = abs(2*A + s*randn(size(A)));
y_true = A(:);
y_pred = min(1 + 2*s, Anoise(:)) / (1 + 2*s);

end

function [fpr, tpr, prec, reca, roc, p_r] = run_python(y_true, y_pred)

writeNPY(y_true, '/tmp/mymatlab/y_true.npy')
writeNPY(y_pred, '/tmp/mymatlab/y_pred.npy')
% Get python version
[status,cmdout] = system('python3 roc_test.py /tmp/mymatlab');
vals = strsplit(cmdout);
roc = str2double(vals{1});
p_r = str2double(vals{2});
% fprintf('ROC: %.6e, P-R: %.6e\n', roc, p_r)

fpr = readNPY('/tmp/mymatlab/fpr.npy');
tpr = readNPY('/tmp/mymatlab/tpr.npy');
prec = readNPY('/tmp/mymatlab/precision.npy');
reca = readNPY('/tmp/mymatlab/recall.npy');

end

function [fpr, tpr, prec, reca, roc, p_r] = binary_clf_curve(y_true, y_pred)

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
% p_r = abs(sum(diff(reca).*prec(1:end-1)));

end



