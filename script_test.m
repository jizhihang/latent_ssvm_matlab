function script_test
% TEST_SVM_STRUCT_LEARN
%   A demo function for SVM_STRUCT_LEARN(). It shows how to use
%   SVM-struct to learn a standard linear SVM.

  randn('state',0) ;
  rand('state',0) ;

  % ------------------------------------------------------------------
  %                                                      Generate data
  % ------------------------------------------------------------------
  th = pi/3 ;
  c = cos(th) ;
  s = sin(th) ;

  patterns = {} ;
  labels = {} ;
  labels_latent = {};
  for i=1:100
    patterns{i} = diag([2 .5]) * randn(2, 1) ;
    labels{i}   = 2*(randn > 0) - 1 ;
    patterns{i}(2) = patterns{i}(2) + labels{i} ;
    patterns{i} = [c -s ; s c] * patterns{i}  ;
    labels_latent{i} = 1;
  end
  

  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.labels_latent = labels_latent;
  parm.lossFn = @lossCB ;
  parm.constraintFn  = @constraintCB ;
  parm.featureFn = @featureCB ;
  parm.inferLatentFn = @inferLatentCB;
  parm.dimension = 2 ;
  parm.verbose = 1 ;
  model = svm_latent_struct_learn_mex(' -c 1.0', parm) ;
  w = model.w ;

  % ------------------------------------------------------------------
  %                                                              Plots
  % ------------------------------------------------------------------

  figure(1) ; clf ; hold on ;
  x = [patterns{:}] ;
  y = [labels{:}] ;
  plot(x(1, y>0), x(2,y>0), 'g.') ;
  plot(x(1, y<0), x(2,y<0), 'r.') ;
  set(line([0 w(1)], [0 w(2)]), 'color', 'y', 'linewidth', 4) ;
  xlim([-3 3]) ;
  ylim([-3 3]) ;
  set(line(10000*[w(2) -w(2)], 10000*[-w(1) w(1)]), ...
      'color', 'y', 'linewidth', 2, 'linestyle', '-') ;
  axis equal ;
  set(gca, 'color', 'b') ;
  w
end

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------


function delta = lossCB(param, y, ybar, hbar)
% compute the loss prediction (ybar, hbar) against correct y.
  delta = double(y ~= ybar) ;
end

function psi = featureCB(param, x, y, h)
% psi(x, y, h), the returned value has to be a sparse vector.
  psi = sparse(y*x/2) ;
end

function [yhat, h] = constraintCB(param, model, x, y)
% Loss augmented inference:
% argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end
  h = 1;
end

function h = inferLatentCB(param, model, x, y)
% computing argmax_{h} <w,psi(x,y,h)>. 
  h = 1;
end
