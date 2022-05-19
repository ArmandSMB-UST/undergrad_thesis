% Linear Regression
mdl_lin = fitlm(trainset, 'linear', 'Intercept', true);

% Stepwise Regression %%%%%%%%%%%%%%%%%%%%%%%%
mdl_step = stepwiselm(trainset, 'linear','PEnter',0.05, ...
     'PRemove',0.1, 'Intercept', true);
% 
% % mdl_step100 = stepwiselm(trainsetr100, 'linear','PEnter',0.05, ...
% %     'PRemove',0.1, 'Intercept', true);
% 
% mdl_step200 = stepwiselm(trainsetr200, 'linear','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_step300 = stepwiselm(trainsetr300, 'linear','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_step350 = stepwiselm(trainsetr350, 'linear','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_step400 = stepwiselm(trainsetr400, 'linear','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_step450 = stepwiselm(trainsetr450, 'linear','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stepwise Quadratic %%%%%%%%%%%%%%%%%%%%%%%%%%%%
mdl_quad = stepwiselm(trainset, 'quadratic','PEnter',0.05, ...
     'PRemove',0.1, 'Intercept', true);
% 
% % mdl_quad100 = stepwiselm(trainsetr100, 'quadratic','PEnter',0.05, ...
% %     'PRemove',0.1, 'Intercept', true);
% 
% mdl_quad200 = stepwiselm(trainsetr200, 'quadratic','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_quad300 = stepwiselm(trainsetr300, 'quadratic','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_quad350 = stepwiselm(trainsetr350, 'quadratic','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_quad400 = stepwiselm(trainsetr400, 'quadratic','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);
% 
% mdl_quad450 = stepwiselm(trainsetr450, 'quadratic','PEnter',0.05, ...
%     'PRemove',0.1, 'Intercept', true);

% Residuals

display(mdl_lin.Rsquared);
display(mdl_step.Rsquared);
display(mdl_quad.Rsquared);

% display(mdl_step0.Rsquared);
% display(mdl_step200.Rsquared);
% display(mdl_step300.Rsquared);
% display(mdl_step350.Rsquared);
% display(mdl_step400.Rsquared);
% display(mdl_step450.Rsquared);

% display(mdl_quad0.Rsquared);
% display(mdl_quad200.Rsquared);
% display(mdl_quad300.Rsquared);
% display(mdl_quad350.Rsquared);
% display(mdl_quad400.Rsquared);
% display(mdl_quad450.Rsquared);

% Plotting
% subplot(331);
% plot(mdl_step0);
% subplot(332);
% plot(mdl_step200);
% subplot(333);
% plot(mdl_step300);
% subplot(334);
% plot(mdl_step350);
% subplot(335);
% plot(mdl_step400);
% subplot(336);
% plot(mdl_step450);


% subplot(121);
% plotResiduals(mdl_sse90_quad);
% subplot(122);
% plotResiduals(mdl_sse90_quad, 'fitted');


%[T, P, df] = BPtest(test_resid, false);

function[T,P,df] = BPtest(z,studentize)

% INPUTS:
% z:            an object of class 'LinearModel' or a (n x p) matrix with the last
%               column corresponding to the dependent (response) variable and the first p-1 columns
%               corresponding to the regressors. Do not include a column of ones for the
%               intercept, this is automatically accounted for.

% studentize:   optional flag (logical). if True the studentized Koenker's statistic is
%               used. If false the statistics from the original Breusch-Pagan test is
%               used.

% OUTPUTS: 
% BP:    test statistics.
% P:     P-value. 
% df:    degrees of freedom of the asymptotic Chi-squared distribution of BP



if nargin == 1
    studentize = true;
end

if isa(z, 'LinearModel')
    
    n  = height(z.Variables);
    df = z.NumPredictors;    
    x = z.Variables{:,z.PredictorNames};
    r = z.Residuals.Raw;
    
else   
    
    x = z(:,1:end-1);
    y = z(:,end);    
    n = numel(y);
    df = size(x,2);    
    lm = fitlm(x,y);
    r = lm.Residuals.Raw;    
    
end


aux = fitlm(x,r.^2);
T = aux.Rsquared.Ordinary*n;

if ~studentize
    lam = (n-1)/n*var(r.^2)/(2*((n-1)/n*var(r)).^2);
    T  = T*lam;
end

P = 1-chi2cdf(abs(T),df);

end