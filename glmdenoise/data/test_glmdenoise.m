clear
rootpath = '~/Documents/GitHub/pyGLMdenoise/glmdenoise/data'
stimdur = 0.5
tr = .764
nConditions = 40
for runI=1:8
    d=load(fullfile(rootpath,sprintf('data_run%d.mat',runI)));
    data{runI} = single(d.data)';
    d=load(fullfile(rootpath,sprintf('design_run%d.mat',runI)));
    design{runI} = d.design;
end
results = GLMdenoisedata(design,data,stimdur,tr, ...
    'optimize',[],struct('numboots',100,'numpcstotry',20,'wantparametric',1), ...
    []);


% limit the betas to the valid conditions
modelmd = results.modelmd{2}(:,1:nConditions);
% limit the standard errors to the valid conditions
modelse = results.modelse{2}(:,1:nConditions);
% get the pooled error
poolse  = sqrt(mean(modelse.^2,2));
% normalise the betas by the pooled error to get t-patterns
modelmd = bsxfun(@rdivide,modelmd,poolse);