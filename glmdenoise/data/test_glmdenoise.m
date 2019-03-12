clear
rootpath = '~/Documents/GitHub/pyGLMdenoise/glmdenoise/data'
stimdur = 0.5
tr = .764
for runI=1:8
    d=load(fullfile(rootpath,sprintf('data_run%d.mat',runI)));
    data{runI} = single(d.data)';
    d=load(fullfile(rootpath,sprintf('design_run%d.mat',runI)));
    design{runI} = d.design;
end
results = GLMdenoisedata(design,data,stimdur,tr, ...
    'optimize',[],struct('numboots',100,'numpcstotry',20,'wantparametric',1), ...
    []);
