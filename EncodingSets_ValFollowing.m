function [ntrain,set_index,Stimtrain,Stimval,spstrain,spsval,ttrain,tval,slentrain,slenval] = EncodingSets_ValFollowing(cells,train_frac,val_frac,Stim,sps,tstep)

slen = length(Stim);
ntrain=length(cells); % Number of total training sets
slentrain = floor(train_frac*slen); 
slenval = floor(val_frac*train_frac*slen); % Validation set must be smaller to account for creation of prior from ACF
start = randi([1 slen-slentrain-slenval+1],1,1);

set_index = repmat(start,ntrain,1);
val_index = start+slentrain;

% Validation sets
Stimval = Stim(val_index:val_index+slenval-1);
tval = tstep(val_index:val_index+slenval-1);
spsval = sps(val_index:val_index+slenval-1,cells);

% Training sets
Stimtrain = Stim(set_index(1):set_index(1)+slentrain-1); % Initial training sets
spstrain = sps(set_index(1):set_index(1)+slentrain-1,cells);
ttrain = tstep(set_index(1):set_index(1)+slentrain-1);

end
