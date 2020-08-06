function [nval,set_index,Stimtrain,Stimval,spstrain,spsval,ttrain,tval,slentrain,slenval] = ValidationSets(sets,train_index,Stim,sps,tstep)
 
slen = length(Stim);
slentrain = floor(slen/sets);
slenval = floor(0.75*slentrain); % Validation set must be smaller to account for creation of prior from ACF
set_index = 1; % First set index
for j=2:sets
   set_index = [set_index 1+(j-1)*slentrain];
end
nval=length(set_index) - 1; % Number of validation sets for each training set
val_init = set_index([1:train_index-1,train_index+1:end]); % Starting indices for validation sets

% Training sets
Stimtrain = Stim(set_index(train_index):set_index(train_index)+slentrain-1);
spstrain = sps(set_index(train_index):set_index(train_index)+slentrain-1,:);
ttrain = tstep(set_index(train_index):set_index(train_index)+slenval-1);

% Validation sets
Stimval = Stim(val_init(1):val_init(1)+slenval-1);
spsval = sps(val_init(1):val_init(1)+slenval-1,:);
tval = tstep(val_init(1):val_init(1)+slenval-1);
for i = 2:nval
    Stimval = [Stimval, Stim(val_init(i):val_init(i)+slenval-1)];
    spsval = [spsval, {sps(val_init(i):val_init(i)+slenval-1,:)}];
    tval = [tval, tstep(val_init(i):val_init(i)+slenval-1)];
end

end
