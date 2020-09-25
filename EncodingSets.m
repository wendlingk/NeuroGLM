function [ntrain,set_index,Stimtrain,Stimval,spstrain,spsval,ttrain,tval,slentrain,slenval] = EncodingSets(sets,range,train_frac,val_frac,Stim,sps,tstep)
 
slen = length(Stim);
slentrain = floor(train_frac*slen); 
slenval = floor(val_frac*train_frac*slen); % Validation set must be smaller to account for creation of prior from ACF
val_index = randi([1 slen-slenval+1],1);
% val_index = slen - slenval + 1; % Use last portion of stimulus as validation
ntrain=sum(sets); % Number of total training sets
% set_index = randi(val_index-slentrain,ntrain,1); % Starting indices for randomly selected training setsif 
start = randi([1 slen-slentrain-range+1],1,1);
set_index = randi([start start+range-1],ntrain,1);

% Make copies of spike trains based on number of sets requested for each
% cell in 'sets' variable
tempsps = ones(slen,1);
for i = 1:length(sets)
   if sets(i) > 0 
       tempsps = [tempsps, repmat(sps(:,i),[1 sets(i)])];
   end
end
tempsps = tempsps(:,2:end);

% Validation sets
Stimval = Stim(val_index:val_index+slenval-1);
tval = tstep(val_index:val_index+slenval-1);
spsval = tempsps(val_index:val_index+slenval-1,:);

% Training sets
Stimtrain = Stim(set_index(1):set_index(1)+slentrain-1); % Initial training sets
spstrain = tempsps(set_index(1):set_index(1)+slentrain-1,1);
ttrain = tstep(set_index(1):set_index(1)+slentrain-1);
for i = 2:ntrain
    Stimtrain = [Stimtrain, Stim(set_index(i):set_index(i)+slentrain-1)];
    spstrain = [spstrain, tempsps(set_index(i):set_index(i)+slentrain-1,i)];
    ttrain = [ttrain, tstep(set_index(i):set_index(i)+slentrain-1)];
end

end
