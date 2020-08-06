function [prob,actual_spikes] = EncModel_Val(gg1,Stimval,spsval,dt)

% gg1: Model structure that includes stimulus, post-spike filters, and
% constant correction
% Stimval: Stimulus for validation set
% spsval: Spike trian for validation set

stimfilt = gg1.k;
histfilt = gg1.ih;
constant = gg1.dc;

initial = 1 + max(length(stimfilt),length(histfilt));

Xtemp = toeplitz(Stimval);
%X = Xtemp(initial:end,1:length(stimfilt));
X = Xtemp(:,1:length(stimfilt));
Xkt = X*stimfilt;

Spiketemp = toeplitz(spsval);
%Spike = Spiketemp(initial:end,1:length(histfilt));
Spike = Spiketemp(:,1:length(histfilt));
Spikeht = Spike*histfilt;

%lambda = exp(constant + Xkt(length(stimfilt)+1:end,:) + Spikeht(length(histfilt)+1:end,:));
lambda = exp(constant + Xkt + Spikeht);
prob = dt*lambda;

actual_spikes = sum(spsval);

    
end