load('FakeData_All.mat')

sets = 10; % Number of total training/validation sets
train_vector = 1:1:sets; % Index for training set from total sets

% Set params for fitting, including bases 
k_length = 100:10:300;    % Number of time bins in stimulus filter k
hpeakvector = .06:.03:.27;   % time of peak of last basis vector for h
nkbasis = 15;  % number of basis vectors for representing k
nhbasis = 15;  % number of basis vectors for representing h

% Selects neurons from population of 7
cell_choice = 1:1:size(sps,2); % Include index for relevant neurons
ncells = length(cell_choice);
sps = sps(:,cell_choice); % Edit cell_choice to select subset of neurons

%% 1. Curate Data

slen = length(Stim);
sps = sps(1:slen,:);
Stim = Stim(1:slen);
swid = 1;  % Stimulus width  (pixels); must match # pixels in stim filter
tinit = dt*(0:1:slen-1).'; % Index for time

%% 2. Downsampling

down = 5; % Defines how many data points will be assessed together
slen = floor(slen/down); % Must divide slen or we have to do another method
dt = dt*down; % Update time bin size

meanfilterfunction = @(theBlockStructure) mean(theBlockStructure.data(:));
sumfilterfunction = @(theBlockStructure) sum(theBlockStructure.data(:));
if down > 1
    tstep = blockproc(tinit(1:slen*down),[down 1],meanfilterfunction);
    sps = blockproc(sps(1:slen*down,:),[down 1],sumfilterfunction);
    Stim = interp1(tinit(1:slen*down),Stim(1:slen*down),tstep,'pchip');
end

%% 3. Set Up Training and Validation Sets

negloglik = zeros(length(k_length),length(hpeakvector));
stimlength = zeros(length(k_length),length(hpeakvector));
postlength = zeros(length(k_length),length(hpeakvector));
for m = 1:length(train_vector)
    train_index = train_vector(m);
        
[nval,set_index,Stimtrain,Stimval,spstrain,spsval,ttrain,tval,slentrain,slenval] = CrossValidationSets(sets,train_index,Stim,sps,tstep);


for n = 1:length(k_length)
    nkt = k_length(n);
    nkt = round(nkt/down); % Update filter length (so it's the same time)
    
%% 4.  Set parameters and display for GLM 

% makeSimStruct_GLM(nkt,dtStim,dtSp); % Create GLM structure with default params
ggsim = makeSimStruct_GLM(nkt,dt,dt); % Create GLM structure with default params

%% 5. Setup fitting params for random training set 

% Compute the STA
sta = zeros(nkt,ncells);
for j = 1:ncells
    sta(:,j) = simpleSTC(Stimtrain,spstrain(:,j),nkt); % Compute STA
end

% Set mask (if desired)
exptmask= []; %[1 slen*dtStim];  % data range to use for fitting (in s).

for p=1:length(hpeakvector)
    
hpeakFinal = hpeakvector(p);

for j=1:ncells
    % gg0 = makeFittingStruct_GLM(dtStim,dtSp,nkt,nkbasis,sta,nhbasis,hpeakFinal);
    temp = makeFittingStruct_GLM(dt,dt,nkt,nkbasis,sta(:,j),nhbasis,hpeakFinal); 
    temp.sps = spstrain(:,j);
    temp.mask = exptmask; % insert mask (optional)
    temp.ihw = randn(size(temp.ihw))*1; % initialize spike-history weights randomly
    gg0(j) = temp;
end
clearvars temp
% gg.dc currently set at 0 in code. Could change to [] to allow constant to
% be calculated

negloglival0_partial = zeros(ncells,1);
rr_partial = zeros(slentrain,ncells);
for j = 1:ncells
    % Compute conditional intensity at initial parameters 
    [tempneglog,temprr] = neglogli_GLM(gg0(j),Stimtrain);
    negloglival0_partial(j) = tempneglog;
    rr_partial(:,j) = temprr;
end
clearvars tempneglog temprr
negloglival0 = sum(negloglival0_partial);
fprintf('Initial negative log-likelihood: %.5f\n', negloglival0);

%% 6. Do ML fitting %=====================

% Because the cells are uncoupled, the encoding models are done separately
opts = {'display', 'iter', 'maxiter', 200}; % options for fminunc
for j = 1:ncells
    [tempgg1, tempnegloglival] = MLfit_GLM(gg0(j),Stimtrain,opts); % do ML (requires optimization toolbox)
    tempgg1.sta = sta(:,j);
    tempgg1.RefreshRate = 1000; % Is this the proper refresh rate?
    tempgg1.dt = dt;
    gg1(j) = tempgg1;
    negloglival1_partial(j) = tempnegloglival;
    stimfilter = tempgg1.k;
end
clearvars tempgg1 tempnegloglival
negloglival1 = sum(negloglival1_partial);

fprintf('Model negative log-likelihood: %.5f\n', negloglival1);

negloglik(n,p) = negloglival1;
stimlength(n,p) = nkt;
postlength(n,p) = length(gg1(1).ih);

end
end
info_train(m).NLL = negloglik;
info_train(m).StimFilterLength = stimlength;
info_train(m).PostFilterLength = postlength;

end

%% 7. Negloglik Plots

NLL = zeros(size(info_train(1).NLL));
% -2LL ~ Chi-squared distribution
% Independent chi-squared values can be summed
% Thus, we can sum up a composite negative log likelihood
for k=1:length(info_train)
    NLL = NLL + info_train(k).NLL; 
end
figure
h_NLL = heatmap(hpeakvector,k_length,NLL,'Colormap',jet);
h_NLL.YLabel = 'Time Lag of Stimulus Filter (ms)';
h_NLL.XLabel = 'Time of Last Post-Spike Basis Vector (ms)';

figure
hold on
for j=1:length(k_length)
    plot(1000*hpeakvector,NLL(j,:))
end
xlabel('Time of Last Post-Spike Basis Vector (ms)')
ylabel('Negative Log Likelihood')

% Stimulus Filter Length
figure
hold on
for j=1:length(hpeakvector)
    plot(k_length,NLL(:,j))
end
xticks([100 125 150 175 200 225 250])
xlabel('Time Lag of Stimulus Filter (ms)')
ylabel('Negative Log Likelihood')
