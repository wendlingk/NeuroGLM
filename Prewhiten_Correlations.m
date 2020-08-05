% If loading different Decoding Stats structures and combing them into one
% Otherwise, load initial Decoding Stats structure and skip to next section

load Decoding_Stats_Range500.mat %1x6 struct, netSizes=2,..,7
ErrStat=Error_Stats;
nmNtSz=size(Error_Stats,2); %6 total network sizes, 2,..,7
load Decoding_Stats_Range1000.mat
for j=1:nmNtSz
    Error_Stats(j).Range=[Error_Stats(j).Range ErrStat(j).Range];
    Error_Stats(j).Error=[Error_Stats(j).Error ErrStat(j).Error];
    Error_Stats(j).Bias_diff=[Error_Stats(j).Bias_diff ErrStat(j).Bias_diff];
    Error_Stats(j).Stim_diff=[Error_Stats(j).Stim_diff ErrStat(j).Stim_diff];
    Error_Stats(j).Post_diff=[Error_Stats(j).Post_diff ErrStat(j).Post_diff];
    Error_Stats(j).Stimulus=[Error_Stats(j).Stimulus ErrStat(j).Stimulus];
    Error_Stats(j).Optimal=[Error_Stats(j).Optimal ErrStat(j).Optimal];
    Error_Stats(j).CholeskyFlag=[Error_Stats(j).CholeskyFlag Err_Stats(j).CholeskyFlag];
end
%repeat
ErrStat=Error_Stats;
load Decoding_Stats_Range2000.mat
for j=1:nmNtSz
    Error_Stats(j).Range=[Error_Stats(j).Range ErrStat(j).Range];
    Error_Stats(j).Error=[Error_Stats(j).Error ErrStat(j).Error];
    Error_Stats(j).Bias_diff=[Error_Stats(j).Bias_diff ErrStat(j).Bias_diff];
    Error_Stats(j).Stim_diff=[Error_Stats(j).Stim_diff ErrStat(j).Stim_diff];
    Error_Stats(j).Post_diff=[Error_Stats(j).Post_diff ErrStat(j).Post_diff];
    Error_Stats(j).Stimulus=[Error_Stats(j).Stimulus ErrStat(j).Stimulus];
    Error_Stats(j).Optimal=[Error_Stats(j).Optimal ErrStat(j).Optimal];
    Error_Stats(j).CholeskyFlag=[Error_Stats(j).CholeskyFlag Err_Stats(j).CholeskyFlag];
end

%% Assess stationarity via Adjusted Dickey-Fuller and KPSS Tests

% Original time series (Actual stimulus and decoded stimulus)
adfhyp_x = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
adfhyp_y = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
kpsshyp_x = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
kpsshyp_y = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
for i = 1:length(Error_Stats)
    x = Error_Stats(i).Stimulus;
    y = Error_Stats(i).Optimal;    
    for j=1:size(x,2)
        adfhyp_x(j,i) = adftest(x(:,j),'Alpha',0.01);
        adfhyp_y(j,i) = adftest(y(:,j),'Alpha',0.01);
        kpsshyp_x(j,i) = kpsstest(x(:,j),'Alpha',0.1);
        kpsshyp_y(j,i) = kpsstest(y(:,j),'Alpha',0.1);
    end
end

% Time series differenced once
adfhyp_xdiff = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
adfhyp_ydiff = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
kpsshyp_xdiff = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
kpsshyp_ydiff = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
for i = 1:length(Error_Stats)
    x = diff(Error_Stats(i).Stimulus);
    y = diff(Error_Stats(i).Optimal);    
    for j=1:size(x,2)
        adfhyp_xdiff(j,i) = adftest(x(:,j),'Alpha',0.01);
        adfhyp_ydiff(j,i) = adftest(y(:,j),'Alpha',0.01);
        kpsshyp_xdiff(j,i) = kpsstest(x(:,j),'Alpha',0.1);
        kpsshyp_ydiff(j,i) = kpsstest(y(:,j),'Alpha',0.1);
    end
end

%% Prewhiten Time Series and Then Find Correlations

% Try-catch needed for large number of iterations for when original
% parameters produce a matrix that is unstable or not invertible
corradj = zeros(size(Error_Stats(1).Stimulus,2),length(Error_Stats));
for i = 1:length(Error_Stats)
    % Differenced time series chosen here for more reliable stationarity
    % Thus, all ARIMA models below actually are ARIMA(p,1,q)
    x = diff(Error_Stats(i).Stimulus); 
    y = diff(Error_Stats(i).Optimal);
    for j=1:size(Error_Stats(i).Stimulus,2)
        try
            try
                xMdl = estimate(arima(6,0,3),x(:,j)); % ARIMA(6,1,3)
                yMdl = estimate(arima(6,0,3),y(:,j));
            catch
                xMdl = estimate(arima(8,0,4),x(:,j)); % ARIMA(8,1,4)
                yMdl = estimate(arima(8,0,4),y(:,j));
            end
        catch
            xMdl = estimate(arima(4,0,2),x(:,j)); % ARIMA(4,1,2)
            yMdl = estimate(arima(4,0,2),y(:,j));
        end
        xresid = infer(xMdl,x(:,j)); % Extracts residuals from model
        yresid = infer(yMdl,y(:,j));

        corradj(j,i) = corr(xresid,yresid); % Unbiased correlations
    end
end
