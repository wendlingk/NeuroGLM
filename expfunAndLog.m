function [f,df,ddf, lf, dlf, ddlf] = expfunAndLog(x)
%  [f,df,ddf] = expfun(x)
%
%  replacement for 'exp' that returns 3 arguments (value, 1st & 2nd deriv)

f = exp(x);
df = f;
ddf = df;
lf = x;
dlf = ones(length(x),1);
ddlf = zeros(length(x),1);
