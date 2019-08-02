%%% artificial EMG on GPU
function artificialEMG_onGPU(MUAP,ISIstats)
gpu = gpuDevice(1);
%seed = 12345;
%parallel.gpu.rng(seed);
timer = tic();
T = 100;   % in [s]
[M, L] = size(MUAP);  % number and length of MUAPs
dt = 16/L * 10^-3;  % temporal resolution in [s] defined by the length of MUAP, 256 data points in x ms (if x=8, sampling frequency = 32kHz)
twitch = ones(M,L);
twitch(:,[1 end]) = 0;
% define GPU arrays
MUs = gpuArray([1:M]');
time = gpuArray([1:fix(T/dt)]);  % in [s]
N = length(time);

% define neural drive and firing parameters
a = log(30)/M;
recthresh = zeros(1,M);           % recruit all
%recthresh = exp(a * [1:M]);      % recruit with exponential increase
lambdamin = 0;     % in [Hz]
lambdamax = 100;   % in [Hz]
g = 1;             % gain between neural drive and MUAP firing rate
ndmax = recthresh(end)*g + lambdamax;   % maximum neural drive
% define neural drive function
%f = 2*pi*0.05;
%neuraldrive = abs(sin(f*[1:N]*dt)) * ndmax;

% ramp function
neuraldrive = linspace(0,1,N)*ndmax;


% stepwise increase
%steps = 10;
%tmp = zeros(ceil(N/steps),steps);
%tmp(1,:) = 1;
%neuraldrive = cumsum(tmp(:))/steps * ndmax;

% constant drive
%neuraldrive = ones(1,N) * 10;

    function spike = makeSpikeTrain(MU, t)
        if recthresh(MU) < neuraldrive(t)
            lambda = g*(neuraldrive(t)-recthresh(MU)) + lambdamin;
            lambda = min(lambda, lambdamax);
            % ISI exponentially distributed
            exprn = -log(rand())/lambda;
            spike = exprn < dt;
        else
            spike = false;
        end
    end

for x = 5:5:5
sEMG = zeros(1, N + L);
force = zeros(1, N + L);
spikes = zeros(1, N);
reps = ceil(N*M/double(intmax('int32')));
N_p = floor(N/reps);
fprintf('%d, %d\n', N_p, reps);
% ISI distributed according to a Gaussian
if strcmp(ISIstats, 'gauss')
    lambda = x;
    gauss_st = cumsum(lambda^-1 + lambda^-1*0.2*gpuArray.randn(T*lambda*2,M),1);
end
for r = 1:reps
    fprintf('working on %d of %d repetitions.\n',r, reps);
    gpuSimpleTime = toc(timer);
    fprintf('elapsed time [min]: %f\n', gpuSimpleTime/60);
    st = gpuArray(logical(zeros(M, N_p, 'uint8')));
    switch ISIstats
        case 'gauss'
            if reps == 1
                tmp = histc(gauss_st, time(N_p*(r-1)+1:N_p*r)*dt);
                tmp(end,:) = zeros(1,M);
            elseif r == 1
                tmp = histc(gauss_st, time(N_p*(r-1)+1:N_p*r+1)*dt);
                tmp(end,:) = [];
            elseif r == reps && reps*N_p == N
                tmp = histc(gauss_st, time(N_p*(r-1):N_p*r)*dt);
                tmp(1,:) = [];
                tmp(end,:) = zeros(1,M);
            else
                tmp = histc(gauss_st, time(N_p*(r-1):N_p*r+1)*dt);
                tmp([1 end],:) = [];
            end
            st = tmp' > 0;
        case 'exponential'
            st = arrayfun(@makeSpikeTrain, MUs, time(N_p*(r-1)+1:N_p*r));
        case 'weibull'

    end
    for t = N_p*(r-1)+1:N_p*r
        sEMG(1,t:t+L-1) = sEMG(1,t:t+L-1) + sum(MUAP(st(:,t-N_p*(r-1)),:), 1);
        force(1,t:t+L-1) = force(1,t:t+L-1) + sum(twitch(st(:,t-N_p*(r-1)),:), 1);
        spikes(t) = sum(gather(st(:,t-N_p*(r-1))));
    end
end
rest = N - reps*N_p;
fprintf('%d, %d, %d\n', rest, reps*N_p,N);
if rest > 0
    st = gpuArray(logical(zeros(M, rest, 'uint8')));
    if strcmp(ISIstats, 'gauss')
        tmp = histc(gauss_st, time(N_p*(r-1):N_p*r)*dt);
        tmp(1,:) = [];
        tmp(end,:) = zeros(1,M);
        st = tmp' > 0;
    else
        st = arrayfun(@makeSpikeTrain, MUs, time(N_p*reps+1:end));
    end
    for t = N_p*reps+1:N
        sEMG(1,t:t+L-1) = sEMG(1,t:t+L-1) + sum(MUAP(st(:,t-N_p*reps),:), 1);
        force(1,t:t+L-1) = force(1,t:t+L-1) + sum(twitch(st(:,t-N_p*reps),:), 1);
        spikes(t) = sum(st(:,t-N_p*(r-1)));
    end
end
%spiketimes = gather(gauss_st);
time = gather(time);
save(['../results/test_withforce_' ISIstats 'ISI_fr' num2str(neuraldrive(end)) '_' num2str(M) 'MUAP.mat'],'sEMG','recthresh','neuraldrive','dt','T','spikes','time','force')
gpuSimpleTime = toc(timer);
fprintf('%f\n', gpuSimpleTime);
end
end

