
function [ twitch ] = active_force( P, T, MU_spike_train, discharge_cnt )

gain_at_intersection = (1 - exp(-2 * (0.4^3)))/0.4; 
if discharge_cnt == 1,
    nfr = 0;
else
    current_ISI = MU_spike_train(discharge_cnt)-MU_spike_train(discharge_cnt-1);
    nfr = T / (current_ISI); %normalized firing rate
end

if nfr <= 0.4,
    gain_f = 1;
else
    S = 1 - exp(-2 * (nfr^3)); %equation 16
    gain_f = (S / nfr) / gain_at_intersection; %equation 17
end

%twitch shape:
twitch = gain_f * ((P .* [1:1000]) ./ T .* exp(1 - ([1:1000] ./ T))); %equation 18

end 
