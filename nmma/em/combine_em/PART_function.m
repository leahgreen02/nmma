function [combined_posterior] = PART_function(event1, event2)
    run("~/random-tree-parallel-MCMC/src/init.m");
    %options = part_options('cut_type', 'kd', 'resample_N', 10000);
    options = part_options('cut_type', 'kd', 'resample_N', 1000, 'parallel', false);
    sub_chain = cell(1,2);
    %f = ['~/nmma_andrew/nmma/em/combine_em/event_files/event_' num2str(1) '_H_posterior.txt'];
    %e1 = importdata(f);
    %f = ['~/nmma_andrew/nmma/em/combine_em/event_files/event_' num2str(2) '_H_posterior.txt'];
    %e2 = importdata(f);
    %event1 = e1
    %event2 = e2
    disp(class(event1)) 
    %event1 = [event1{:}]
    %event2 = [event2{:}]

    event1 = reshape(event1, [length(event1), 1])
    event2 = reshape(event2, [length(event2), 1])
    
    sub_chain{1} = event1;
    sub_chain{2} = event2;
    %sub_chain = [event1; event2];
    disp(sub_chain)
    combined_posterior = aggregate_PART_pairwise(sub_chain, options);
end

%N_events = 8 % up to 29

%options = part_options('cut_type', 'kd', 'resample_N', 10000);
%sub_chain = cell(1,N_events);
%for i = 1:N_events
%    disp(i);
%    f = ['~/nmma_andrew/nmma/em/combine_em/event_files/event_' num2str(i-1) '_H_posterior.txt'];
%    e1 = importdata(f);
%    sub_chain{i} = e1;
%end

%combined_posterior_kd_pairwise = aggregate_PART_pairwise(sub_chain, options);

%writematrix(combined_posterior_kd_pairwise, 'event_files/combined_posterior.txt');

