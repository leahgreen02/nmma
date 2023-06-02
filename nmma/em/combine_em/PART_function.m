function [combined_posterior] = PART_function(event1, event2)
    run("~/random-tree-parallel-MCMC/src/init.m");
    %options = part_options('cut_type', 'kd', 'resample_N', 10000);
    options = part_options('cut_type', 'kd', 'resample_N', 50000, 'parallel', false);
    sub_chain = cell(1,2);

    event1 = reshape(event1, [length(event1), 1]);
    event2 = reshape(event2, [length(event2), 1]);

    sub_chain{1} = event1;
    sub_chain{2} = event2;
    disp(sub_chain);
    %combined_posterior = aggregate_PART_pairwise(sub_chain, options);
    combined_posterior = aggregate_PART_onestage(sub_chain, options);
end
