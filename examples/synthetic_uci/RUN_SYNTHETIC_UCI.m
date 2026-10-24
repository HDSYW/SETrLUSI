function RUN_SYNTHETIC_UCI()
% Run the original synthetic and UCI demo from its own data directory.
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(here));
previous_path = path;
cleanup = onCleanup(@() path(previous_path));
addpath(genpath(fullfile(root,'Model')),genpath(fullfile(root,'tools')));
run_original_script();
end

function run_original_script()
% Isolate the original script's clear command from the path cleanup object.
run(fullfile(fileparts(mfilename('fullpath')),'Running_SETrLUSI.m'));
end
