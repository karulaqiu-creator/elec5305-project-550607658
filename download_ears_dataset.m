clc; clear;

% ==============================
% ⚙️ Configuration
% ==============================
output_folder = 'EARS_dataset_subset';
max_files_per_speaker = 160;   
num_speakers_to_download = 15; 
base_url = 'https://github.com/facebookresearch/ears_dataset/releases/download/dataset/';
blind_url = 'https://github.com/facebookresearch/ears_dataset/releases/download/blind_testset/blind_testset.zip';

% ==============================
% 📁 Create folder
% ==============================
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

original_dir = pwd;
cd(output_folder);

fprintf('🔽 Starting EARS dataset download...\n');

% ==============================
% 👤 Download speakers
% ==============================
for i = 1:num_speakers_to_download
    speaker_id = sprintf('p%03d', i);
    zip_name = [speaker_id '.zip'];
    url = [base_url zip_name];
    fprintf('\n📦 Downloading %s ...\n', zip_name);

    try
        websave(zip_name, url);
        unzip(zip_name, speaker_id);
        delete(zip_name);

        
        speaker_path = fullfile(pwd, speaker_id);
        wav_files = dir(fullfile(speaker_path, '*.wav'));
        if numel(wav_files) > max_files_per_speaker
            fprintf('⚙️  Trimming %d → %d files for %s\n', ...
                numel(wav_files), max_files_per_speaker, speaker_id);
            for j = max_files_per_speaker+1:numel(wav_files)
                delete(fullfile(speaker_path, wav_files(j).name));
            end
        end
    catch ME
        warning('⚠️ Failed to download %s (%s)', zip_name, ME.message);
    end
end

% ==============================
% 🎧 Download blind test set
% ==============================
fprintf('\n🔽 Downloading blind test set...\n');
try
    websave('blind_testset.zip', blind_url);
    unzip('blind_testset.zip', 'blind_testset');
    delete('blind_testset.zip');
catch ME
    warning('⚠️ Failed to download blind test set (%s)', ME.message);
end

% ==============================
% ✅ Finish
% ==============================
cd(original_dir);

fprintf('\n✅ All downloads complete. Dataset stored in "%s"\n', fullfile(pwd, output_folder));
