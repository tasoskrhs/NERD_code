%
% plot outcomes of Nerd computations over alpha
% Then load ITE computations and compare


clear; close all; clc;
% PARAMETERS
%------------
d = 16%10%6;
bs = 4000; %2000; 
iters = 5%20 %3%2%4%1;
gp = 0.1; % gradient penalty parameter (Lipschitz)
bsofnerd2 = 4000; %2000;%  

warning('changed alpha range')
alpha = [0.1: 0.1: 0.9]; %linspace(0.1, 1.1, 10); %linspace(-2, -0.1, 20); %linspace(0.1, 5.1, 50); % This has to match the range in the "*_alphas.py"
perc = [0.1, 0.05, 0.01, 0.2];  

p=2%4%2%1%3%4%1%2%3 %2%1%4%1%4%2 1%2%3; % 2 --> 5%, 1 --> 10%
    
         
for i= 1: iters
    
    %load nerd_1 (batchsize)
    %RD_nerd1_M50(i,:) = csvread(['data/data_PREPARED_at_various_spikein_percentage_asinh5_allcells/arch_16_16_16_8_1/C_b_Nerd/H_and_CBF_perc' num2str(p) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
  %  RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/H3_vs_CBF_5perc_markers_1_2_3_4_5_6/Cb_arch_6_16_16_1/H3_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%       RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10/Cb50_arch_10_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%      RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10/Cb50_arch_10_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10/Cb50_arch_10_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10/Cb10_arch_10_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%       RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb50_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%      RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb50_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%      RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%      RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
% SICK FIRST. a=[0.1, 0.2, ..., 0.9], M=1
%    RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
   RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd1_M50(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Cb5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_lambda_1.0_bs_' num2str(bs(1)) '_nerd_iter' num2str(i) '.csv']);


    
    % load nerd_2 (Lipschitz)
%     RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H3_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_1/H3_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H7_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1/H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%       RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H7_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_32_32_16_1/H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%       RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H7_vs_CBF_5perc_markers_1_2_3_4_5_6/peg/H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H7_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_32_32_16_1_lr0.005/H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%      RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H7_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1_lr0.01/H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H3_H7_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1_lr_0.01/H3_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H3_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1_lr_0.01/H3_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%    RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H3_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_32_32_1_lr_0.005/H3_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%      RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/H7_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_32_32_1_lr0.005/H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%     RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10/Lip5_arch_10_16_16_8_1_lr_0.01_alpha_0.1_5.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
    %  RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10/Lip5_arch_10_16_16_8_1_lr_0.01_alpha_0.1_5.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%       RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10/Lip5_arch_10_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%        RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%         RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%        RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6/Lip5_arch_6_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%         RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%         RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%         RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%          RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%          RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_min2_upto_0/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%          RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_4_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%          RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_32_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%          RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1_SWAP_distributions/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
% SICK FIRST. a=[0.1, 0.2, ..., 0.9], K=1
%          RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
%         RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_20perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
    %NOTE: at w=5% and a=0.9, the Lip-1 required l-r=0.005 to converge...
        RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_upto_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);
        % AND....
        RD_nerd2(i,end) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.005_alpha_0.9_sickFirst/H3_H4_H5_H6_H7_vs_CBF_perc' num2str(p) '_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);




%        %load Healthy-Healthy only!
%        %--------------------------
%        RD_nerd2(i,:) = csvread(['data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_iter' num2str(i) '.csv']);

    
end

RD_M50_mean(p, 1:length(alpha)) = mean(RD_nerd1_M50,1); %save average RD for all alphas, for specific perc.
RD_M50_std(p, 1:length(alpha)) = std(RD_nerd1_M50,1);

RD_Lip_mean(p, 1:length(alpha)) = mean(RD_nerd2,1); %save average RD for all alphas, for specific perc.
RD_Lip_std(p, 1:length(alpha)) = std(RD_nerd2,1);




    
figure(1)
 errorbar(alpha(1:end), RD_M50_mean(p,:), 1.96*RD_M50_std(p,:), 'LineWidth', 2); hold on; % C_b NERD
%errorbar(alpha(1:end), RD_Lip_mean(p,:), 1.96*RD_Lip_std(p,:),'magenta', 'LineWidth', 2); hold on;% Lip Nerd
plot(alpha(1:end), RD_Lip_mean(p,:) ,'magenta--', 'LineWidth', 2);

% legend('Nerd Cb, M=5, [16,16,16,8,1]','Nerd Lip-5, [16,16,16,8,1]');
legend('NERD (Bounded)','NERD (Lipschitz)', 'location', 'north');
xlabel('$$\alpha$$','interpreter','latex')
ylabel('$$\mathcal{R}_{\alpha}(H+CBF 5\% ||H)$$','interpreter','latex'); %<-- HARDCODED w...
%my_str = sprintf('dim: %d, spike-in= %0.2f, batchsize: %d, iters: %d', d, perc(p), bs, iters);
%my_str = sprintf('%d m, w= %d%%', d, perc(p)*100);
%title(my_str)
title('$$ w=0.05 $$','interpreter','latex');  %<-- HARDCODED w...
% ylim([-0.25, 2.6])
% xlim([0.05, 0.95])
set(gca,'FontSize',14); 
%error('checkpoint')




% LOAD ITE computations
%======================

% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_vs_CBF_perc2_markers_1_2_3_4_5_6.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H7_vs_CBF_perc2_markers_1_2_3_4_5_6.mat');
%P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H7_vs_CBF_perc2_markers_1_2_3_4_5_6.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at10perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc1_markers_1_2_3_4_5_6_7_8_9_10.mat');
%  P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10.mat');
%   P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at20perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc4_markers_1_2_3_4_5_6_7_8_9_10.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at10perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc1_markers_1_2_3_4_5_6.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at20perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc4_markers_1_2_3_4_5_6.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at20perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc4_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at10perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc1_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16.mat');
%P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at1perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc3_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16.mat');
%==== SICK FIRST. a=[0.1, 0.2, ..., 0.9] ====
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at10perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc1_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_sick_first.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_sick_first.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at10perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc1_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_sick_first_1minalpha.mat');
%P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_sick_first_1minalpha.mat');
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at20perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc4_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_sick_first_1minalpha.mat');
%P = load('ITE_comps_w02_1iter_without_factor.mat')
%  P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at20perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc4_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_H_first_1minalpha.mat');
P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_H_first_1minalpha.mat');
% k = 50
% P = load('~/Documents/GSRT_NeoiErevnites/diffcyt-evaluations-master/Tasos/ITE_computations_all_cells_asin5_H3_H4_H5_H6_H7_vs_CBF_at5perc/data_files_for_NNs/ITE_computations_H3_H4_H5_H6_H7_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_alpha_0.1_upto_0.9_H_first_1minalpha_k50.mat');


figure(1)
errorbar(P.alpha, P.RD_ITE_mean, 1.96*P.RD_ITE_std, 'black','DisplayName', 'ITE', 'LineWidth', 2);
%plot(P.alpha, P.RD_ITE, 'black','DisplayName', 'ITE', 'LineWidth', 2);



% use export_fig
% cd ~/Documents/GSRT_NeoiErevnites/export_fig-master
% code_dir = pwd;
% addpath(genpath(code_dir));

% %use an external library for saving image files (cropped, high resolution)
%   set(gcf, 'Color', 'w'); %background color set to white
%   export_fig out3.png -m6 %magnifies resolution 6 times...





