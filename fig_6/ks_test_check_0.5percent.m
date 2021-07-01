%
%   check KS test
%
%   x1 is the distribution of R_a(Healthy || Healthy) values, for fixed a
%   x2 is the distribution of R_a(CBF_at_w% || Healthy ) values
%
clear; close all; clc;

%parameters
%alpha_range = linspace(0.1, 1.1, 10); % <- verify this range from .py files!
%warning('changed alpha_range!')
alpha_range = [0.2, 0.5, 0.9]; %0.95; %
%warning('changed iters!')
iters = 100 %60 % %20% 20;   
alpha_pos = 2 %1%    1 --> a=0.2, 2 --> a=0.5, 3 --> a=0.9

%load Healthy-Healthy data
for i = 1: iters
  %k==5
 % P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==1
%  P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==1, half samples
%   P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

%==========================
% ALPHAS = [0.2, 0.5, 0.9]
  %k==1
   P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.2_0.5_0.9/H3_H4_H5_H6_H7_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

   %==========================
% ALPHAS = 0.95
  %k== 0.1
 %  P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip0.1_arch_16_16_16_8_1_lr_0.01_alpha_0.95/H3_H4_H5_H6_H7_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

  x1(i) = P(alpha_pos); % get the R_a value for a specific a
  
  %check for alpha_range
  if length(alpha_range) ~= length(P)
      error('alpha_range is not consistent with the file R_a values')
  end
end
    
%load Healthy-CBF at 1% spike-in data
for i = 1: iters
    %k==5 (1%)
%   P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip5_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1_20iters/H3_H4_H5_H6_H7_vs_CBF_perc3_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
    %k==1 (1%)
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1_60iters/H3_H4_H5_H6_H7_vs_CBF_perc3_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
   %k==1 (0.5%)
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc5_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
   %k==1 (0.2%)
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
    %k==1 (10%), half-samples
%     P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_10perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc1_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
    %k==1 (5%), half samples
%     P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc2_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
    %k==1 (1%), half-samples
%     P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc3_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
    %k==1 (0.5%), half-samples
%      P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc5_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
   %k==1 (0.2%), half-samples
%      P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.1_1.1/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

%==================================================
% ALPHAS = [0.2, 0.5, 0.9]
    %k==1 (1%), 
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.2_0.5_0.9/H3_H4_H5_H6_H7_vs_CBF_perc3_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==1 (0.5%), 
     P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.2_0.5_0.9/H3_H4_H5_H6_H7_vs_CBF_perc5_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==1 (0.2%), 
%  P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.2_0.5_0.9/H3_H4_H5_H6_H7_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

% ALPHAS = 0.95
    %k==0.1 (0.2%), 
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/Lip0.1_arch_16_16_16_8_1_lr_0.01_alpha_0.95/H3_H4_H5_H6_H7_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);



  x2(i) = P(alpha_pos); % get the R_a value for a specific a
end


%plot
figure(1)
plot(x1); hold on;
plot(x2);
legend('R_a (H || H)', 'R_a (CBF at 0.5 || H)'); % legend('R_a (H ||H)', 'R_a (H ||CBF at 0.05)');
xlabel('iterations');
ylabel('divergence')
my_str = sprintf('a=%f', alpha_range(alpha_pos));
title(my_str)

% KS-test
[h,p,k] = kstest2(x1,x2)


% plot histogram
figure(2)
%h = histogram(x1, 10,'Normalization','probability'); hold on; %determine Nbins=10
edges = [0.0056: 0.00086: 0.032];%[0.0056 0.00646 0.00732 0.00818 0.00904 0.0099 0.01076 0.01162 0.01248 0.01334 0.0142 0.01506];  %fix bin locations
h = histogram(x1, edges,'Normalization','probability'); hold on;

%h2 = histogram(x2, 10,'Normalization','probability');
edges2 = [0.0056: 0.0008: 0.032];
h2 = histogram(x2, edges2,'Normalization','probability');
% legend({'$\mathcal{R}_{\alpha}(H ||H)$', '$\mathcal{R}_{\alpha}(H ||H+CBF 0.2\%)$'},'interpreter','latex')
% legend({'$\mathcal{R}_{\alpha}(H ||H)$', '$\mathcal{R}_{\alpha}(H+CBF 0.2\% ||H)$'},'interpreter','latex','Location','north') % <-- CHANGE MANUALLY (according to spike-in)!!!
% legend({'Healthy vs Healthy', 'Healthy + $0.2\%$ CBF  vs Healthy'},'interpreter','latex','Location','northeast') 
legend({'Healthy vs Healthy', 'Healthy + $0.5\%$ diseased  vs Healthy'},'interpreter','latex','Location','northeast') % <-- CHANGE MANUALLY (according to spike-in)!!!
xlabel('$$\hat{\mathcal{R}}_{\alpha}(Q ||P)$$','interpreter','latex', 'FontWeight','bold');
ylabel('histogram (normalized)')
set(gca,'FontSize',14);
%title('\alpha = 0.5'); % <-- CHANGE MANUALLY (according to alpha_pos)!!!

%Nbins = morebins(h) %automatically adds bins if needed
%Nbins = morebins(h2)


xlim([0.005, 0.0325])
ylim([0, 0.3])


% use export_fig
% cd ~/Documents/GSRT_NeoiErevnites/export_fig-master
% code_dir = pwd;
% addpath(genpath(code_dir));

%use an external library for saving image files (cropped, high resolution)
set(gcf, 'Color', 'w'); %background color set to white
export_fig out3.png -m6 %magnifies resolution 6 times...

