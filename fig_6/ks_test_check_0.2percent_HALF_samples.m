%
%   check KS test
%
%   x1 is the distribution of R_a(Healthy || Healthy) values, for fixed a
%   x2 is the distribution of R_a(CBF_at_w%|| Healthy ) values
%
clear; close all; clc;

%parameters
%alpha_range = linspace(0.1, 1.1, 10); % <- verify this range from .py files!
alpha_range = [0.5, 0.8]; %0.95 %[0.9, 0.95, 1.01]; %

iters = 60 %20% 20;   
alpha_pos = 1  %1 %    1 --> a=0.5, 2 --> a=0.8

%load Healthy-Healthy data
for i = 1: iters
  
%==========================
% HALF samples, ALPHAS = [0.5, 0.8]
  %k==1
  P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.5_0.8_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==0.2, a=1.5
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip0.2_arch_16_16_16_8_1_lr_0.01_alpha_1.5_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==0.1, a=0.95
%     P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip0.1_arch_16_16_16_8_1_lr_0.01_alpha_0.95_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==0.05, a=0.95
%     P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip0.05_arch_16_16_16_8_1_lr_0.01_alpha_0.95_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

%==========================
% HALF samples, ALPHAS = [0.9, 0.95, 1.01]
%k== 1
 %    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.9_0.95_1.01_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_Healthy_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

   
  x1(i) = P(alpha_pos); % get the R_a value for a specific a
  
%   %check for alpha_range
%   if length(alpha_range) ~= length(P)
%       error('alpha_range is not consistent with the file R_a values')
%   end
end
    
%load Healthy-CBF at 1% spike-in data
for i = 1: iters
%==================================================
% HALF SAMPLES, ALPHAS = [0.5, 0.8]
    %k==1 (1%), 
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_1perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.5_0.8_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc3_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==1 (0.5%), 
%  P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.5perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.5_0.8_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc5_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==1 (0.2%), 
 P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.5_0.8_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

  %k==0.2, a=1.5 (w=0.2%), 
%   P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip0.2_arch_16_16_16_8_1_lr_0.01_alpha_1.5_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==0.1, a=0.95 (w=0.2%), 
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip0.1_arch_16_16_16_8_1_lr_0.01_alpha_0.95_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);
  %k==0.05, a=0.95 (w=0.2%), 
%    P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip0.05_arch_16_16_16_8_1_lr_0.01_alpha_0.95_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

  %k== 1, a = [0.9, 0.95, 1.01]
%  P = csvread(['~/Documents/GSRT_NeoiErevnites/multivar_gaussian_various/diffcyt_blast_cell_data/data/ORIGINAL_DATA/all_Healthy_vs_CBF_0.2perc_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16/N_39000/Lip1_arch_16_16_16_8_1_lr_0.01_alpha_0.9_0.95_1.01_sick_first/H3_H4_H5_H6_H7_HALF_SAMPLES_vs_CBF_perc6_markers_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16lambda_1.0_gp_0.1_bs_4000_nerd_iter' num2str(i) '.csv']);

 

  x2(i) = P(alpha_pos); % get the R_a value for a specific a
end


%plot
figure(1)
plot(x1); hold on;
plot(x2);
legend('R_a (H || H)', 'R_a (CBF at 0.05 || H)'); % legend('R_a (H ||H)', 'R_a (H ||CBF at 0.05)');
xlabel('iterations');
ylabel('divergence')
my_str = sprintf('a=%f', alpha_range(alpha_pos));
title(my_str)

% KS-test
[h,p,k] = kstest2(x1,x2)


% plot histogram
figure(2)
%h = histogram(x1, 10,'Normalization','probability'); hold on; %determine Nbins=10
 edges = [0.02: 0.0012: 0.052];
% edges = [0.: 0.0025: 0.045]; % for a>=0.8
h = histogram(x1, edges,'Normalization','probability'); hold on;

%h2 = histogram(x2, 10,'Normalization','probability');
%edges2 = [0.0056: 0.0008: 0.032];
edges2=edges;
h2 = histogram(x2, edges2,'Normalization','probability');
% legend({'$\mathcal{R}_{\alpha}(H ||H)$', '$\mathcal{R}_{\alpha}(H ||H+CBF 0.2\%)$'},'interpreter','latex')
% legend({'$\mathcal{R}_{\alpha}(H ||H)$', '$\mathcal{R}_{\alpha}(H+CBF 0.2\% ||H)$'},'interpreter','latex','Location','north') % <-- CHANGE MANUALLY (according to spike-in)!!!
legend({'Healthy vs Healthy', 'Healthy + $0.2\%$ diseased  vs Healthy'},'interpreter','latex','Location','northeast') % <-- CHANGE MANUALLY (according to spike-in)!!!
xlabel('$$\hat{\mathcal{R}}_{\alpha}(Q ||P)$$','interpreter','latex', 'FontWeight','bold');
ylabel('histogram (normalized)')
set(gca,'FontSize',14);
%title('\alpha = 0.8'); % <-- CHANGE MANUALLY (according to alpha_pos)!!!

%Nbins = morebins(h) %automatically adds bins if needed
%Nbins = morebins(h2)


xlim([0.02, 0.052])
ylim([0, 0.3])


% use export_fig
% cd ~/Documents/GSRT_NeoiErevnites/export_fig-master
% code_dir = pwd;
% addpath(genpath(code_dir));

%use an external library for saving image files (cropped, high resolution)
set(gcf, 'Color', 'w'); %background color set to white
export_fig out3.png -m6 %magnifies resolution 6 times...

