%
%
%   paper figure for NERD over \rho: two multi-Dim Gaussians
%   comparison against ITE
%   
close all; clear; clc;

% path(path, '~/Documents/GSRT_NeoiErevnites/ITE_toolbox/ITE-0.63_code/code/estimators');
%
% cd ~/Documents/GSRT_NeoiErevnites/ITE_toolbox/ITE-0.63_code/code
% ITE_add_to_path

% PARAMETERS
%------------
d = 4; %50; %20; % 
bs = 4000; %40000; % 20000; %20000; % 
%warning('changed rho!')
rho = [0.1: 0.2: 0.9]; %-0.7; %[0.1: 0.2: 0.5]; %[-0.9: 0.2: 0.9]; % [-0.9: 0.2: -0.1];  %
    
gp = 0.1; % gradient penalty parameter (Lipschitz)
bsofnerd2 = 4000; %40000; %   
tic;

for i = 1:length(rho) 
     disp(i)
 
   P = load(['data/varying_rho_Sigma1_eye_2/d_4/input_N50000_dim4/gaussian_d_' num2str(d) '_params_' num2str(rho(i)) '.mat']);
%      P = load(['data/varying_rho_Sigma1_eye_2/d_20/input_N150000_dim20/gaussian_d_' num2str(d) '_params_' num2str(rho(i)) '.mat']);
%    P = load(['data/varying_rho_Sigma1_eye_2/d_50/input_N300000_dim50/gaussian_d_' num2str(d) '_params_' num2str(rho(i)) '.mat']);
%     P = load(['data/gaussian_d_' num2str(d) '_params_' num2str(rho(i)) '.mat']);


    alpha = P.alpha;
    RD_exact = P.RD_exact;

    Distros = load(['data/varying_rho_Sigma1_eye_2/d_4/input_N50000_dim4/gaussian_d_' num2str(d) '_data_' num2str(rho(i)) '.mat']);
%      Distros = load(['data/varying_rho_Sigma1_eye_2/d_20/input_N150000_dim20/gaussian_d_' num2str(d) '_data_' num2str(rho(i)) '.mat']);
%      Distros = load(['data/varying_rho_Sigma1_eye_2/d_50/input_N300000_dim50/gaussian_d_' num2str(d) '_data_' num2str(rho(i)) '.mat']);
%     Distros = load(['data/gaussian_d_' num2str(d) '_data_' num2str(rho(i)) '.mat']);
 
     % choose alpha == 0.5 and plot RD vs t
    a_Ren_pos = 1; %for 1 we get a == 0.5  (position of the vector with alpha's)
     
    
    % load nerd_1 (batchsize)
     RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_4/paper_data_arch_4_8_8_4_1/N_50k_BATCHSIZE_M50_discr_varying_rho_d4_epochs40k_BS_4k/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
%      RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_BATCHSIZE_M50_discr_varying_rho_d20_epochs20k_BS_4k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
%      RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_50/arch_50_32_32_16_1/test_functions/N_300000_BATCHSIZE_M50_discr_varying_rho_d50_epochs40k_BS_40k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
%             RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_50/arch_50_32_16_1/N_300k_batchsize/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
  %subtract means  
%      RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_BATCHSIZE_M50_discr_varying_rho_d20_epochs20k_BS_4k_arch32_subtract_means/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
  % M=20  
%     RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_BATCHSIZE_M20_discr_varying_rho_d20_epochs20k_BS_4k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
  % M=10  
%     RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_BATCHSIZE_M10_discr_varying_rho_d20_epochs20k_BS_4k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
 % M=5  
%      RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_BATCHSIZE_M5_discr_varying_rho_d20_epochs20k_BS_4k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);
% M=20, d=50
%       RD_nerd1_M50 = csvread(['data/varying_rho_Sigma1_eye_2/d_50/arch_50_32_32_16_1/test_functions/N_300000_BATCHSIZE_M20_discr_varying_rho_d50_epochs40k_BS_40k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(rho(i)) '.csv']);

    % load nerd_2 (Lipschitz)
     RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_4/paper_data_arch_4_8_8_4_1/N_50k_Lip_K5_varying_rho_d4_epochs40k_BS_4k/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
%     RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_Lip_K5_varying_rho_d20_epochs20k_BS_4k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
%    RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_50/arch_50_32_32_16_1/test_functions/N_300000_Lip_K5_varying_rho_d50_epochs40k_BS_40k_arch_32/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
%      RD_nerd2 = csvread(['data/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
  %subtract means  
%     RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_Lip_K5_varying_rho_d20_epochs20k_BS_4k_arch32_subtract_means/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
  %old data, D-20 K=1  
  %  RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_20/paper_data_arch_20_32_32_16_1/N_150k_Lip_K1_varying_rho_d20_epochs20k_BS_4k_arch32/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
%  old data, D-50, K=1
%       RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_50/arch_50_32_32_16_1/test_functions/N_300000_Lip_K1_varying_rho_d50_epochs40k_BS_40k_arch_32/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);
%  old data, D-50, K=20
%     RD_nerd2 = csvread(['data/varying_rho_Sigma1_eye_2/d_50/arch_50_32_32_16_1/test_functions/N_300000_Lip_K20_varying_rho_d50_epochs40k_BS_40k_arch_32/gaussian_d_' num2str(d) '_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bsofnerd2(1)) '_nerd_' num2str(rho(i)) '.csv']);

    

    RD_M50_of_t_mean(i) = mean( RD_nerd1_M50(a_Ren_pos, 1:end) );
    RD_M50_of_t_std(i) = std( RD_nerd1_M50(a_Ren_pos, 1:end) );
    RD_M50_of_t(:,i) = RD_nerd1_M50(a_Ren_pos, 1:end); %for boxplot
    RD_M50_of_t_median(i) = median( RD_nerd1_M50(a_Ren_pos, 1:end) );
    
    % keep the exact Renyi distance
    RD_exact_t(i) = RD_exact(a_Ren_pos);
    
    
    % Lipschitz values
    RD2_of_t_mean(i) = mean( RD_nerd2(a_Ren_pos, 1:end) );
    RD2_of_t_std(i) = std( RD_nerd2(a_Ren_pos, 1:end) );
    RD2_of_w(:,i) = RD_nerd2(a_Ren_pos, 1:end); %for boxplot
    RD2_of_t_median(i) = median( RD_nerd2(a_Ren_pos, 1:end) );

    
    % Renyi divergence using ITE (using all distribution samples)
    %----------------------------------------------------------
    mult = 1; %multiplicative constant is important
    co = DRenyi_kNN_k_initialization(mult);%initialize the estimator
    co.alpha = alpha(a_Ren_pos); % specify alpha (default: a=0.99)
    co.k = 20; % kNN k (default: k=3)
    %initialize the divergence (’D’) estimator (’Renyi_kNN_k’)
    
    % run ITE on 80% random subsets of Distros
    %-----------------------------------------
    samples = size(Distros.x,1)
    iters = size(RD_nerd2,2);
    % warning('hardcoded iters')
    % iters=5%10
    %warning('divide with 1/1-a!!')
    %par
    for it= 1: iters
        idx = randperm(samples);
        last_sample = ceil(samples*0.8); % pick perc% of the samples
        tmp_x = Distros.x(idx(1:last_sample),:); %subsample...
        idx = randperm(samples);
        last_sample = ceil(samples*0.8); % pick perc% of the samples
        tmp_y = Distros.y(idx(1:last_sample),:);
        
        tmp_RD_ITE(it) = DRenyi_kNN_k_estimation(tmp_x', tmp_y', co); % NOTE layout of data should be DIM x samples!!
        tmp_RD_ITE(it) = tmp_RD_ITE(it)/alpha(a_Ren_pos); %NOTE: Result from ITE needs 1/alpha normalization (ITE-0.63_documentation.pdf eq. (49))
%         tmp_RD_ITE(it) = tmp_RD_ITE(it)/(1-alpha(a_Ren_pos)); %NOTE: Result from ITE needs 1/alpha normalization (ITE-0.63_documentation.pdf eq. (49))

    end  
    RD_ITE_mean(i) = mean(tmp_RD_ITE);
    RD_ITE_std(i) = std(tmp_RD_ITE);
    RD_ITE_of_w(:,i) = tmp_RD_ITE; % for boxplot
   
     t=toc;
     sprintf('time taken: %f [min]',t/60)

end

% %load saved ITE computations that took hours
%---------------------------------------------
% d=4;
%*****
% load('ITE_comps_D4_iters10_N50k_rho_min0_9_upto_0_9.mat');

% d=20
%*****
% Z =  load('ITE_comps_D20_iters10_N150k_rho_pos_for_BOXPLOT.mat');
% RD_ITE_of_w = Z.RD_ITE_of_w;
%--------
% load('ITE_comps_D20_iters10_N150k_rho_neg.mat')
%load('ITE_comps_D20_iters10_N150k_rho_pos.mat');
% RD_ITE_mean = [RD_ITE_mean, Z.RD_ITE_mean]
%  RD_ITE_std = [RD_ITE_std, Z.RD_ITE_std]
 
% % d=50
% %*****
% load('ITE_comps_D50_iters5_N300k_rho_pos_for_BOXPLOT.mat');
%---------
%  load('ITE_comps_D50_iters10_N300k_rho_neg.mat')
%  Z = load('ITE_comps_D50_iters10_N300k_rho_pos.mat');
% %****** CONCATENATE ****
%  RD_ITE_mean = [RD_ITE_mean, Z.RD_ITE_mean]
%  RD_ITE_std = [RD_ITE_std, Z.RD_ITE_std]


    figure(2)
    %exact solution
    plot(rho(1:end), RD_exact_t,'-r', 'LineWidth', 3);  hold on;

    errorbar(rho(1:end), RD_M50_of_t_mean, 1.96*RD_M50_of_t_std, 'blue', 'LineWidth', 2);

    % replot nerd2 (Lipschitz)
     errorbar(rho(1:end), RD2_of_t_mean, 1.96*RD2_of_t_std, 'green', 'LineWidth', 2); hold on;
    
    %plot(rho(:), RD_ITE, 'black');  % Renyi divergence using ITE
    errorbar(rho(:), RD_ITE_mean, 1.96*RD_ITE_std, 'black', 'LineWidth', 2);  % Renyi divergence using ITE


    xlabel('\rho')
    ylabel('Renyi Divergence')
%     my_str = sprintf('%d-D,samples= %d, BS: %d, a= %0.2f, iters: %d', d, samples, bs, alpha(a_Ren_pos), iters);
%     title(my_str)

    title('two 50-D Gaussians, \alpha = 0.5') %<---- CHANGE dim manually! (because tex interpreter doesn't work with sprintf for \alpha)
    my_str = sprintf('Renyi Div. ITE (kNN), k= %d', co.k);
    legend('Exact Renyi Divergence', 'NERD (batchsize), M=50','NERD (5-Lipschitz)', my_str);
    set(gca,'FontSize',12); 
    %xlim([-0.95, 0.95]);
    xticks([-0.9 -0.5 -0.1 0.1 0.5 0.9]) 

    xlim([-0.65 0.65]);
    ylim([0, 7]);%([-10, 80]);%([0, 4]);%
    
    %xlim([0, rho(end)]);
    %ylim([0, 5.5]);%ylim([0, 10.5]); % relevant to the 2*M discriminator cutoff
    
    
    
    % ============== replot with subplot ==============
    figure(3)

    subplot(1,3,1)
    boxplot(RD_M50_of_t,'Labels',{'0.1', '0.3', '0.5', '0.7', '0.9'},'Whisker', 10); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
    plot([1:1:5], RD_exact_t,'r*-', 'LineWidth', 2 );
    ytickformat('%.1f')
    xlabel('$$\rho$$','interpreter','latex')
    ylabel('$$\mathcal{R}_{\alpha}(Q||P)$$','interpreter','latex')
    ylim([0, 3]);%([0, 40]);%([0, 14]);%
    title('NERD (Bounded)') %<---- CHANGE dim manually! (because tex interpreter doesn't work with sprintf for \alpha)
    set(gca,'FontSize',14);
    
    subplot(1,3,2)
    boxplot(RD2_of_w,'Labels',{'0.1', '0.3', '0.5', '0.7', '0.9'},'Whisker', 10); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
    plot([1:1:5], RD_exact_t,'r*-', 'LineWidth', 2 );
    ytickformat('%.1f')
    xlabel('$$\rho$$','interpreter','latex')
    %ylabel('$$\mathcal{R}_{\alpha}(Q||P)$$','interpreter','latex')
    ylim([0, 40]);%([0, 3]);%([0, 14]);%
    title('NERD (Lipschitz)')
    set(gca,'FontSize',14);   
     
    subplot(1,3,3)
    boxplot(RD_ITE_of_w,'Labels',{'0.1', '0.3', '0.5', '0.7', '0.9'},'Whisker', 10); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
    plot([1:1:5], RD_exact_t,'r*-', 'LineWidth', 2 );
    ytickformat('%.1f')
    xlabel('$$\rho$$','interpreter','latex')
    %ylabel('$$\mathcal{R}_{\alpha}(Q||P)$$','interpreter','latex')
    title('ITE')
    ylim([0, 3]);%([0, 40]);%([0, 14]);%
    set(gca,'FontSize',14);   
    
    
% use export_fig
% cd ~/Documents/GSRT_NeoiErevnites/export_fig-master
% code_dir = pwd;
% addpath(genpath(code_dir));

%use an external library for saving image files (cropped, high resolution)
% set(gcf, 'Color', 'w'); %background color set to white
% export_fig out3.png -m6 %magnifies resolution 6 times...


%end
