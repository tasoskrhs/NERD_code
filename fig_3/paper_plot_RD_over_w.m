%
%   check how w affect nerd estimations, i.e. how far from the exact
%   current densities are:
%                           GMM: p = (1-w)N(mu_1,sigma) + wN(mu_2,sigma/4)   VARYING mu_2
%                           Gaussian: q = N(mu_0, sigma) FIXED
%   compare agaist ITE
%
close all; clear; clc;

% path(path, '~/Documents/GSRT_NeoiErevnites/ITE_toolbox/ITE-0.63_code/code/estimators');
%
% cd ~/Documents/GSRT_NeoiErevnites/ITE_toolbox/ITE-0.63_code/code
% ITE_add_to_path

% use export_fig
% cd ~/Documents/GSRT_NeoiErevnites/export_fig-master
% code_dir = pwd;
% addpath(genpath(code_dir));

% PARAMETERS
%------------
d = 1; 
N= 40000;   
mu_2 = 1; %0; %2; %0; % check for this value for GMM p 
gp = 0.1; % gradient penalty parameter (Lipschitz)

w_range =[0, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1];

%P = load(['data/paper_data/data_from_GMM_varying_w/input_files_N40000_alpha_min10_upto_12/GMM_d_1_params_' num2str(w_range(i)) '.mat']);
%  Distros = load(['data/paper_data/data_from_GMM_varying_w/input_files_N40000_alpha_min10_upto_12/GMM_d_1_data_' num2str(w_range(1)) '.mat']);




%=====================================================
%  Now vary w wrt RD
%=====================================================

bs = 4000;
RD_ITE_N = zeros(length(w_range), 1);

for i = 1:length(w_range)
    disp(i)
    
    % Reference divergence (exact)
    P = load(['data/paper_data/data_from_GMM_varying_w/input_files_N40000_alpha_min10_upto_12/GMM_d_1_params_' num2str(w_range(i)) '.mat']);
    alpha = P.alpha;
    RD_exact = P.RD_exact;
    
    % choose alpha == 0.5
    a_Ren_pos = 14%16%1%6%20%25%14; % 14 -> a=0.5 (position of the vector with alpha's)
    alpha(a_Ren_pos)
    
    % keep the exact Renyi divergence, at already chosen mu_2, w
    RD_exact_w(i) = RD_exact(a_Ren_pos);
    
    
    
    %load nerd_1 (batchsize)
%     RD_nerd1_M50 = csvread(['data/paper_data/data_from_GMM_varying_w/Cb50_arch_1_16_16_8_1/GMM_d_1_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);
%     RD_nerd1_M50 = csvread(['data/paper_data/data_from_GMM_varying_w/Cb50_arch_1_8_8_4_1/GMM_d_1_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);
%   RD_nerd1_M50 = csvread(['data/paper_data/data_from_GMM_varying_w/Cb5_arch_1_8_8_4_1/GMM_d_1_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);
  RD_nerd1_M50 = csvread(['data/paper_data/data_from_GMM_varying_w/Cb1_arch_1_16_16_8_1/GMM_d_1_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);

    % load nerd_2 (Lipschitz)
%     RD_nerd2 = csvread(['data/paper_data/data_from_GMM_varying_w/Lip5_arch_1_16_16_8_1/GMM_d_1_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);
%     RD_nerd2 = csvread(['data/paper_data/data_from_GMM_varying_w/Lip5_arch_1_8_8_4_1/GMM_d_1_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);
    RD_nerd2 = csvread(['data/paper_data/data_from_GMM_varying_w/Lip1_arch_1_16_16_8_1/GMM_d_1_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bs(1)) '_nerd_' num2str(w_range(i)) '.csv']);
    
    % mean values    
    RD_M50_of_w_mean(i) = mean( RD_nerd1_M50(a_Ren_pos, 1:end) );
    RD_M50_of_w_std(i) = std( RD_nerd1_M50(a_Ren_pos, 1:end) );
    RD_M50_of_w(:,i) = RD_nerd1_M50(a_Ren_pos, 1:end); %for boxplot

        
    RD2_of_w_mean(i) = mean( RD_nerd2(a_Ren_pos, 1:end) );
    RD2_of_w_std(i) = std( RD_nerd2(a_Ren_pos, 1:end) );
    RD2_of_w(:,i) = RD_nerd2(a_Ren_pos, 1:end); %for boxplot
    

    %load different distribution files and compute with ITE as well.
    Distros = load(['data/paper_data/data_from_GMM_varying_w/input_files_N40000_alpha_min10_upto_12/GMM_d_1_data_' num2str(w_range(i)) '.mat']);
    
    % Renyi divergence using ITE (using all distribution samples)
    %----------------------------------------------------------
    mult = 1; %multiplicative constant is important
    co = DRenyi_kNN_k_initialization(mult);%initialize the estimator
    co.alpha = alpha(a_Ren_pos); % specify alpha (default: a=0.99)
    co.k = 20; % kNN k (default: k=3)    
    
    
    
    %   ===================
    %  ITE SUBSAMPLING ESTIMATIONS HERE...
    %  =============================
    
    % run ITE on 80% random subsets of Distros
    %-----------------------------------------
    iters = size(RD_nerd1_M50,2);
    samples = size(Distros.x,1)
    
    for it= 1: iters
        idx = randperm(samples);
        last_sample = ceil(samples*0.8); % pick perc% of the samples
        tmp_x = Distros.x(idx(1:last_sample),:); %subsample...
        idx = randperm(samples);
        last_sample = ceil(samples*0.8); % pick perc% of the samples
        tmp_y = Distros.y(idx(1:last_sample),:);
        
        tmp_RD_ITE(it) = DRenyi_kNN_k_estimation(tmp_x', tmp_y', co); % NOTE layout of data should be DIM x samples!!
        tmp_RD_ITE(it) = tmp_RD_ITE(it)/alpha(a_Ren_pos);     % NOTE: Result from ITE needs 1/alpha normalization (ITE-0.63_documentation.pdf eq. (49))

    end
    RD_ITE_mean(i) = mean(tmp_RD_ITE);
    RD_ITE_std(i) = std(tmp_RD_ITE);
    RD_ITE_of_w(:,i) = tmp_RD_ITE; % for boxplot
    
    
end


figure(1)
plot(w_range, RD_exact_w,'r', 'LineWidth', 3 ); hold on;
legend('Exact')
errorbar(w_range(1:end), RD_M50_of_w_mean, 1.96*RD_M50_of_w_std, 'LineWidth', 2, 'DisplayName','Cb-1');

errorbar(w_range(1:end), RD2_of_w_mean, 1.96*RD2_of_w_std, 'green', 'LineWidth', 2, 'DisplayName','Lip-1'); hold on;

% ITE
% for i = 1:length(N_range)
%     
%     my_str = sprintf('RD ITE (kNN), N= %d', N_range(i));
%     plot(x_sup, RD_ITE_N(i) *ones(size(x_sup)), 'LineWidth', 2, 'DisplayName', my_str)
% end
% plot(N_range(1:end), RD_ITE_N(:)', 'black', 'LineWidth', 2, 'DisplayName', 'RD ITE (kNN)')
errorbar(w_range, RD_ITE_mean, 1.96*RD_ITE_std, 'black', 'LineWidth', 2, 'DisplayName', 'RD ITE (kNN, k=20)');



iters = size(RD_nerd1_M50,2);
my_str = sprintf('GMM, ep: %d, a= %0.2f, iters: %d, mu_2= %3.1f, BS=%d', 20000, alpha(a_Ren_pos), iters, mu_2, bs);
title(my_str)
ylim([-0.01, 0.025]);%([-0.1, 0.1]);

%xticks([5000, 25000, 40000, 75000, 150000])
%xticklabels({'5','25','40','75','150'})
xlabel('w')

xlim([w_range(1), w_range(end)]);

ylabel('Renyi Divergence')
set(gca,'FontSize',12);


%replot with subplot (each against the exact)
%--------------------------------------------
figure(2)
subplot(1,3,1)
plot(w_range, RD_exact_w,'r', 'LineWidth', 3 ); hold on;
legend('Exact')
errorbar(w_range(1:end), RD_M50_of_w_mean, 1.96*RD_M50_of_w_std, 'LineWidth', 2, 'DisplayName','Cb-1');
ylim([-0.001, 0.02]);
xlabel('w')
xlim([w_range(1), w_range(end)]);
ylabel('Renyi Divergence')
xticks([0, 0.006, 0.025, 0.05, 0.1])
xticklabels({'0',  '0.006', '0.025', '0.05', '0.1'})
set(gca,'FontSize',14);


subplot(1,3,2)
plot(w_range, RD_exact_w,'r', 'LineWidth', 3 ); hold on;
legend('Exact')
errorbar(w_range(1:end), RD2_of_w_mean, 1.96*RD2_of_w_std, 'green', 'LineWidth', 2, 'DisplayName','Lip-1'); hold on;
ylim([-0.001, 0.02]);
xlabel('w')
xlim([w_range(1), w_range(end)]);
xticks([0, 0.006, 0.025, 0.05, 0.1])
xticklabels({'0',  '0.006', '0.025', '0.05', '0.1'})
%ylabel('Renyi Divergence')
set(gca,'FontSize',14);

subplot(1,3,3)
plot(w_range, RD_exact_w,'r', 'LineWidth', 3 ); hold on;
legend('Exact')
errorbar(w_range, RD_ITE_mean, 1.96*RD_ITE_std, 'black', 'LineWidth', 2, 'DisplayName', 'RD ITE (kNN, k=20)');
ylim([-0.001, 0.02]);
xlabel('w')
xlim([w_range(1), w_range(end)]);
xticks([0, 0.006, 0.025, 0.05, 0.1])
xticklabels({'0',  '0.006', '0.025', '0.05', '0.1'})
%ylabel('Renyi Divergence')
set(gca,'FontSize',14);



%==============================================
figure(3)

subplot(1,3,1)
boxplot(RD_M50_of_w,'Labels',{'0', '0.003', '0.006', '0.012', '0.025', '0.05', '0.1'},'Whisker', 3); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
plot([1:1:7], RD_exact_w,'r', 'LineWidth', 2 );
ytickformat('%.1f')
xlabel('$$w$$','interpreter','latex')
%ylabel('$$\mathcal{R}_{\alpha}$$','interpreter','latex', 'FontWeight','bold');
ylabel('$$\mathcal{R}_{\alpha}(Q||P)$$','interpreter','latex')
ylim([-0.007, 0.02]);
set(gca,'FontSize',14);

subplot(1,3,2)
boxplot(RD2_of_w,'Labels',{'0', '0.003', '0.006', '0.012', '0.025', '0.05', '0.1'},'Whisker', 5); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
plot([1:1:7], RD_exact_w,'r', 'LineWidth', 2 ); 
ytickformat('%.1f')
xlabel('$$w$$','interpreter','latex')
ylim([-0.007, 0.02]);
set(gca,'FontSize',14);

subplot(1,3,3)
boxplot(RD_ITE_of_w,'Labels',{'0', '0.003', '0.006', '0.012', '0.025', '0.05', '0.1'},'Whisker', 4); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
plot([1:1:7], RD_exact_w,'r', 'LineWidth', 2 ); 
ytickformat('%.1f')
xlabel('$$w$$','interpreter','latex')
ylim([-0.007, 0.02]);
set(gca,'FontSize',14);

% %use an external library for saving image files (cropped, high resolution)
% set(gcf, 'Color', 'w'); %background color set to white
% export_fig out3.png -m6 %magnifies resolution 6 times...


