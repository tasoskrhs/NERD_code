%
%   check how BS and N affect nerd estimations, i.e. how far from the exact
%   current densities are:
%                           GMM: p = (1-w)N(mu_1,sigma) + wN(mu_2,sigma/4)   VARYING mu_2
%                           Gaussian: q = N(mu_0, sigma) FIXED
%
% *_boxplot: plots are plotted with boxplot() instead of errorbar()

close all; clear; clc;

% path(path, '~/Documents/GSRT_NeoiErevnites/ITE_toolbox/ITE-0.63_code/code/estimators');
%
% cd ~/Documents/GSRT_NeoiErevnites/ITE_toolbox/ITE-0.63_code/code
% ITE_add_to_path

% PARAMETERS
%------------
d = 1; 
w= 0.2;   
mu_2 = 1; %0; %2; %0; % check for this value for GMM p 

gp = 0.1; % gradient penalty parameter (Lipschitz)

% REFERENCE divergence: (exact is present) in every file and ITE for this N
%['data/paper_data/mu2_1_arch_16/N40000_a_0.5_and_3.7/input_data/GMM_d_1_params_' num2str(w) '.mat']
% P = load(['data/paper_data/mu2_1_arch_16/N40000_a_0.5_and_3.7/input_data/GMM_d_1_params_' num2str(w) '.mat']);
P = load(['data/paper_data/mu2_1_arch_16/N5000_a_0.5_and_3.7/input_data/GMM_d_1_params_' num2str(w) '.mat']);
alpha = P.alpha;
RD_exact = P.RD_exact;

% choose alpha == 0.5 
a_Ren_pos = 1 %2; % 1 -> a=0.5, 2->a=3.7  (position of the vector with alpha's)
alpha(a_Ren_pos)

% keep the exact Renyi distance, at already chosen mu_0
RD_exact_t = RD_exact(a_Ren_pos);
        
% Distros = load(['data/paper_data/mu2_1_arch_16/N40000_a_0.5_and_3.7/input_data/GMM_d_1_data_' num2str(w) '.mat']);
Distros = load(['data/paper_data/mu2_1_arch_16/N5000_a_0.5_and_3.7/input_data/GMM_d_1_data_' num2str(w) '.mat']);
    


% Renyi divergence using ITE (using all distribution samples)
%----------------------------------------------------------
mult = 1; %multiplicative constant is important
co = DRenyi_kNN_k_initialization(mult);%initialize the estimator
co.alpha = alpha(a_Ren_pos); % specify alpha (default: a=0.99)
co.k = 20; % kNN k (default: k=3)
%initialize the divergence (’D’) estimator (’Renyi_kNN_k’)
RD_ITE = DRenyi_kNN_k_estimation(Distros.x', Distros.y', co); % NOTE layout of data should be DIM x samples!!
% NOTE: Result from ITE needs 1/alpha normalization (ITE-0.63_documentation.pdf eq. (49))
RD_ITE = RD_ITE/alpha(a_Ren_pos); % in effect RD = \frac{1}{a(a-1)}\int...



%=====================================================
%  Now vary sample-size N wrt RD
%=====================================================

N_range = [5000, 10000, 20000, 50000, 100000, 150000];%[5000, 25000, 40000, 75000, 150000];%[2500, 5000, 25000, 40000, 75000, 150000];%[25000, 40000, 75000, 150000];
bs = 4000;
RD_ITE_N = zeros(length(N_range), 1);

for i = 1:length(N_range)
    disp(i)
      % load nerd_1 (batchsize)
    RD_nerd1_M50 = csvread(['data/paper_data/mu2_1_arch_16/N' num2str(N_range(i)) '_a_0.5_and_3.7/GMM_d_1_lambda_1.0_bs_' num2str(bs(1)) '_nerd_' num2str(w) '.csv']);
    
    % load nerd_2 (Lipschitz)
    RD_nerd2 = csvread(['data/paper_data/mu2_1_arch_16/N' num2str(N_range(i)) '_a_0.5_and_3.7/GMM_d_1_lambda_1.0_gp_' num2str(gp(1)) '_bs_' num2str(bs(1)) '_nerd_' num2str(w) '.csv']);
    
    RD_M50_of_t_mean(i) = mean( RD_nerd1_M50(a_Ren_pos, 1:end) );
    RD_M50_of_t_std(i) = std( RD_nerd1_M50(a_Ren_pos, 1:end) );
    RD_M50_of_N(:,i) = RD_nerd1_M50(a_Ren_pos, 1:end); %for boxplot
    
    % Lipschitz values
    RD2_of_t_mean(i) = mean( RD_nerd2(a_Ren_pos, 1:end) );
    RD2_of_t_std(i) = std( RD_nerd2(a_Ren_pos, 1:end) );
    RD2_of_N(:,i) = RD_nerd2(a_Ren_pos, 1:end); %for boxplot
    
    
    %load different distribution files and compute with ITE as well.
    Distros = load(['data/paper_data/mu2_1_arch_16/N' num2str(N_range(i)) '_a_0.5_and_3.7/input_data/GMM_d_1_data_' num2str(w) '.mat']);
    RD_ITE = DRenyi_kNN_k_estimation(Distros.x', Distros.y', co); % NOTE layout of data should be DIM x samples!!
    % NOTE: Result from ITE needs 1/alpha normalization (ITE-0.63_documentation.pdf eq. (49))
    RD_ITE_N(i) = RD_ITE/alpha(a_Ren_pos);
    
    %   ===================
    %  ITE SUBSAMPLING ESTIMATIONS HERE...
    %  =============================
    
    % run ITE on 80% random subsets of Distros
    %-----------------------------------------
    iters = size(RD_nerd1_M50,2);
    samples = size(Distros.x,1)
%     warning('hardcoded iters')
%     iters=2

    for it= 1: iters
        idx = randperm(samples);
        last_sample = ceil(samples*0.8); % pick perc% of the samples
        tmp_x = Distros.x(idx(1:last_sample),:); %subsample...
        idx = randperm(samples);
        last_sample = ceil(samples*0.8); % pick perc% of the samples
        tmp_y = Distros.y(idx(1:last_sample),:);
        
        tmp_RD_ITE(it) = DRenyi_kNN_k_estimation(tmp_x', tmp_y', co); % NOTE layout of data should be DIM x samples!!
        tmp_RD_ITE(it) = tmp_RD_ITE(it)/alpha(a_Ren_pos);
    end  
    RD_ITE_mean(i) = mean(tmp_RD_ITE);
    RD_ITE_std(i) = std(tmp_RD_ITE);
    RD_ITE_of_N(:,i) = tmp_RD_ITE; % for boxplot
    
   
    
    
end


figure(1);
x_sup = (0: 1 : N_range(end)+2);
subplot(1,3,1);
boxplot(RD_M50_of_N,'Labels',{'5','10','20','50','100','150'},'Whisker', 2); hold on;  % CHECK LABELS! ,{'5','25','40','75','150'}
plot(x_sup, RD_exact_t*ones(size(x_sup)),'r', 'LineWidth', 3 );
title('NERD (Bounded)')%('C_b-1 neRd')
ylim([0.045 0.09])
xlabel('$$N\quad [\times 10^3]$$','interpreter','latex')
ylabel('$$\mathcal{R}_{\alpha}(Q||P)$$','interpreter','latex')
set(gca,'FontSize',14);

subplot(1,3,2);
boxplot(RD2_of_N,'Labels',{'5','10','20','50','100','150'},'Whisker', 4); hold on; %longer whisker to remove outliers (red crosses)
plot(x_sup, RD_exact_t*ones(size(x_sup)),'r', 'LineWidth', 3 ); 
title('NERD (Lipschitz)')%('Lip-1 neRd')
xlabel('$$N\quad [\times 10^3]$$','interpreter','latex')
ylim([0.045 0.09])
set(gca,'FontSize',14);

subplot(1,3,3);
boxplot(RD_ITE_of_N,'Labels',{'5','10','20','50','100','150'},'Whisker', 4); hold on;
plot(x_sup, RD_exact_t*ones(size(x_sup)),'r', 'LineWidth', 3 ); 
title('ITE')
xlabel('$$N\quad [\times 10^3]$$','interpreter','latex')
ylim([0.045 0.09])
set(gca,'FontSize',14);


%use an external library for saving image files (cropped, high resolution)
% set(gcf, 'Color', 'w'); %background color set to white
% export_fig out3.png -m6 %magnifies resolution 6 times...


figure(2)
x_sup = (0: 1 : N_range(end)+2);
plot(x_sup, RD_exact_t*ones(size(x_sup)),'r', 'LineWidth', 3 ); hold on;
legend('Exact')
errorbar(N_range(1:end), RD_M50_of_t_mean, 1.96*RD_M50_of_t_std, 'LineWidth', 2, 'DisplayName','D_{NN} C_b-1');

errorbar(N_range(1:end), RD2_of_t_mean, 1.96*RD2_of_t_std, 'green', 'LineWidth', 2, 'DisplayName','D_{NN} Lip-5'); hold on;


errorbar(N_range(), RD_ITE_mean, 1.96*RD_ITE_std, 'black', 'LineWidth', 2, 'DisplayName', 'RD ITE (kNN)');



iters = size(RD_nerd1_M50,2);
my_str = sprintf('dim: %d, epochs: %d, a= %0.2f, iters: %d, BS=%d, mu_2= %4.2f', d, 20000, alpha(a_Ren_pos), iters, bs, mu_2);
title(my_str)
%ylim([0.03, 0.08]);

xticks([5000, 10000, 20000, 50000, 100000, 150000])%([5000, 25000, 40000, 75000, 150000])
xticklabels({'5','10','20','50','100','150'})%({'5','25','40','75','150'})
xlabel('sample size N [\times 10^3]')

xlim([4000, 160000]);

ylabel('Renyi Divergence')
set(gca,'FontSize',12);





