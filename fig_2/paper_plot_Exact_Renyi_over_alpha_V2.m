% Paper figure 1
%
% Define the Gaussians, compute the exact Renyi divergence over a for two:
%  a. 1-D Gaussians centered at 0, with perturbed variance. (analytic estimation) 
%
%     Identify for which a,  R_a(N(0,sigma^2(1+eps)^2) || N(0,sigma^2)) quickly increases
%
%  b. 2-D Gaussians centered at [0,0], with perturbed covariance matrix (analytic estimation)
%
%
%  c. 1-D Gaussuan mixture model with small mode parameter w AND Gaussian (statistical estimation)
% 
%  d. same as c. for different choice of w's
%  


clear; close all; clc;

d = 1;   
% mean values
mu1 = zeros(d,1);  
mu2 = zeros(d,1); 

Sigma2 = eye(d) ;  
Sigma2_det = det(Sigma2);
Sigma2_inv = inv(Sigma2);

%  a. 1-D Gaussians R(Q||P)
%==========================
% perturb the variance value of \sigma_1 (of the first Gaussian) 
% Q = N(0, sigma^2(1+eps))
% P = N(0, sigma^2)
% here, we set sigma = 1
%===========================================================================
eps = [0.1, -0.2] ; % perturbation
Legend=cell(length(eps),1);

for iter = 1: length(eps)
    
     Sigma1_new = eye(d)+ eps(iter);
%      Sigma1_new = eye(d);
%      Sigma1_new(1:d/2, d/2 +1:end) = rho(p)*eye(d/2); %triu(rho(p)*ones(d,d), +1); % fill upper triangular part
%      Sigma1_new(d/2 +1:end, 1:d/2) = rho(p)*eye(d/2); %triu(rho(p)*ones(d,d), +1)'; % fill lower triangular part
    Sigma1_det = det(Sigma1_new);
    Sigma1_inv = inv(Sigma1_new);
    
    L1 = chol(Sigma1_new); % \Sigma^{1/2}. If not pos-def, there's error!
    
    % Renyi estimation
     dalpha = 0.0001; % discretization length
     alpha = [-12: dalpha :12.0]';
    
    No_alpha = length(alpha);
    
    idx = false(No_alpha, 1);
    RD_exact = zeros(No_alpha, 1);
    
    for i = 1:No_alpha
        Sigma_inv_alpha = alpha(i)*Sigma1_inv + (1-alpha(i))*Sigma2_inv;
        lambda_inv_alpha = eig(Sigma_inv_alpha);
        
        if all(lambda_inv_alpha > 0)
            idx(i) = true;
            
            % exact (analytical formula estimation)
            Sigma_alpha = (1-alpha(i))*Sigma1_new + alpha(i)*Sigma2;
            
            if alpha(i)==0
                RD_exact(i) = 0.5 * ((mu1-mu2)'*Sigma1_inv*(mu1-mu2) + trace(Sigma1_inv*Sigma2) + log(Sigma1_det/Sigma2_det) - d);
            elseif alpha(i)==1
                RD_exact(i) = 0.5 * ((mu1-mu2)'*Sigma2_inv*(mu1-mu2) + trace(Sigma2_inv*Sigma1_new) + log(Sigma2_det/Sigma1_det) - d);
            else
                RD_exact(i) = 0.5 * (mu1-mu2)' * inv(Sigma_alpha) * (mu1-mu2) ...
                    + 1/(2*alpha(i)*(1-alpha(i))) * log(det(Sigma_alpha) / (Sigma1_det^(1-alpha(i)) * Sigma2_det^alpha(i)));
            end
       
        end
    end
    
    
    %plot R_a over a  
    %-----------------
    figure(1);  
    subplot(2,2,1)
    
    %hardcode vector limits, in order to avoid plotting NaN and zeros
    if iter == 1
        %plot(alpha(1:180000), RD_exact(1:180000), '-', 'LineWidth',2); hold on;
        plot(alpha(1:230000), RD_exact(1:230000), '-', 'LineWidth',2); hold on;
        Legend{iter}= '\epsilon = 0.1';%hardcoded because my_str string cannot be interpreted with latex
    elseif iter == 2
        %plot(alpha(30002:end), RD_exact(30002:end), '-', 'LineWidth',2);
        plot(alpha(80002:end), RD_exact(80002:end), '-', 'LineWidth',2);
        Legend{iter}= '\epsilon = -0.2';    
    end
    
    
    ylim([0, 0.04]);%([0, 0.05]); 
    xlim([-12, 12]);
    xticks([-11 -4 0 11 ])
    xticklabels({'-11','-4','0','11'})
    %axis tight %square %
    set(gca,'FontSize',16);  
    
    if iter == 2
        xlabel('$$\alpha$$','interpreter','latex', 'FontWeight','bold')
        ylabel('$$\mathcal{R}_{\alpha}(Q || P)$$','interpreter','latex', 'FontWeight','bold')
        legend(Legend,'Location','north')
        text(-11,0.035,'(a)','FontSize',16, 'FontWeight','bold')
        %axis square
    end
    
end



% b. 2-D Gaussians R(Q||P)
%=========================
% Q = N([0,0], \Sigma_1),                 |1    \rho |             |1  0|
% P = N([0,0], \Sigma_2)        Sigma_1 =                Sigma_2 =
%==============                           |\rho   1  |             |0  1|
d = 2;  

% mean values
mu1 = zeros(d,1);   
mu2 = zeros(d,1); 
Sigma2 = eye(d) ;
Sigma2_det = det(Sigma2);
Sigma2_inv = inv(Sigma2);

% change the value of \rho (of the first multivariate Gaussian)
%===========================================================================
rho = [0.1 0.2]; %[-0.9: 0.2: 0.9];
for p = 1: length(rho)
    
     Sigma1_new = eye(d);
     Sigma1_new(1:d/2, d/2 +1:end) = rho(p)*eye(d/2); %triu(rho(p)*ones(d,d), +1); % fill upper triangular part
     Sigma1_new(d/2 +1:end, 1:d/2) = rho(p)*eye(d/2); %triu(rho(p)*ones(d,d), +1)'; % fill lower triangular part
    Sigma1_det = det(Sigma1_new);
    Sigma1_inv = inv(Sigma1_new);
    
    L1 = chol(Sigma1_new); % \Sigma^{1/2}. If not pos-def, there's error!

    % Renyi estimation
    dalpha = 0.001;%0.1 %1 %0.01;%0.05 %0.01;
    alpha = (-20: dalpha:20)';   %  0.5; %[0.1, 0.5, 0.7, 0.95]'; %(0:0.05:1)';
    No_alpha = length(alpha);
    
    idx = false(No_alpha, 1);
    RD_exact = zeros(No_alpha, 1);
    
    for i = 1:No_alpha
        %     Sigma_inv_alpha = (1-alpha(i))*Sigma1 + alpha(i)*Sigma2;
        Sigma_inv_alpha = alpha(i)*Sigma1_inv + (1-alpha(i))*Sigma2_inv;
        lambda_inv_alpha = eig(Sigma_inv_alpha);
        
        if all(lambda_inv_alpha > 0)
            idx(i) = true;
            
            % exact (analytical formula estimation)
            Sigma_alpha = (1-alpha(i))*Sigma1_new + alpha(i)*Sigma2;
            
            if alpha(i)==0
                RD_exact(i) = 0.5 * ((mu1-mu2)'*Sigma1_inv*(mu1-mu2) + trace(Sigma1_inv*Sigma2) + log(Sigma1_det/Sigma2_det) - d);
            elseif alpha(i)==1
                RD_exact(i) = 0.5 * ((mu1-mu2)'*Sigma2_inv*(mu1-mu2) + trace(Sigma2_inv*Sigma1_new) + log(Sigma2_det/Sigma1_det) - d);
            else
                RD_exact(i) = 0.5 * (mu1-mu2)' * inv(Sigma_alpha) * (mu1-mu2) ...
                    + 1/(2*alpha(i)*(1-alpha(i))) * log(det(Sigma_alpha) / (Sigma1_det^(1-alpha(i)) * Sigma2_det^alpha(i)));
            end
            
            
        end
    end
    
    figure(1)
    subplot(2,2,2)
    if p == 1
        plot(alpha(11002:31002), RD_exact(11002:31002), '-', 'LineWidth',2); hold on;
        Legend{p}= '\rho = 0.1';%hardcoded because my_str string cannot be interpreted with latex
    elseif p == 2
        plot(alpha(16002:26000), RD_exact(16002:26000), '-', 'LineWidth',2);
        Legend{p}= '\rho = 0.2';
    end
 
    
    if p == 2
        xticks([-9 -4 0 6 11])
        xticklabels({'-9','-4','0','6','11'})
        xlim([-12, 12]);
        ylim([0, 0.04]);
        xlabel('$$\alpha$$','interpreter','latex', 'FontWeight','bold')
        %ylabel('R_{\alpha}(Q || P)')
        legend(Legend,'Location','north')
        set(gca,'FontSize',16);  
        text(-11,0.035,'(b)','FontSize',16, 'FontWeight','bold')
        %axis square
    end
    
end


%error('checkpoint')


% c. and d.  1-D GMM and 1-D Gaussian R(p||q) (RD using statistical estimation)
%===============================================================================
d = 1; %dimension
dx = 0.005; %0.01;
x_sup = -20:dx:20; %-10:dx:10; % Support of distributions p, q

% Gaussian vs GMM  (Vary mass of the smaller mode in p AND alpha)
% parameters of GMM distribution p
mu0 = 0;
sigma1 = 1; %2;
% parameters of Gaussian distribution q
mu1 = 0;
%warning('changed mu2!!')
mu2 = 2; %1; 
sigma2 = 1;

w = [0.01, 0.02, 0.1, 0.2];  % percentage of samples in the smaller (Gaussian) mode

dalpha = 0.02;
alpha = (-100: dalpha: 100)';

[X1, X2] = meshgrid(w, alpha);

[L1, L2] = size(X1);

Ra = zeros(L1, L2); % size: #alpha's x #w's


for j = 1:L2  % as w increases, more samples (mass) are transfered to the smaller mode
    
    p = (1-w(j))*normpdf(x_sup, mu1, sigma2) + w(j)*normpdf(x_sup, mu2, sigma2/4); %GMM
    q = normpdf(x_sup, mu0, sigma1); % Gaussian
    
    for i = 1:L1 %j = 1:L2
        Ra(i,j) = comp_Renyi_GMM(p, q, dx, alpha(i)); % the first argument is the mixture distribution while the second is the single-mode distribution as in Fig. 1 of Minka 2005.
    end
    
end

% There are NaN values at alpha = {0,1}, Instead of using reverse D_{KL},
% simply interpolate values
% [row, col] = find(isnan(Ra));
[ri,ci] = find(~isfinite(Ra))
for k =1: length(ri)
    
   Ra(ri(k),ci(k)) = 0.5*( Ra(ri(k)-1,ci(k)) + Ra(ri(k)+1,ci(k)) ) ;
   
end

figure(1)
subplot(2,2,3)
plot(alpha, Ra(:,3),'Linewidth', 2 ); hold on;
%legend('w=0.1') %handle first manually to crete the legend once...
Legend{1}= '$w = 0.1$';%hardcoded because my_str string cannot be interpreted with latex
my_str= sprintf('w=%0.1f',w(4));
plot(alpha, Ra(:,4),'Linewidth', 2); %, 'DisplayName', my_str )
Legend{2}= '$w = 0.2$';%hardcoded because my_str string cannot be interpreted with latex
%text(-19, 0.07,'(c)') %for mu2= 1
text(-4, 0.33,'(c)','FontSize',16, 'FontWeight','bold')
xlabel('$$\alpha$$','interpreter','latex', 'FontWeight','bold')
ylabel('$$\mathcal{R}_{\alpha}(Q || P)$$','interpreter','latex', 'FontWeight','bold')
%ylabel('R_{\alpha}(Q || P)')
%xlim([-20, 20]); %for mu2= 1
xlim([-5, 14]); %for mu2= 2
%ylim([0, 0.08]); %for mu2= 1
ylim([0, 0.38]); %for mu2= 2
%xticks([-20 0 3.6 7.2 20]) %for mu2= 1
%xticklabels({'-20', '0', '3.6','7.2','20'})
xticks([-5 0 1 2.9 4.7 14]) %for mu2= 2
xticklabels({'-5', '0', '1','2.9', '4.7','14'})
legend(Legend,'Location','northeast','interpreter','latex')
line([2.9 2.9], [0 0.35],'Color','red','LineStyle','--','HandleVisibility','off'); % solid line indicating maximum
line([4.66 4.66], [0 0.152],'Color','red','LineStyle','--','HandleVisibility','off'); % solid line indicating maximum
set(gca,'FontSize',16); 

%axis square

figure(1)
subplot(2,2,4)
plot(alpha, Ra(:,1),'Linewidth', 2); hold on;
Legend{1}= '$w = 0.01$';%legend('$$w=0.01$$','interpreter','latex') %handle first manually to crete the legend once...
my_str= sprintf('w=%0.2f',w(2))
plot(alpha, Ra(:,2),'Linewidth', 2);%, 'DisplayName', my_str );
Legend{2}= '$w = 0.02$';
my_str= sprintf('w=%0.2f',w(3));
xlabel('$$\alpha$$','interpreter','latex', 'FontWeight','bold')
% text(-90,0.0011,'(d)')
text(-15,0.015,'(d)','FontSize',16, 'FontWeight','bold')
xlim([-20, 60]);%([-60, 60]);
% xticks([-100 0 35 70 100]) %for mu2= 1
% xticklabels({'-100', '0', '35','70','100'})
xticks([-20 0 15.8 29.1 60]) %for mu2= 2
xticklabels({'-20', '0', '15.8','29.1','60'})
legend(Legend,'Location','northeast','interpreter','latex')
line([15.84 15.84], [0 0.015],'Color','red','LineStyle','--','HandleVisibility','off'); % solid line indicating maximum
line([29.08 29.08], [0 0.0045],'Color','red','LineStyle','--','HandleVisibility','off'); % solid line indicating maximum
set(gca,'FontSize',16); 
ylim([0, 0.018]);


%axis square


% use export_fig
% cd ~/Documents/GSRT_NeoiErevnites/export_fig-master
% code_dir = pwd;
% addpath(genpath(code_dir));

%use an external library for saving image files (cropped, high resolution)
set(gcf, 'Color', 'w'); %background color set to white
export_fig out3.png -m6 %magnifies resolution 6 times...
