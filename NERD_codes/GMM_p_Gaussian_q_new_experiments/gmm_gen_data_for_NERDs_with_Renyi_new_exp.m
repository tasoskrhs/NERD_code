%function gmm_example_Renyi
%
% 1-D examples
%
%   in this version: we compute log(p/q), p:GMM, q:Gaussian
%   ===============
%
clear; close all; clc;

rng(1);
d = 1; %dimension
dx = 0.01; %0.005; %
x = -20:dx:20; %-3:dx:3; % %-10:dx:10; % Support of distributions p, q
n = 5000; %150000; %50000; %20000; %100000; %10000; %25000; %75000; %40000; %4000; % %2500; %sample size





% if 0
% % 1st example: Gaussian vs Gaussian
% disp('1st example')
% mu1 = 0;
% mu2 = -3:0.1:3;
% sigma = 1;
% L = length(mu2);
% 
% alpha = 0.5;
% Ra = zeros(1, L);
% 
% for i = 1:L
%     p = normpdf(x, mu1, sigma); % p doesn't move.
%     q = normpdf(x, mu2(i), sigma);
%     
% %     plot(x,p);hold on;
% %     plot(x,q);hold off;
%     
%     Ra(i) = comp_Renyi(p, q, dx, alpha);
% end
% figure(11)
% plot(mu2, Ra);
% end
% 
% 
% 
% 
% if 0
% % 2nd example: Gaussian vs GMM  (Vary mass of the smaller mode in p AND alpha)
% disp('second example')
% mu0 = 0;
% mu1 = 0;
% %warning('CHANGED mu2!!')
% mu2 = 1; %0; %2;
% %warning('CHANGED sigma1!')
% sigma1 = 1; %2;
% sigma2 = 1;
% 
% % w = [0, 0.025, 0.05, 0.1, 0.2]  %0:0.01:0.2;  % percentage of samples in the smaller (Gaussian) mode
% 
% % non-uniform w values...
% dw = 0.1;
% for i=1:6
%     w(i) = dw
%     dw = 0.5*dw
% end
% 
% w(7) =0.0;
% 
% 
% alpha = [-10 -9 -8 -7 -6 -5 -4 -3 -2 -0.95 -0.7 -0.5 0.1 0.5 0.7 0.95 2 3 4 5 6 7 8 9 10 12]';%0.5; % [0.1 0.5 0.7 0.95]'; %(0:0.05:1)';
% %Ra = zeros(1, L);
% [X1, X2] = meshgrid(w, alpha);
% 
% [L1, L2] = size(X1);
% 
% Ra = zeros(L1, L2); % size: #alpha's x #w's
% x_sup = x; % support of x
% D_star = zeros(L2, size(x_sup,2));
% 
% for j = 1:L2 %i = 1:L  % as w increases, more samples (mass) are transfered to the smaller mode
%     
%     %p = (1-w(j))*normpdf(x_sup, mu1, sigma2) + w(j)*normpdf(x_sup, mu2, sigma2/16);
%     p = (1-w(j))*normpdf(x_sup, mu1, sigma2) + w(j)*normpdf(x_sup, mu2, sigma2/4);
%     q = normpdf(x_sup, mu0, sigma1);
%     
%     %find theoretical maximum distance D^{*}
%     D_star(j,:) = log(p./q);
%     
%     %MINE: generate n  1-D samples and save to file
%     q_x = repmat(mu0,n,1) + randn(n,d) *sigma1;
% %      figure(1);
% %      histogram(q_x,100,'FaceColor','r','Normalization','probability');
% %      xlim([-5,5]);
% %      title('Gaussian distribution q')
% %      ylim([0, 0.14]);
%     mode1 = repmat(mu1,n,1) + randn(n,d) *sigma2;
%     mode2 = repmat(mu2,n,1) + randn(n,d) *sigma2/4; %/16;
%     idx = randperm(n); %randomize the two distributions
%     last_sample = ceil(n*(1-w(j))); % pick (1-w)% of the samples for mode2
%     p_y = [mode1(idx(1:last_sample)); mode2(idx(last_sample+1:end))];
% %      figure(2);
% %      histogram(p_y,100,'Normalization','probability');
% %      xlim([-5,5]);
% %      title('GMM distribution p'); 
% %      ylim([0, 0.14]);
% 
%     x = p_y; y = q_x;
%     
%  %   save(['data/GMM_d_' num2str(d) '_data_' num2str(w(j)) '.mat'], 'x', 'y');
%     
%     for i = 1:L1 %j = 1:L2
%         Ra(i,j) = comp_Renyi(p, q, dx, alpha(i)); % the first argument is the mixture distribution while the second is the single-mode distribution as in Fig. 1 of Minka 2005.
%     end
%     
%     RD_exact = Ra(:,j); % NOTE!! it's not approximation of n samples, but not analytic (exact) either...
%  %   save(['data/GMM_d_' num2str(d) '_params_' num2str(w(j)) '.mat'], 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha',  'RD_exact');
%   
%     
% end
% 
% figure(21)
% % plot(w, Ra(1,:), '-*', 'LineWidth', 2); hold on;
% semilogy(w, Ra(1,:), '-*', 'LineWidth', 2); hold on;
% legend('a= -10', 'Location', 'northwest'); %('\alpha= -10'); % the latex interpreter works only in legends/titles/xlabels... NOT sprintf!
% cur_str = sprintf('a = %0.2f', alpha(6));
% semilogy(w, Ra(6,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);%plot(w, Ra(6,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('a = %0.2f', alpha(10));
% semilogy(w, Ra(10,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(w, Ra(10,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('a = %0.2f', alpha(12));
% semilogy(w, Ra(12,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(w, Ra(12,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('a = %0.2f', alpha(14));
% semilogy(w, Ra(14,:), '--', 'LineWidth', 2, 'DisplayName', cur_str);  %plot(w, Ra(14,:), '--', 'LineWidth', 2, 'DisplayName', cur_str); 
% cur_str = sprintf('a = %0.2f', alpha(16));
% semilogy(w, Ra(16,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(w, Ra(16,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('a = %0.2f', alpha(20));
% semilogy(w, Ra(20,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(w, Ra(20,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('a = %0.2f', alpha(25));
% semilogy(w, Ra(25,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(w, Ra(25,:), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('a = %0.2f', alpha(26));
% semilogy(w, Ra(26,:), '--*', 'LineWidth', 2, 'DisplayName', cur_str); 
% set(gca, 'FontSize', 12);
% xlabel('w')
% ylabel('Divergence')
% title('Exact Renyi Divergence, varying \alpha')
% xlim([0.001, 0.1]);
% 
% 
% 
% figure(22)
% semilogy(alpha, Ra(:,6), '-*', 'LineWidth', 2); hold on; %plot(alpha, Ra(:,7), '-*', 'LineWidth', 2); hold on;
% legend('w= 0.0031', 'Location', 'northwest');
% cur_str = sprintf('w = %0.5f', w(5));
% semilogy(alpha, Ra(:,5), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(alpha, Ra(:,6), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('w = %0.5f', w(4));
% semilogy(alpha, Ra(:,4), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(alpha, Ra(:,5), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('w = %0.5f', w(3));
% semilogy(alpha, Ra(:,3), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(alpha, Ra(:,4), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('w = %0.5f', w(2));
% semilogy(alpha, Ra(:,2), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(alpha, Ra(:,3), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% cur_str = sprintf('w = %0.5f', w(1));
% semilogy(alpha, Ra(:,1), '-*', 'LineWidth', 2, 'DisplayName', cur_str); %plot(alpha, Ra(:,2), '-*', 'LineWidth', 2, 'DisplayName', cur_str);
% set(gca, 'FontSize', 12);
% xlabel('\alpha')
% ylabel('Divergence')
% title('Exact Renyi Divergence, varying w')
% 
% 
% 
% figure(23); % choose 4 different mu_0 values and plot theoretical dist D^{*}
% plot(x_sup, D_star(1,:), 'LineWidth', 2); hold on;
% plot(x_sup, D_star(2,:), 'LineWidth', 2); 
% plot(x_sup, D_star(3,:), 'LineWidth', 2); 
% plot(x_sup, D_star(5,:), 'LineWidth', 2); 
% my_str1 = sprintf(' w_0= %0.2f', w(1));
% my_str2 = sprintf(' w_0= %0.2f', w(2));
% my_str3 = sprintf(' w_0= %0.2f', w(3));
% my_str4 = sprintf(' w_0= %0.2f', w(5));
% legend( my_str1, my_str2, my_str3, my_str4);
% xlabel('x');
% ylabel('D^{*}')
% hold off;
% 
% 
% end




if 1
% 2nd example: Gaussian vs GMM  (Vary mass of the smaller mode in p AND alpha)
disp('second example, one value for w')
mu0 = 0;
mu1 = 0;
mu2 = 1; %0; %2;
sigma1 = 1; %2;
sigma2 = 1;

w = 0.2; %0.05; %0.1; %[0, 0.025, 0.05, 0.1, 0.2]  %0:0.01:0.2;  % percentage of samples in the smaller (Gaussian) mode

%alpha = [-10 -9 -8 -7 -6 -5 -4 -3 -2 -0.95 -0.7 -0.5 0.1 0.5 0.7 0.95 2 3 4 5 6 7 8 9 10 12]';%0.5; % [0.1 0.5 0.7 0.95]'; %(0:0.05:1)';
%dalpha = 0.02;
%alpha = (-100: dalpha: 100)';%(-20: dalpha: 20)';
alpha = [0.5, 3.7]';

%Ra = zeros(1, L);
[X1, X2] = meshgrid(w, alpha);

[L1, L2] = size(X1);

Ra = zeros(L1, L2); % size: #alpha's x #w's
x_sup = x; % support of x
D_star = zeros(L2, size(x_sup,2));
%paq1ma = Ra;


for j = 1:L2 %i = 1:L  % as w increases, more samples (mass) are transfered to the smaller mode
    
    %p = (1-w(j))*normpdf(x_sup, mu1, sigma2) + w(j)*normpdf(x_sup, mu2, sigma2/16);
    p = (1-w(j))*normpdf(x_sup, mu1, sigma2) + w(j)*normpdf(x_sup, mu2, sigma2/4);
    q = normpdf(x_sup, mu0, sigma1);
    
    %find theoretical maximum distance D^{*}
    D_star(j,:) = log(p./q);
    
    %MINE: generate n  1-D samples and save to file
    q_x = repmat(mu0,n,1) + randn(n,d) *sigma1;
%      figure(1);
%      histogram(q_x,100,'FaceColor','r','Normalization','probability');
%      xlim([-5,5]);
%      title('Gaussian distribution q')
%      ylim([0, 0.14]);
    mode1 = repmat(mu1,n,1) + randn(n,d) *sigma2;
    mode2 = repmat(mu2,n,1) + randn(n,d) *sigma2/4; %/16;
    idx = randperm(n); %randomize the two distributions
    last_sample = ceil(n*(1-w(j))); % pick (1-w)% of the samples for mode2
    p_y = [mode1(idx(1:last_sample)); mode2(idx(last_sample+1:end))];
%      figure(2);
%      histogram(p_y,100,'Normalization','probability');
%      xlim([-5,5]);
%      title('GMM distribution p'); 
%      ylim([0, 0.14]);


    x = p_y; y = q_x;
    
    save(['data/GMM_d_' num2str(d) '_data_' num2str(w(j)) '.mat'], 'x', 'y');
    %warning('I reversed p <-> q!')
    for i = 1:L1 %j = 1:L2
        Ra(i,j) = comp_Renyi(p, q, dx, alpha(i)); % the first argument is the mixture distribution while the second is the single-mode distribution as in Fig. 1 of Minka 2005.
        %Ra(i,j) = comp_Renyi(q, p, dx, alpha(i));
        %paq1ma(i,:) = p.^alpha(i) .* q.^(1-alpha(i));

    end
    
    RD_exact = Ra(:,j); % NOTE!! it's not approximation of n samples, but not analytic (exact) either...
    save(['data/GMM_d_' num2str(d) '_params_' num2str(w(j)) '.mat'], 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha',  'RD_exact');
  
    
end

% figure(24)
% plot(alpha, Ra(:,1) ) % 1 because we only use one w only...
% xlabel('\alpha')
% ylabel('Divergence')
% my_str = sprintf('dim: %d, w= %0.3f, delta alpha= %0.4f', d, w, dalpha);
% title(my_str)
% 
% figure(25)
% plot(x_sup, D_star)
% xlabel('\alpha')
% ylabel('D_star')
% my_str = sprintf('dim: %d, w= %0.3f, delta alpha= %0.4f', d, w, dalpha);
% title(my_str)
% 
% 
% figure(26)
% id_a = 1 ;
% plot(x_sup, paq1ma(id_a,:) ) % first argument of paq1ma defines alpha value
% xlabel('x')
% ylabel('p^a * q^{1-a}')
% my_str = sprintf('w= %0.3f, alpha= %0.4f', w, alpha(id_a));
% title(my_str)


end








if 0
% 3rd example: Gaussian vs GMM (main mode of p, NOT centered with q) 
% VARY Renyi parameter alpha 
disp('3rd example')
mu0 = 0;
mu1 = 1;
mu2 = 2;
sigma1 = 1;
sigma2 = 1;

w = 0.1;

alpha = -50.05:.1:50.05;
L2 = length(alpha);

Ra = zeros(1, L2);

for i = 1:L2
    p = (1-w)*normpdf(x, mu1, sigma2) + w*normpdf(x, mu2, sigma2/4);
    q = normpdf(x, mu0, sigma1);
    
%     plot(x,p);hold on;
%     plot(x,q);hold off;
%     pause(.3);
    
    Ra(i) = comp_Renyi(p, q, dx, alpha(i));
end

figure(31)
plot(alpha, Ra);
xlabel('\alpha parameter')  % NOTE: inf or NAN for |alpha| > 20!!
end





if 0
% 4th example: Gaussian vs GMM (moving SMALL mode mean mu2 (GMM), varying alpha)
disp('4th example, moving mode!')
mu0 = 0; %[0, 1]; %-3:1:3; %-3:0.1:3;
mu1 = 0;
mu2 = [0, 1, 2]; %2;
sigma1 = 1;  %2;
sigma2 = 1;

w = 0.2; %0.1; %0.1;

alpha = 0.95; %0.5; %[0.1 0.5 0.7 0.95]'; % (0:0.05:1)';  

[X1, X2] = meshgrid(mu2, alpha); %meshgrid(mu0, alpha);

[L1, L2] = size(X1);

Ra = zeros(L1, L2);
x_sup = x; % support of x
D_star = zeros(L2, size(x_sup,2));  

for j = 1:L2 %i = 1:L1
    p = (1-w)*normpdf(x_sup, mu1, sigma2) + w*normpdf(x_sup, mu2(j), sigma2/4);
    q = normpdf(x_sup, mu0, sigma1);
    
    %     plot(x,p);hold on;
    %     plot(x,q);hold off;
    %     pause(.3);
    
    %find theoretical maximum distance D^{*}
    D_star(j,:) = log(p./q);
    
    %MINE: generate n  1-D samples and save to file
    q_x = repmat(mu0,n,1) + randn(n,d) *sigma1;
%     figure(1);
%     histogram(q_x,100,'FaceColor','r','Normalization','probability');
%     xlim([-5,5]);
%     title('Gaussian distribution q')
    mode1 = repmat(mu1,n,1) + randn(n,d) *sigma2;
    mode2 = repmat(mu2(j),n,1) + randn(n,d) *sigma2/4;
    idx = randperm(n); %randomize the two distributions
    last_sample = ceil(n*(1-w)); % pick (1-w)% of the samples for mode2
    p_y = [mode1(idx(1:last_sample)); mode2(idx(last_sample+1:end))];
%     figure(2);
%     histogram(p_y,100,'Normalization','probability');
%     xlim([-5,5]);
%     title('GMM distribution p')
    x = p_y; y = q_x;
    
    save(['data/GMM_d_' num2str(d) '_data_' num2str(mu2(j)) '.mat'], 'x', 'y');
    
    for i = 1:L1 %j = 1:L2       
        
%         Ra(i,j) = comp_Renyi(q, p, dx, alpha(i));       
        Ra(i,j) = comp_Renyi(p, q, dx, alpha(i));
    end
    
    RD_exact = Ra(:,j); % NOTE!! it's not approximation of n samples, but not analytic (exact) either...
    save(['data/GMM_d_' num2str(d) '_params_' num2str(mu2(j)) '.mat'], 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha',  'RD_exact');
    
    
end

% figure(41)
% mesh(X1, X2, Ra);

% figure(42)
% contour(X1, X2, log(4+log(Ra)));  % Why 4?--> positive values for log (this example)
% title('log(4+log(Ra))')
% xlabel('\mu_0 of distribution q')
% ylabel('\alpha parameter')

%  for i = 1:L1
%      plot(mu0, Ra(i,:));hold on;
%      pause(0.2);
%  end
%  hold off;

% figure(43); % choose 4 different mu_0 values and plot theoretical dist D^{*}
% plot(x_sup, D_star(1,:), 'LineWidth', 2); hold on;
% plot(x_sup, D_star(31,:), 'LineWidth', 2); 
% plot(x_sup, D_star(41,:), 'LineWidth', 2); 
% plot(x_sup, D_star(61,:), 'LineWidth', 2); 
% my_str1 = sprintf(' mu_0= %0.1f', mu0(1));
% my_str31 = sprintf(' mu_0= %0.1f', mu0(31));
% my_str41 = sprintf(' mu_0= %0.1f', mu0(41));
% my_str61 = sprintf(' mu_0= %0.1f', mu0(61));
% legend( my_str1, my_str31, my_str41, my_str61);
% xlabel('support of p');
% ylabel('D^{*}')
% hold off;


end




if 0
% 5th example: Gaussian vs GMM (stretching q (Gaussian), varying alpha)
mu0 = 2; % 0 or 1 or 2
mu1 = 0;
mu2 = 2;
sigma1 = 0.2:0.1:5;
sigma2 = 1;

w = 0.1;

alpha =  (0:0.05:1)'; %-10.05:.1:10.05; 

[X1, X2] = meshgrid(sigma1, alpha);

[L1, L2] = size(X1);

Ra = zeros(L1, L2);
x_sup = x; % support of x
D_star = zeros(L2, size(x_sup,2));  

for j = 1:L2 %i = 1:L1
    p = (1-w)*normpdf(x_sup, mu1, sigma2) + w*normpdf(x_sup, mu2, sigma2/4);
    q = normpdf(x_sup, mu0, sigma1(j));
    
    %  plot(x,p);hold on;
    %  plot(x,q);hold off;
    %  pause;
    
    %find theoretical maximum distance D^{*}
    D_star(j,:) = log(p./q);
    
    %MINE: generate n  1-D samples and save to file
    q_x = repmat(mu0,n,1) + randn(n,d) *sigma1(j);
%     figure(1);
%     histogram(q_x,100,'FaceColor','r');
%     title('Gaussian distribution q')
%     xlim([-5,5]);
    mode1 = repmat(mu1,n,1) + randn(n,d) *sigma2;
    mode2 = repmat(mu2,n,1) + randn(n,d) *sigma2/4;
    idx = randperm(n); %randomize the two distributions
    last_sample = ceil(n*(1-w)); % pick (1-w)% of the samples for mode2
    p_y = [mode1(idx(1:last_sample)); mode2(idx(last_sample+1:end))];
%     figure(2);
%     histogram(p_y,100);
%     xlim([-5,5]);
%     title('GMM distribution p')
    x = p_y; y = q_x;
    
  %  save(['data/GMM_d_' num2str(d) '_data_' num2str(sigma1(j)) '.mat'], 'x', 'y');
    
    for i = 1:L1 % j = 1:L2
        
        Ra(i,j) = comp_Renyi(p, q, dx, alpha(i));
    end
    
    RD_exact = Ra(:,j); % NOTE!! it's not approximation of n samples, but not analytic (exact) either...
  %  save(['data/GMM_d_' num2str(d) '_params_' num2str(sigma1(j)) '.mat'], 'mu1', 'mu2', 'sigma1', 'sigma2', 'alpha',  'RD_exact');
    
end

% mesh(X1, X2, Ra);
figure(3)
for i = 1:L1
    plot(sigma1, Ra(i,:));hold on;
    %ylim([0,1]);
    pause(0.2);
end
%xlim([0,1]);
xlabel('\sigma_1 of distribution q')
ylabel('\alpha parameter')
hold off;

figure(53); % choose 4 different sigma1 values and plot theoretical dist D^{*}
plot(x_sup, D_star(1,:), 'LineWidth', 2); hold on;
plot(x_sup, D_star(21,:), 'LineWidth', 2); 
plot(x_sup, D_star(31,:), 'LineWidth', 2); 
plot(x_sup, D_star(49,:), 'LineWidth', 2); 
my_str1 = sprintf(' sigma1= %0.1f', sigma1(1));
my_str2 = sprintf(' sigma1= %0.1f', sigma1(21));
my_str3 = sprintf(' sigma1= %0.1f', sigma1(31));
my_str4 = sprintf(' sigma1= %0.1f', sigma1(49));
legend( my_str1, my_str2, my_str3, my_str4);
xlabel('support of p');
ylabel('D^{*}')
hold off;



end











if 0
% 6th example: Gaussian vs GMM (moving AND stretcing q (Gaussian), FIX alpha)
mu0 = -1:0.1:3;
mu1 = 0;
mu2 = 2;
sigma1 = 0.2:0.1:5;
sigma2 = 1;

w = 0.1;

alpha = 10; % -10 or 0.5 or 10 

[X1, X2] = meshgrid(mu0, sigma1);

[L1, L2] = size(X1);

Ra = zeros(L1, L2);

for i = 1:L1
    for j = 1:L2
        p = (1-w)*normpdf(x, mu1, sigma2) + w*normpdf(x, mu2, sigma2/4);
        q = normpdf(x, mu0(j), sigma1(i));

%         plot(x,p);hold on;
%         plot(x,q);hold off;
%         pause;

        %Ra(i,j) = comp_Renyi(q, p, dx, alpha);
        Ra(i,j) = comp_Renyi(p, q, dx, alpha);
    end
end

% mesh(X1, X2, log(Ra));
figure(1)
contour(X1, X2, log(Ra));
% for i = 1:L1
%     plot(sigma1, Ra(i,:));hold on;
%     ylim([0,1]);
%     pause(0.2);
% end
% hold off;
end


