%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this file, we used gradient descent method to solve the
% optimization problem mentioned in the report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% constants
featureNum = 4; % number of features
changeType = 4; % number of types of operations
sentenceNum = 5; % number of sentences
trainNum = 5; % number of sentences used for training
ita = 1e-1;  % convergence condition
alpha = 0.0008;  % step
maxGradient = 1; % used to control element of A

%%

% feature matrix with 1 or 0
F = randi(2, sentenceNum, featureNum) - 1;

% mapping matrix linear
A = rand(changeType, featureNum);

% label matrix
L = rand(sentenceNum, changeType);

% result of KL number
KLvec = zeros(1,trainNum);

% used for updating mapping matrix A
gradientMatrix = zeros(changeType, featureNum);

flag = 1;
n = 1;
while flag == 1
    for s = 1:sentenceNum
        f = A * F(s,:)';
%         for idx=1:changeType
%             sumValue = sum(f);
%             f = f/sumValue;
%         end
        tempA = zeros(changeType, featureNum);
        for i = 1:changeType
            for j = 1:featureNum
                
                if f(i) == 0
                    tempA(i,j) = A(i,j);
                    continue
                end
                gradientAij = F(s,j)*log(f(i)/L(s,i)) + F(s,j);
                gradientMatrix(i,j) = gradientAij;
                if abs(gradientAij) > maxGradient
                    maxGradient = abs(gradientAij);
                end
                tempA(i,j) = A(i,j)- alpha * gradientAij;
            end
        end
        A = tempA;
    end
    
    % afer updating A in one round
    totalKL = 0;
    for s1 = 1:sentenceNum
        KL = 0;
        f_result = A * F(s1,:)'
        for i1 = 1:changeType
            KL = KL + f_result(i1)*log(f_result(i1)/L(s1,i1));
        end
        totalKL = totalKL + KL;
    end
    totalKL
    KLvec(n) = totalKL;
    if n > 1
        % if kl < 0, we stop descending
        if totalKL <= 0
            flag = 0;
        end
        
        %  if kl in two rounds of iteration are quite similar, stop
        %  descending
        if abs(KLvec(n-1)-KLvec(n)) < 1e-6 
            flag = 0;
        end
    end
    n = n + 1;
end
x = 1 : 1 : size(KLvec,2);
plot(x, KLvec)
        
    

