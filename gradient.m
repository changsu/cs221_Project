%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this file, we used gradient descent method to solve the
% optimization problem mentioned in the report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% constants
featureNum = 33; % number of features
changeType = 7; % number of types of operations
sentenceNum = 100; % number of sentences
trainNum = 70; % number of sentences used for training
testNum = sentenceNum - trainNum;
ita = 1e-1;  % convergence condition
alpha = 0.000008;  % stepsize
maxGradient = 1; % used to control element of A

%%

% feature matrix with 1 or 0
F = randi(2, sentenceNum, featureNum) - 1;

% mapping matrix linear
A = rand(changeType, featureNum);

% label matrix
L = rand(sentenceNum, changeType);

% result of KL number
KLvecTrain = zeros(1,trainNum);
KLvecTest = zeros(1, testNum);

% used for updating mapping matrix A
gradientMatrix = zeros(changeType, featureNum);

flag = 1;
n = 1;
while flag == 1
    for s = 1:trainNum
        f = A * F(s,:)';
        tempA = zeros(changeType, featureNum);
        for i = 1:changeType
            for j = 1:featureNum
                
                if f(i) <= 0
                    tempA(i,j) = A(i,j);
                    continue
                end
                gradientAij = F(s,j)*log(f(i)/L(s,i)) + F(s,j);
                gradientMatrix(i,j) = gradientAij;
                tempA(i,j) = A(i,j)- alpha * gradientAij;
                if (tempA(i,j) < 0)
                    tempA(i,j) = A(i,j);
                end
            end
        end
        A = tempA;
    end
    
    % after updating A in one round, calculate total KL in testing data
    totalTestKL = 0;
    for s1 = trainNum + 1 : sentenceNum
        KL = 0;
        f_result = A * F(s1,:)';
        for i1 = 1:changeType
            KL = KL + f_result(i1)*log(f_result(i1)/L(s1,i1));
        end
        totalTestKL = totalTestKL + KL;
    end
    KLvecTest(n) = totalTestKL;
    
    % afer updating A in one round, calculate total KL in training data
    totalTrainKL = 0;
    for s1 = 1 : trainNum
        KL = 0;
        f_result = A * F(s1,:)';
        for i1 = 1:changeType
            KL = KL + f_result(i1)*log(f_result(i1)/L(s1,i1));
        end
        totalTrainKL = totalTrainKL + KL;
    end
    totalTrainKL
    
    KLvecTrain(n) = totalTrainKL;
    
    if n > 1
        % if kl < 0, we stop descending
        if totalTrainKL <= 0
            flag = 0;
        end
        
        %  if kl in two rounds of iteration are quite similar, stop
        %  descending
        if abs(KLvecTrain(n-1)-KLvecTrain(n)) < 1e-6 
            flag = 0;
        end
    end
    n = n + 1;
    
    
end
x = 1 : 1 : size(KLvecTrain,2);
y = 1 : 1 : size(KLvecTest,2);
plot(x, KLvecTrain, 'b');
hold on;
plot(y, KLvecTest, 'r');

%%%%%%%%%% Print out final result %%%%%%%%%%
ourResult = KLvecTest(1, end);
str = ['KL on testingn data using algorithm: ', num2str(ourResult)];
display(str);

%% Evaluate random guess
randKL = 0;
L_rand = rand(sentenceNum, changeType);
for n = trainNum + 1 : sentenceNum
    KL = 0;
    for x = 1:changeType
        KL = KL + L_rand(n,x)*log(L_rand(n,x)/L(n,x));
    end
    randKL = randKL + KL;
end
str = ['KL on testingn data using random gusess: ', num2str(randKL)];
display(str);


    

