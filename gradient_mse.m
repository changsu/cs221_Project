%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this file, we used gradient descent method to solve the
% optimization problem mentioned in the report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% read and pre-process feature and label files
load feature_matrix.txt
load label_independent.txt

%% declare global variables used in query.m
global sentenceMap F L A;

% chunk feature matrix to be consisitent with label matrix
F = feature_matrix(1:size(label_independent, 1), :);
L = label_independent;

% build sentence number map
fid = fopen('sentence_map.txt');
sentenceMap = [];
for i = 1 : size(F,1)
    line = fgetl(fid);
    [sentenceNum totalScript] = strread(line, '%s %d', 'delimiter', ' \t');
    sentenceMap = [sentenceMap; sentenceNum totalScript];
end
fclose(fid);

% remove all rows where feature/label sum up to 0
zeroRowsF = find(all(F==0,2));
zeroRowsL = find(all(L==0,2));
zeroRows = [zeroRowsF;zeroRowsL];
F(zeroRows, :) = [];
L(zeroRows, :) = [];
sentenceMap(zeroRows, :) = [];

% approximate 0.0 in order KL can work
L(find(L == 0)) = 0.0001;

% shuffle the rows to fully test the method
FL = [F L];
randIndex = randperm(size(FL, 1));
FL = FL(randIndex, :);
F = FL(:, 1:size(F,2));
seperator = size(F,2) + 1;
L = FL(:, seperator : seperator + size(L,2) - 1);
sentenceMap = sentenceMap(randIndex, :);

%% declare constants
featureNum = size(F, 2); % number of features
changeType = size(L, 2); % number of types of operations
sentenceNum = size(F, 1); % number of sentences
trainNum = round(sentenceNum * 0.7); % number of sentences for training
testNum = sentenceNum - trainNum;
ita = 1e-4;  % convergence condition
alpha = 8e-5;  % stepsize
alpha_fine = 8e-7; % finer stepsize
numRand = 5; % number of runs to compute random result 

%% apply gradient descend solving the optimization problem

% initialize mapping matrix randomly
A = rand(changeType, featureNum);

% store result of KL summation
MSEvecTrain = zeros(1,trainNum);
MSEvecTest = zeros(1, testNum);

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
                % it is possible that L(s,i) == 0
                gradientAij = 2 * (f(i) - L(s,i)) * F(s,j);
                gradientMatrix(i,j) = gradientAij;
                tempA(i,j) = A(i,j)- alpha * gradientAij;
                
                % if Aij is negative after updates, it is possible f(i)<0
                % in next iteration, to avoid this, we do not change Aij
                % it should be noted that even though we do not change Aij,
                % converge rate will be decreased, however, the loss
                % function still decreases and we can finally get the
                % optimal solution
                if (tempA(i,j) < 0)
                    tempA(i,j) = A(i,j);
                end
            end
        end
        A = tempA;
    end
    
    % after updating A in one round, calculate total KL in testing data
    totalTestMSE = 0;
    for s1 = trainNum + 1 : sentenceNum
        f_result = A * F(s1,:)';
        MSE = norm((f_result - L(s1,:)'),2);
        totalTestMSE = totalTestMSE + MSE;
    end
    MSEvecTest(n) = totalTestMSE/(sentenceNum - trainNum);
    
    % afer updating A in one round, calculate total KL in training data
    totalTrainMSE = 0;
    for s1 = 1 : trainNum
        f_result = A * F(s1,:)';
        MSE = norm((f_result - L(s1,:)'),2);
        totalTrainMSE = totalTrainMSE + MSE;
    end
    totalTrainMSE = totalTrainMSE / (trainNum)
    MSEvecTrain(n) = totalTrainMSE;
    
    if n > 1
        %  if kl in two rounds of iteration are quite similar, stop
        %  descending
        if abs(MSEvecTrain(n-1)-MSEvecTrain(n)) < ita
            flag = 0;
        end
    end
    
    % change step size when approaching the optimal
%     if totalTrainMSE < 5
%         alpha = alpha_fine;
%     end
    
    n = n + 1;   
end

%% Plot result
x = 1 : 1 : size(MSEvecTrain,2);
y = 1 : 1 : size(MSEvecTest,2);
plot(x, MSEvecTrain, 'b');
hold on;
plot(y, MSEvecTest, 'r');

%%%%%%%%%% Print out final result %%%%%%%%%%
ourResult = min(MSEvecTest);
str = ['KL on testingn data using algorithm: ', num2str(ourResult)];
display(str);

%% Evaluate random guess
for k = 1: numRand
    randMSE = 0;
    L_rand = rand(sentenceNum, changeType);
    for n = trainNum + 1 : sentenceNum
        MSE = norm((L_rand(n,:) - L(n,:)),2);
        randMSE = randMSE + MSE;
    end
    totalRandMSE(k) = randMSE / (sentenceNum - trainNum);
end
str = ['KL on testingn data using random gusess: ', ...
    num2str(mean(totalRandMSE))];
display(str);


    

