%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this file, we used gradient descent method to solve the
% optimization problem mentioned in the report
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% read and pre-process feature and label files
load feature_matrix.txt
load label_dependent.txt

%% declare global variables used in query.m
global sentenceMap F L A method;

% set method
method = 'kl';

% chunk feature matrix to be consisitent with label matrix
F = feature_matrix(1:size(label_dependent, 1), :);
L = label_dependent;

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
ita = 5e-5;  % convergence condition
alpha = 8e-3;  % stepsize
alpha_fine = 8e-7; % finer stepsize
numRand = 5; % number of runs to compute random result

%% apply gradient descend solving the optimization problem

% initialize mapping matrix randomly
A = rand(changeType, featureNum);

% store result of KL summation
KLvecTrain = zeros(1,trainNum);
KLvecTest = zeros(1, testNum);

% used for updating mapping matrix A
gradientMatrix = zeros(changeType, featureNum);

flag = 1;
n = 1;
while flag == 1
    for s = 1:trainNum
        f = A * F(s,:)';
        numerator = sum(f);
        tempA = zeros(changeType, featureNum);
        for i = 1:changeType
            for j = 1:featureNum
                % if f(i) <= 0 or L(s,i) == 0, 
                % log (f(i)) will become complex number
                % we do not change Aij in this case
                if f(i) <= 0 || L(s,i) == 0
                    tempA(i,j) = A(i,j);
                    continue
                end
                % it is possible that L(s,i) == 0
%                 gradientAij = F(s,j)*log(f(i)/L(s,i)) + F(s,j);
                selfItem = (F(s, j) * numerator - f(i) * F(s,j)) / ...
                    (numerator ^ 2) * (log(f(i) / numerator / L(s,i)) + 1); 
                otherItem = 0;
                for k = 1:changeType 
                    if k == i
                        continue
                    end
                    otherItem = otherItem + (0-f(k)*F(s,j))/ ...
                        (numerator ^ 2) * (log(f(k)/numerator/L(s,k)) + 1);
                        
                end
                
                gradientAij = selfItem + otherItem;
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
    totalTestKL = 0;
    for s1 = trainNum + 1 : sentenceNum
        KL = 0;
        f_result = A * F(s1,:)';
        numerator = sum(f_result);
        for i1 = 1:changeType
            KL = KL + f_result(i1)/numerator * ...
                log(f_result(i1)/numerator/L(s1,i1));
        end
        totalTestKL = totalTestKL + KL;
    end
    KLvecTest(n) = totalTestKL;
    
    % afer updating A in one round, calculate total KL in training data
    totalTrainKL = 0;
    for s1 = 1 : trainNum
        KL = 0;
        f_result = A * F(s1,:)';
        numerator = sum(f_result);
        for i1 = 1:changeType
            KL = KL + f_result(i1)/numerator * ...
                log(f_result(i1)/numerator/L(s1,i1));
        end
        totalTrainKL = totalTrainKL + KL;
    end
    totalTrainKL   
    KLvecTrain(n) = totalTrainKL;
    
    if n > 1
        % if kl <= 0, we stop descending
        if totalTrainKL <= 0;
            flag = 0;
        end
        
        %  if kl in two rounds of iteration are quite similar, stop
        %  descending
        if abs(KLvecTrain(n-1)-KLvecTrain(n)) < ita
            flag = 0;
        end
    end
    
    % change step size when approaching the optimal
    if totalTrainKL < 5
        alpha = alpha_fine;
    end
    
    n = n + 1;   
end

%% Plot result
x = 1 : 1 : size(KLvecTrain,2);
y = 1 : 1 : size(KLvecTest,2);
plot(x, KLvecTrain, 'b');
hold on;
plot(y, KLvecTest, 'r');

%%%%%%%%%% Print out final result %%%%%%%%%%
Loss_KL = ['Loss Function on testingn data using KL method: ', ...
    num2str(min(KLvecTest))];
display(Loss_KL);

%% Evaluate random guess
for k = 1: numRand
    randKL = 0;
    L_rand = rand(sentenceNum, changeType);
    for n = trainNum + 1 : sentenceNum
        KL = 0;
        f_result = L_rand(n,:)';
        numerator = sum(f_result);
        for x = 1:changeType
            KL = KL + f_result(x)/numerator * ...
                log(f_result(x)/numerator/L(n,x));
        end
        randKL = randKL + KL;
    end
    totalRandKL(k) = randKL;
end
Loss_rand = ['Loss Function on testingn data using random gusess: ', ...
    num2str(mean(totalRandKL))];
display(Loss_rand);

% display available testing sentences number
querySentencesPool = sentenceMap(trainNum + 1 : sentenceNum,:)

    

