featureNum = 10;
changeType = 4;
sentenceNum = 100;
trainNum = 100;
ita = 1e-1;
alpha = 0.5;
maxGradient = 1;

F = randi(2, sentenceNum, featureNum) - 1;
A = rand(changeType, featureNum);
L = rand(sentenceNum, changeType);
KLvec = zeros(1,trainNum);
gradientMatrix = zeros(changeType, featureNum);

for i=1:sentenceNum
    sumValue = sum(L(i,:));
    L(i,:) = L(i,:)/sumValue;
end

for i=1:changeType
    sumValue = sum(A(i,:));
    A(i,:) = A(i,:)/sumValue;
end
flag = 1;
n = 1;
% for n = 1:trainNum
while flag == 1
    totalKL = 0;
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
                gradientAij = F(s,j)*log(f(i)/L(s,i)) + L(s,i)*F(s,j);
                gradientMatrix(i,j) = gradientAij;
                if abs(gradientAij) > maxGradient
                    maxGradient = abs(gradientAij);
                end
                tempA(i,j) = A(i,j)-A(i,j)*0.9*gradientAij/maxGradient;                
%                 tempA(i,j) = A(i,j)-A(i,j)*sign(gradientAij)*max(0.5, 0.6*abs(gradientAij)/maxGradient);
%                 tempA(i,j) = A(i,j) - (1/n^alpha) * gradientAij;
%                 tempA(i,j) = A(i,j) - ita * gradientAij;
            end
        end
        A = tempA;
        for x = 1:changeType
            sumValue = sum(A(x,:));
            A(x,:) = A(x,:)/sumValue;
        end
    end
    
    for s1 = 1:sentenceNum
        KL = 0;
        f_result = A * F(s1,:)';
        for idx=1:changeType
            sumValue = sum(f_result);
            f_result = f_result/sumValue;
        end
        for i1 = 1:changeType
            KL = KL + f_result(i1)*log(f_result(i1)/L(s1,i1));
        end
        totalKL = totalKL + KL;
    end
    KLvec(n) = totalKL;
    if n > 1
        if abs(KLvec(n-1)-KLvec(n)) < 1e-7
            flag = 0;
        end
    end
    n = n + 1;
end
x = 1 : 1 : size(KLvec,2);
plot(x, KLvec)
        
    

