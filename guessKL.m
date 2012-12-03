featureNum = 10;
changeType = 4;
sentenceNum = 100;
totalKL = 0;

L = rand(sentenceNum, changeType);
% L_guess = rand(sentenceNum, changeType);
L_guess = f_result;
for i=1:sentenceNum
    sumValue = sum(L(i,:));
    L(i,:) = L(i,:)/sumValue;
%     sum2 = sum(L_guess(i,:));
%     L_guess(i,:) = L_guess(i,:)/sum2;
end

for n = 1:sentenceNum
    KL = 0;
    for x = 1:changeType
        KL = KL + L_guess(x)*log(L_guess(x)/L(n,x));
    end
    totalKL = totalKL + KL;
end
totalKL
    

