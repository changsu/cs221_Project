totalKL = 0;

L = rand(sentenceNum, changeType);
% L_guess = rand(sentenceNum, changeType);
L_guess = f_result;

for n = 1:sentenceNum
    KL = 0;
    for x = 1:changeType
        KL = KL + L_guess(x)*log(L_guess(x)/L(n,x));
    end
    totalKL = totalKL + KL;
end
totalKL
    

