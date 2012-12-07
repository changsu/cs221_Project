%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This function is used to support user query based on a particular
%%% sentence and return KL distance, plot distribution graphs
%%% To run the query, you must run "gradient.m" first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = query(sentenceNum, method)

    global sentenceMap F L A;

    rowIndex = find(strcmp(sentenceMap(:,1), sentenceNum));
    confidence = sentenceMap(rowIndex, 2);
    realValue = L(rowIndex,:)'
    
    % if it's kl method, we need to normalize to get the final result
    if (strcmp(method, 'kl'))
        estimatedValue = A*F(rowIndex, :)'/sum(A*F(rowIndex, :)')
    elseif(strcmp(method, 'rl'))
        estimatedValue = A*F(rowIndex, :)'
    else
        error('Usage: query(sentenceNumber, method) where method is rl or kl');
    end
    Y = [realValue estimatedValue];
    %% plot basic bars
    subplot(2,1,1);
    bar(Y);
    set(gca, 'XTickLabel', {'del', 'ins', 'r_const', 'r_pred', ...
        'r_conn', 'r_mix', 'permute'});
    
    %% plot difference
    subplot(2,1,2);
    diffY = Y(:,1) - Y(:,2);
    bar(diffY, 'g');
    set(gca, 'XTickLabel', {'del', 'ins', 'r_const', 'r_pred', ...
        'r_conn', 'r_mix', 'permute'});
    
    %% calculate KL distance and log
    totalKL = 0;
    KL = 0;
    changeType = size(realValue);
    for x = 1: changeType
        KL = KL + estimatedValue(x)*log(estimatedValue(x)/realValue(x));
    end
    totalKL = totalKL + KL;
    if (strcmp(method, 'kl'))
        result = ['KL distance: ', num2str(totalKL), 'Confidence: ', ...
            confidence];
    else
        result = ['norm2 distance: ', ...
            num2str(norm(realValue - estimatedValue),2), ...
            'Confidence: ', confidence];
    end
    display(result);
    
end