%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This function is used to support user query based on a particular
%%% sentence and return KL distance, plot distribution graphs
%%% To run the query, you must run "gradient.m" first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = query(sentenceNum)

    global sentenceMap F L A;
    rowIndex = find(strcmp(sentenceMap(:,1), sentenceNum));
    confidence = sentenceMap(rowIndex, 2);
    realValue = L(rowIndex,:)';
    estimatedValue = A*F(rowIndex, :)';
    Y = [realValue estimatedValue];
    %% plot basic bars
    subplot(2,1,1);
    bar(Y);
    set(gca, 'XTickLabel', {'del', 'ins', 'r_const', 'r_pred', ...
        'r_conn', 'r_mix', 'move'});
    
    %% plot difference
    subplot(2,1,2);
    diffY = Y(:,1) - Y(:,2);
    bar(diffY, 'g');
    set(gca, 'XTickLabel', {'del', 'ins', 'r_const', 'r_pred', ...
        'r_conn', 'r_mix', 'move'});
    
    %% calculate KL distance and log
    totalKL = 0;
    KL = 0;
    changeType = size(realValue);
    for x = 1: changeType
        KL = KL + estimatedValue(x)*log(estimatedValue(x)/realValue(x));
    end
    totalKL = totalKL + KL;
    result = ['KL distance: ', num2str(totalKL), 'Confidence: ', ...
        confidence];
    display(result);
    
end