function [confmat, acc, prec, rec, f1score] = algorithm_score(pred_, gt_)

gt_max = 4;

confmat = confusionmat(gt_,pred_);
TP = trace(confmat);


% Accuracy
acc = TP / sum(confmat(:));

% Precision, recall and f1score
prec = zeros(4,1);
rec = zeros(4,1);
f1score = zeros(4,1);

% Work out precision (fp)
    for i = 1:4
        tp_now = confmat(i,i);
        fp_now = sum(confmat(:,i))-tp_now; % sum along column
        fn_now = sum(confmat(i,:))-tp_now; % sum along row
        prec(i) = tp_now / (tp_now + fp_now);
        rec(i) = tp_now / (tp_now + fn_now);
        f1score(i) = 2*(prec(i)*rec(i))/(prec(i)+rec(i));
    end

end

