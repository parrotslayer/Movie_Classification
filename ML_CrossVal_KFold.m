function [indices_train, indices_val] = ML_CrossVal_KFold(K_, N_)
    spaces = N_/K_;
    
    if K_ > N_ || floor(spaces) ~= spaces
        disp('ERROR K>N');
        return;
    end
    
    data = randperm(N_);
    
    
    data_split = zeros(K_,spaces);
    indices_train = [];
    
    
    count = 1;
    for i = 1:K_
        data_split(i,:) = data(count:count+spaces-1);
        count = count + spaces;   
    end
    
    indices_val = data_split;
    
    for i = 1:K_
        data_split_current = data_split;
        data_split_current(i,:) = [];
        aa = reshape(data_split_current, [], 1)';
        
        indices_train = [indices_train ; aa];
    end
    
    

end
