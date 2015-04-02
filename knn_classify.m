function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)
% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  k: number of nearest neighbors
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CSCI 576 2014 Fall, Homework 1

[train_N,feature_train_N]=size(train_data);
[new_N,feature_new_N]=size(new_data);
class_N=length(unique(train_label));  % label dataset must be like {1,2,3,4}, not binary

M_train=mean(train_data);
S_train=std(train_data);
M_new=mean(new_data);
S_new=std(new_data);

%train_result=repmat(0,train_N,class_N);
%new_result=repmat(0,new_N,class_N);
train_result=zeros(train_N,1);
new_result=zeros(new_N,1);

% normalize data
train_data=(train_data-repmat(M_train,train_N,1))./repmat(S_train,train_N,1);
new_data=(new_data-repmat(M_new,new_N,1))./repmat(S_new,new_N,1);

%calculate distance for train_data, LOO
for n=1:train_N
    n_mat=train_data-repmat(train_data(n,:),train_N,1);
    dis_train=sum(abs(n_mat).^2,2).^0.5;
    order=sort(dis_train);
    new_count=zeros(1,class_N); % buffer to mark number of class in [1,2,3,4]
    for k_new=1:k
        r=find(dis_train==order(k_new+1));
        if length(r)>1
            b=length(r);
            for c=1:b          % in case there are points that have same distance with point k
                position=train_label(r(c));
                new_count(position)=new_count(position)+1;
            end
        else
            position=train_label(r);
            new_count(position)=new_count(position)+1;
        end
        a=find(new_count==max(new_count));
        if length(a)>1
            a=a(randperm(length(a)));     % in case numbers of some classes are equal, random
            a=a(1);
        end
        train_result(n)=a;
    end
end
    %classify train_data
    train_accu=sum(train_result==train_label)/train_N
    
    %calculate distance for new_data, kNN
    for n=1:new_N
        n_mat=train_data-repmat(new_data(n,:),train_N,1);
        dis_new=sum(abs(n_mat).^2,2).^0.5;
        order=sort(dis_new);
        new_count=zeros(1,class_N); % buffer to mark number of class in [1,2,3,4]
        for k_new=1:k
            r=find(dis_new==order(k_new));
            if length(r)>1
                b=length(r);
                for c=1:b          % in case there are points that have same distance with point k
                    position=train_label(r(c));
                    new_count(position)=new_count(position)+1;
                end
            else
                position=train_label(r);
                new_count(position)=new_count(position)+1;
            end
        end
        a=find(new_count==max(new_count));
        if length(a)>1
            a=a(randperm(length(a)));     % in case numbers of some classes are equal, random
            a=a(1);
        end
        new_result(n)=a;
    end
    
    new_accu=sum(new_result==new_label)/new_N
    
    
    
