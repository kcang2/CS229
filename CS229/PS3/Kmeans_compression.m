large=double(imread('mandrill-large.tiff'));
imshow(uint8(round(large)));
small=double(imread('mandrill-small.tiff'));
L=reshape(large,[],3,1);
N=30; i=0; k=16;
A=reshape(small,[],3,1);
Jprev=10e5; eps=10e4; J=0;
miu=zeros(k,3); c=zeros(size(A,1),1);

miu=A(round(size(A,1)*rand(k,1)),:);

while( i < N && abs(J-Jprev) > eps )
    for j=1:size(A,1)
        [~,c(j,1)]=min(diag((A(j,:)-miu)*(A(j,:)-miu)'));
    end
    for j=1:k
        miu(j,:)=round(mean(A(find(c==j),:)));
    end
    i=i+1;
    Jprev=J;
    J=sum(diag((A-miu(c))*(A-miu(c))'));
end

for j=1:size(L,1)
    [~,swap]=min(diag((L(j,:)-miu)*(L(j,:)-miu)'));
    L(j,:)=miu(swap,:);
end

compressed=reshape(L,size(large,1),size(large,2),3);
imshow(uint8(round(compressed)));
