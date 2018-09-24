clear;clc;close all;

Da=load('data_a.txt');
Db=load('data_b.txt');
Xa=Da(:,2:end);
Xa=[ones(100,1) Xa]; % ADD INTERCEPT
Xb=Db(:,2:end);
Xb=[ones(100,1) Xb]; % ADD INTERCEPT
ya=Da(:,1);
yb=Db(:,1);
m=length(ya);
ta=zeros(3,1); tb=zeros(3,1);
a=10; % REDUCE LEARNING RATE DOES NOT HELP CONVERGENCE OF B

% DO NOT FORGET SIGMOID RANGES FROM 0 to 1
ya(find(ya==-1))= 0;
yb(find(yb==-1))= 0;

ia=0; ib=0;

% WHILE LOOP TO TRAIN ON DATASET A (COMPARES CHANGE IN MAGNITUDE OF THETA)

% while true
%     pta=ta;
%     ia=ia+1;
%     ta = ta - a/m.*Xa'*( 1./( ones(m,1)+exp(-Xa*ta) ) - ya );
%     if(mod(ia,10000)==0)
%         fprintf('ia is %d\n',ia);
%         fprintf('norm(pta-ta) is %d\n',norm(pta-ta));
%     end
%     if(norm(pta-ta)<10e-15) 
%         break;
%     end
%     
% end

while true
    ptb=tb;
    ib=ib+1;
    tb = tb - a/m.*Xb'*( 1./( ones(m,1)+exp(-Xb*tb) ) - yb );
    if(mod(ib,10000)==0)
        fprintf('ib is %d\n',ib);
        fprintf('norm(ptb-tb) is %d\n',norm(ptb-tb));
    end
    if(norm(ptb-tb)<10e-3) % RELAXED STOPPING CRITERIA ENSURES CONVERGENCE
        break;
    end
    
end

% STOPPING CRITERIA COMPARES MAGNITUDE OF GRADIENT (del_J/del_Theta)
% ACTUALLY SAME AS norm(Theta - Theta_previous) SHOWN ABOVE

% ia=0;
% ia = 0; a = 10;
% while( norm(logistic_grad(Xa, ya, ta)) > 10e-12 )
%     ia = ia + 1;
%     ta = ta - a*logistic_grad(Xa, ya, ta);
% end

% COST FUNCTION AS STOPPING CRITERIA (CANNOT CONVERGE FOR DATASET B ALSO)

% while true
%     i = i + 1;
%     tp=t;
%     t = t - a*logistic_grad(X, y, t);
%     J=y'*log(sigmoid(X*t))+(ones(length(y),1)-y)'*log(ones(length(y),1)-sigmoid(X*t));
%     Jp=y'*log(sigmoid(X*tp))+(ones(length(y),1)-y)'*log(ones(length(y),1)-sigmoid(X*tp));
%     if( abs(J-Jp) <= 10e-12 ) % CHECK DIFFERENCE OF J FOR CONVERGENCE
%         break;
%     end
% end

% DATA FOR B IS PERFECTLY SEPARABLE, THATS WHY THETA WILL KEEP INCREASING
% AND NOT CONVERGE. NEED TIKONOV REGULARISATION / RELAX STOPPING CRITERIA

%% DIFF(THETA) <-> DEL_J / DEL_THETA <-> DIFF(J)

% figure
% hold on
% for i=1:length(yb)
% if yb(i)==1
% plot(Xb(i,1),Xb(i,2),'bx')
% elseif yb(i)==-1
%     plot(Xb(i,1),Xb(i,2),'ro')
% else
%     fprintf('unknown label\n');
% end
% end

% figure
% hold on
% ya=Da(:,1);
% Xa=Da(:,2:end);
% for i=1:length(ya)
% if ya(i)==1
% plot(Xa(i,1),Xa(i,2),'bx')
% elseif ya(i)==-1
%     plot(Xa(i,1),Xa(i,2),'ro')
% else
%     fprintf('unknown label\n');
% end
% end