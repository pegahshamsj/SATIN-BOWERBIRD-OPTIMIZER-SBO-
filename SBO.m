%% preparing matlab
clc
clear
close all
%%Define problem parameters
varNum=2; % number of decision variables

lowerBound=-100 * ones(1,varNum); % lower bound for each decision variables
upperBound=100 * ones(1,varNum); % upper bound for each decision variables
%% SBO Parameters
N=50; % Number of search agents
maxLoop=100; % Maximum numbef of iterations
Alpha=0.9;
Pmutate=.05;
Z=.1;
Sigma=Z.*(upperBound- lowerBound) ;
%% Define Function and Solution
costFunction=@Spherefunction;
solution=[];
solution.Position=[];
solution.Cost=0;
solution.Fit=0;
%% Initialization Step
Bowers=repmat(solution,[N,1]);
for i=1:N
    Bowers(i).Position=unifrnd(lowerBound,upperBound);
    Bowers(i).Cost=costFunction(Bowers(i).Position,varNum);
    if Bowers(i).Cost>0;
        Bowers(i).Fit=1/(1+Bowers(i).Cost);
    else
        Bowers(i).Fit=1+abs(Bowers(i).Cost);
    end
end
% Find best Bowers
[value,index]=sort([Bowers.Cost]);
Elite=Bowers(index(1));
bestSoFar=zeros(1,maxLoop);
% Main loop
for it=1:maxLoop
    fit=[Bowers(:).Fit];
    P=fit./sum(fit);
    % Update the Position of search agents
    newBowers=repmat(solution,[N,1]);
    for i=1:N
        for v=1:varNum
            %%RouletteWheel
            cfd=cumsum(P);
            j=find(rand<cfd,1);
            Xj=Bowers(j).Position(v);
            %Step Size
            Lambda=Alpha/(1+P(j));
            %Update
            tmp=Xj+Elite.Position(v);
            tmp=tmp/2;
            newBowers(i).Position(v)=Bowers(i).Position(v)+Lambda*(tmp-Bowers(i).Position(v));            
            % Mutate
            if rand <Pmutate
                newBowers(i).Position(v)=Bowers(i).Position(v)+(Sigma(v)*randn);
            end
        end        
        % Check Boundari
        newBowers(i).Position=max(newBowers(i).Position,lowerBound);
        newBowers(i).Position=min(newBowers(i).Position,upperBound);
        %calc fitness
        newBowers(i).Cost=costFunction(newBowers(i).Position,varNum);
        if newBowers(i).Cost>0;
            newBowers(i).Fit=1/(1+newBowers(i).Cost);
        else
            newBowers(i).Fit=1+abs(newBowers(i).Cost);
        end
        %update Target
        if newBowers(i).Cost<Elite.Cost
            Elite=newBowers(i);
        end
    end
    %% Merge
    AllBowers=[Bowers;newBowers];
    [val,idx]=sort([AllBowers(:).Cost]);
    Bowers=AllBowers(idx(1:N));
    disp(['In Iteration= ' num2str(it) ' Best Cost= ' num2str(Elite.Cost)]);
    bestSoFar(it)=Elite.Cost;    
end
%% End of SBO and Display Result
disp(['best fitness =  ' num2str(Elite.Cost)]);
disp(' ');
disp(['best solution found is:  ' num2str( Elite.Position)])
%plot progress of SBO
x=1:maxLoop;
plot(x,bestSoFar)
xlabel('Iteration')
ylabel('Cost')
title('SBO Convergence For Sphere Function')

%plot Function
x=lowerBound(1):1:upperBound;
x=x';
y=x;
for i=1:length(x)
    for v=1:length(y)
        z(i,v)=costFunction([x(i),y(v)],2);
    end
end
figure,
surfc(x,y,z,'LineStyle','none')