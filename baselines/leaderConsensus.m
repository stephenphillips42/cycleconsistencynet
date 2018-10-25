function [X,Pit] = leaderConsensus(Pijt,dimGroup,K,param)


flagLaplacian = 0;


if ~isfield(param,'nIter')
    nIter =200;
end

if ~isfield(param,'Adj')
    Adj = ones(numel(dimGroup));
    Adj(1:size(Adj,1)+1:end) = 0;
else
    Adj = param.Adj;
end


if ~isfield(param,'leadid')
    leadid = 1;
else
    leadid = param.leadid;
end




%% main stuff here
n= dimGroup(:)';
sumn = sum(n);
cumn = [0 cumsum(n)]; 
idxlead = cumn(leadid)+1:cumn(leadid+1);

nViews = size(Adj,1);
AdjI =  eye(nViews) + Adj;


Pt = zeros(sumn); % global pairwise matches,   input to the algorithm

for iView=1:nViews
    for jView=1:nViews
        
        if AdjI(iView,jView) 
    
            idxr = cumn(iView)+1:cumn(iView+1);
            idxc = cumn(jView)+1:cumn(jView+1);
            Pt(idxr,idxc) =  Pijt(idxr,idxc);
         
            
            
        end
        
        

    end
end



if flagLaplacian
   step = 1/(2*max(sum(Pt)));
   Lap = diag( sum(Pt,2))-Pt;
   Lap(idxlead,:) = 0;
   Pts = eye(size(Pt))-step*Lap;
   
else

    % make rows sum to 1 
    Pts = Pt;
    for ir=1:sumn
        Pts(ir,:) = Pt(ir,:)/sum(Pt(ir,:));
    end

    % keep leader fixed

    Pts(idxlead,:) = 0;
    Pts(idxlead,idxlead) = eye(K);
end






% initialization
Pi0 = (1/K)*ones(sumn,K);


%Ik = eye(K);
%for iView=1:nViews   
%    idxr = cumn(iView)+1:cumn(iView+1);
%    Pi0(idxr,:) = Ik(:,randperm(K));
%end

Pi0(idxlead,:) = eye(K);


Pi=Pi0;



    
V = zeros(nIter,1); 
% main loop
for iIter=1:nIter
   
    
    
        if 1

            for i=1:size(Adj,1)
                idxr =  cumn(i)+1: cumn(i+1);
                for j=i+1:size(Adj,2)
                    idxc =  cumn(j)+1: cumn(j+1);
                    if Adj(i,j), 
                        V(iIter) = V(iIter) + .5*norm( Pijt(idxr,idxc)*Pi(idxc,:)-Pi(idxr,:) ,'fro')^2; 
                    end   
                end
            end

        end
    
     Pi = Pts * Pi;
     
end




% thresholding using Munkres
Pit = zeros(size(Pi)); 


for iView=1:nViews
        
    idxr = cumn(iView)+1:cumn(iView+1);

    if iView == leadid
        Pit(idxr,:) = speye(K);
    else

        [ass,~] = munkres(-Pi(idxr,:)); 
        idx=1:n(iView);
        Pit(idxr,:) = sparse(idx(ass >0),ass(ass >0),1,n(iView),K);  

    end
        
end


X = Pit*Pit';
   

% plot cost
%if exist('V')
%figure,plot(V)
%end
%pause(1)


end

