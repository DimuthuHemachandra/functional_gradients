
function exitcode = surf(left_labels,right_labels,gradient_file,out_path)


addpath '/home/dimuthu1/scratch/project2/Project_2/cfg/gifti-master';  % so matlab can find the library  
 mmpL = gifti(left_labels);  % load the left side gifti MMP atlas  
 mmpR = gifti(right_labels);  % and the right side MMP atlas  
 newvals = csvread(gradient_file,1); % 180 integers; new value for each parcel 
 for n=1:4
 newvals_R = newvals(1:180,n); %*1000 +3000;
 newvals_L = newvals(1:180,n+4); %*1000 +3000;
 %newvals_L
 %newvals_R
 Lout = mmpL;  % output gifti  
 Lout.cdata(:,1) = repelem(0, size(mmpL.cdata,1));  % replace the values with zeros  
 Rout = mmpR;  
 Rout.cdata(:,1) = repelem(0, size(mmpR.cdata,1)); 
 %mmpR.cdata
 %mmpL.cdata
 for i=1:180  % i = 1;  
   inds = find(mmpR.cdata == i);  % find vertices for parcel i  
   Rout.cdata(inds,1) = newvals_R(i);    
   inds = find(mmpL.cdata == (i+180)); % in MMP, left hemisphere vertices are 181:360  
   Lout.cdata(inds,1) = newvals_L(i);    
 end  
 save(Lout,strcat(out_path,'/plotL_grad_',int2str(n),'.func.gii'),'Base64Binary');  % save the gifti  
 save(Rout,strcat(out_path,'/plotR_grad_',int2str(n),'.func.gii'),'Base64Binary');
 end
end
