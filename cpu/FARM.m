function FARM(fMRI_filename,lambda,output_foldername)

    thres = 6000
	fMRI_filename  = char(fMRI_filename);
	
	% load nii fmri file 
	intial_location=load_nii(fMRI_filename);
	y = intial_location.img;
	
    Size = size(y);
    a = Size(1);
    b = Size(2);
    c = Size(3);
    d = Size(4);
    itemp = 1;
    
	% Threshold input file and convert 3D array to 1D
    for i = 1:a
        for j = 1:b
            for k = 1:c
                if (mean(y(i,j,k,:)) > thres)
                    number(:,itemp) = y(i,j,k,:);
                    itemp = itemp+1;
                end
            end
        end
    end
    
	% Intialization of paramters
    number = double(number);
    Size1 = size(number);
    a1 = Size1(1);
    b1 = Size1(2);
    %Beta = zeros(b1,b1);
    beta3 = zeros(b1,1);
    beta4 = zeros(b1,1);
    beta5 = zeros(b1,1);
    beta6 = zeros(b1,1);
    
	% Granger Causality Analysis
    tic
    for i1 = 1:b1
        
        disp(i1);
        number1 = number;
        number1(:,i1) = [];
        temp1 = number(2:d,i1);
        temp2 = number1(1:(d-1),:);
        
        % normalize data
        new = zscore(temp1) ;
        new1 = zscore(temp2);
        
		% Lasso function
        [beta ,steps,G,residuals,error,drop] = larsen_gpu(new1, new, 0, 0, lambda);

        %if i1 == 1
         %   Beta(i1,1) = 0;
          %  Beta(i1,2:b1) = beta; 
        %elseif i1 == b1
         %   Beta(i1,1:(b1-1)) = beta;
          %  Beta(i1,b1) = 0;
        %else
         %   Beta(i1,1:(i1-1)) = beta(1:(i1-1));
          %  Beta(i1,i1) = 0;
           % Beta(i1,(i1+1):b1) = beta((i1):(b1-1));
        %end
        
        % beta3(i1) = steps;
         %beta4(i1) = (residuals)^2;
        
       toc 
    end
    %beta1 = sum(abs(Beta),1); % prediction power
    %beta2 = sum(abs(Beta),2); % effect of all voxels on a particular voxel
    %invbeta1 = zeros(a,b,c);
    %invbeta2 = zeros(a,b,c);
    %invbeta3 = zeros(a,b,c);
    %invbeta4 = zeros(a,b,c);
    %itemp = 1;
	% Converting all 1D array to 3D array
    %{
    for i = 1:a
        for j = 1:b
            for k = 1:c
                if (mean(y(i,j,k,:)) > 6000)
                    invbeta1(i,j,k) = beta1(itemp);
                    invbeta2(i,j,k) = beta2(itemp);
                    invbeta3(i,j,k) = beta3(itemp);
                    invbeta4(i,j,k) = beta4(itemp);
                    itemp = itemp+1;
                end
            end
        end
    end
	% Save all the files
    csvwrite(strcat(output_foldername,'/beta.csv'),Beta);
    nii_file = make_nii(single(invbeta1));
    nii_file.hdr = intial_location.hdr;
    nii_file.hdr.dime.dim(5) = 1;
    save_nii(nii_file,strcat(output_foldername,'/beta1.nii.gz'));
    nii_file = make_nii(single(invbeta2));
    nii_file.hdr = intial_location.hdr;
    nii_file.hdr.dime.dim(5) = 1;
    save_nii(nii_file,strcat(output_foldername,'/beta2.nii.gz'));
    nii_file = make_nii(single(invbeta3));
    nii_file.hdr = intial_location.hdr;
    nii_file.hdr.dime.dim(5) = 1;
    save_nii(nii_file,strcat(output_foldername,'/steps.nii.gz'));
    nii_file = make_nii(single(invbeta4));
    nii_file.hdr = intial_location.hdr;
    nii_file.hdr.dime.dim(5) = 1;
    save_nii(nii_file,strcat(output_foldername,'/residuals.nii.gz'));
    %}
    clear
    quit