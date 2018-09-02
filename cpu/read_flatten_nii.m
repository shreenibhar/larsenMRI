function flat = read_flatten_nii(nii_name, thres)
	y = niftiread(nii_name);
	
    Size = size(y);
    a = Size(1);
    b = Size(2);
    c = Size(3);
    d = Size(4);
    itemp = 1;

    for i = 1:a
        for j = 1:b
            for k = 1:c
                if (mean(y(i,j,k,:)) > thres)
                    number(:,itemp) = y(i,j,k,:);
                    itemp = itemp + 1;
                end
            end
        end
    end

    flat = double(number);
end