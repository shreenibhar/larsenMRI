function write_flatten_nii(nii_name, thres)
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

	number = double(number);
	f = fopen('flat.txt', 'w');
	sz = size(number);
	fprintf(f, '%d %d\n', sz(1), sz(2));
	fclose(f);
	save -append -ascii flat.txt number;
end