function nii = load_nii(filename)
   img_idx = [];
   dim5_idx = [];
   dim6_idx = [];
   dim7_idx = [];
   old_RGB = 0;
   tolerance = 0.1;			% 10 percent
   preferredForm= 's';		% Jeff
   v = version;
   [nii.hdr,nii.filetype,nii.fileprefix,nii.machine] = load_nii_hdr(filename);
   nii.ext = load_nii_ext(filename);
   [nii.img,nii.hdr] = load_nii_img(nii.hdr,nii.filetype,nii.fileprefix, ...
		nii.machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB);
   nii = xform_nii(nii, tolerance, preferredForm);
   return					% load_nii
