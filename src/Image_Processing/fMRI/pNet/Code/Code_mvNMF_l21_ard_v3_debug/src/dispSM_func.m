function dispSM_func(smNiiName,refNiiName,outFigName)

smNii = load_untouch_nii(smNiiName);
refNii = load_untouch_nii(refNiiName);

% get the peak point in the sm
[peakVal,peak] = max(smNii.img(:));
[peakX,peakY,peakZ] = ind2sub(size(smNii.img),peak);

sm_view = cell(1,3);
sm_view{1} = rot90(squeeze(smNii.img(:,:,peakZ)));
sm_view{2} = rot90(squeeze(smNii.img(peakX,:,:)));
sm_view{3} = rot90(squeeze(smNii.img(:,peakY,:)));
% interpolate the peak position
sz_factor = smNii.hdr.dime.pixdim(2)/refNii.hdr.dime.pixdim(2);
tPeak = round([peakX,peakY,peakZ]*sz_factor);

ref_view = cell(1,3);
ref_view{1} = rot90(squeeze(refNii.img(:,:,tPeak(3))));
ref_view{2} = rot90(squeeze(refNii.img(tPeak(1),:,:)));
ref_view{3} = rot90(squeeze(refNii.img(:,tPeak(2),:)));

for vi=1:3
    minVal = min(sm_view{vi}(:));
    maxVal = max(sm_view{vi}(:));
    if maxVal~=minVal
        sm_view{vi} = (sm_view{vi}-minVal)/(maxVal-minVal);
    end
    sm_view{vi} = imresize(sm_view{vi},size(ref_view{vi}),'bilinear');
end

% colormap
figure('Visible','Off');
cmap = colormap(hot(255));
for vi=1:3
    subplot(1,3,vi);
    % anatomical image
    h = imshow(ref_view{vi},[min(ref_view{vi}(:)),max(ref_view{vi}(:))]);
    % anatomical transparency
    anaAlpData = ones(size(ref_view{vi}));
    anaAlpData(ref_view{vi}==0) = 0;
    set(h, 'AlphaData', anaAlpData);
    
    hold on;
    
    % spatial map
    sm_rgb = ind2rgb(round(sm_view{vi}*length(cmap)),cmap);
    h = imshow(sm_rgb);
    % spatial map transparency
    set(h, 'AlphaData', sm_view{vi});
    % figure position
    set(gca,'position',[(vi-1)*0.33+0.01, 0, 0.32, 1],'units','normalized');
end

% save the figure
width = 500;
height = 150;
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'points');
myfiguresize = [0, 0, width, height];
set(gcf, 'PaperPosition', myfiguresize);

print('-dtiff','-r150',outFigName);

