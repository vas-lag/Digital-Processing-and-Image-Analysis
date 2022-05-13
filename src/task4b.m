clear all;
clc;
t = Tiff('../images/4/chart.tiff');
original = read(t);
imshow(original);
title('original image');
F = fft2(original);
impsf = psf(original);
maximum = max(impsf, [], 'all');
im = uint8(impsf .* (255 ./ maximum));
figure();
imshow(im);
title('image after psf');

delta = zeros(size(im));
delta(size(im, 1) / 2, size(im, 2) / 2) = 1;
maximum = max(delta, [], 'all');
delta = delta .* (255 / maximum);
impulse = psf(delta);
impulsefft = fft2(impulse);
figure();
imshow(log10(abs(fftshift(impulsefft))))
title('frequency response of psf.p');

restored = restore(im, impulsefft, 1, 1e-8, 1);

error_vec = [];
for t=1:5:100
    threshold = t * 1e-6;
    restored = restore(impsf, F, 1, threshold, 0);
    error = double(original) - double(restored);
    error_vec(end + 1) = mean2(error .* error);
end
size(error_vec, 2);
x = 1:5:100;
figure();
plot(x * 1e-6, error_vec);
xlabel('threshold');
ylabel('MSE');
