function [restored] = restore(im, impulsefft, useThreshold, threshold, plot)
im = double(im);
fft = fft2(im);
H = 1 ./ impulsefft;
if (useThreshold)
    for i=1:size(im, 1)
        for j=1:size(im, 2)
            if (abs(H(i, j)) >= threshold)
                H(i, j) = threshold * abs(impulsefft(i, j)) / impulsefft(i, j);
            end
        end
    end
end
[M, N] = size(im);
mult_exp_spatial = zeros(M, N);
for i = 1:M
    for j = 1:N
        mult_exp_spatial(i, j) = exp(-1i * 2* pi * (-M/2 * i./M + (-N/2) * j /N)); 
    end
end
restored = ifft2(H .* fft .* mult_exp_spatial);
maximum = max(restored, [], 'all');
restored = uint8(restored .* (255 / maximum));
if (plot == 1)
    figure();
    imshow(restored);
    title('restored image');
end

