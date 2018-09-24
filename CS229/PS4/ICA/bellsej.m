%------------------------------------------------------------
% ICA

load mix.dat	% load mixed sources
% Fs = 11025; %sampling frequency being used
% 
% % listen to the mixed sources
% % norm each mixed source with its abs max sample. (audio can be -ve)
% normalizedMix = 0.99 * mix ./ (ones(size(mix,1),1)*max(abs(mix)));
% 
% % handle writing in both matlab and octave
% v = version;
% if (v(1) <= '3') % assume this is octave
%   audiowrite('mix1.audio', normalizedMix(:, 1), Fs, 16);
%   audiowrite('mix2.audio', normalizedMix(:, 2), Fs, 16);
%   audiowrite('mix3.audio', normalizedMix(:, 3), Fs, 16);
%   audiowrite('mix4.audio', normalizedMix(:, 4), Fs, 16);
%   audiowrite('mix5.audio', normalizedMix(:, 5), Fs, 16);
% else
%   audiowrite('mix1.wav',normalizedMix(:,1),Fs,'BitsPerSample',16);
%   audiowrite('mix2.wav',normalizedMix(:,2),Fs,'BitsPerSample',16);
%   audiowrite('mix3.wav',normalizedMix(:,3),Fs,'BitsPerSample',16);
%   audiowrite('mix4.wav',normalizedMix(:,4),Fs,'BitsPerSample',16);
%   audiowrite('mix5.wav',normalizedMix(:,5),Fs,'BitsPerSample',16);
% end
p = randperm(size(mix,1));
W = eye(5);	% initialize unmixing matrix

% this is the annealing schedule I used for the learning rate.
% (We used stochastic gradient descent, where each value in the 
% array was used as the learning rate for one pass through the data.)
% Note: If this doesn't work for you, feel free to fiddle with learning
%  rates, etc. to make it work.
anneal = [0.1 0.1 0.1 0.05 0.05 0.05 0.02 0.02 0.01 0.01 ...
      0.005 0.005 0.001 0.001 0.001 0.001];
   
for iter=1:length(anneal)
   %%%% here comes your code part

   for j=1:size(p,2) 
    W = W + anneal(iter)*( ( 1 - 2./( 1 + exp(-W*mix(p(j),:)') ) )...
        *mix(p(j),:) + inv(W') );
   end
   
end

%%%% After finding W, use it to unmix the sources.  Place the unmixed sources 
%%%% in the matrix S (one source per column).  (Your code.) 

S = mix * W';

S=0.99 * S./(ones(size(mix,1),1)*max(abs(S))); 	% rescale each column to have maximum absolute value 1 

% now have a listen --- You should have the following five samples:
% * Godfather
% * Southpark
% * Beethoven 5th
% * Austin Powers
% * Matrix (the movie, not the linear algebra construct :-) 

% v = version;
% if (v(1) <= '3') % assume this is octave
%   audiowrite('unmix1.audio', S(:, 1), Fs, 16);
%   audiowrite('unmix2.audio', S(:, 2), Fs, 16);
%   audiowrite('unmix3.audio', S(:, 3), Fs, 16);
%   audiowrite('unmix4.audio', S(:, 4), Fs, 16);
%   audiowrite('unmix5.audio', S(:, 5), Fs, 16);
% else
  audiowrite('unmix1.wav',S(:,1),Fs,'BitsPerSample',16);
  audiowrite('unmix2.wav',S(:,2),Fs,'BitsPerSample',16);
  audiowrite('unmix3.wav',S(:,3),Fs,'BitsPerSample',16);
  audiowrite('unmix4.wav',S(:,4),Fs,'BitsPerSample',16);
  audiowrite('unmix5.wav',S(:,5),Fs,'BitsPerSample',16);
% end
