#   xvaltrend = median(temp(pcvoxels,:),1);

#   % choose number of PCs
#   chosen = 0;  % this is the fall-back
#   if opt.pcstop <= 0
#     chosen = -opt.pcstop;  % in this case, the user decides
#   else
#     curve = xvaltrend - xvaltrend(1);  % this is the performance curve that starts at 0 (corresponding to 0 PCs)
#     mx = max(curve);                   % store the maximum of the curve
#     best = -Inf;                       % initialize (this will hold the best performance observed thus far)
#     for p=0:opt.numpcstotry
  
#       % if better than best so far
#       if curve(1+p) > best
    
#         % record this number of PCs as the best
#         chosen = p;
#         best = curve(1+p);
      
#         % if we are within opt.pcstop of the max, then we stop.
#         if best*opt.pcstop >= mx
#           break;
#         end
      
#       end
  
#     end
#   end

def select_noise_regressors():
    pass