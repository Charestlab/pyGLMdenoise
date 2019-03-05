import numpy
#   % prepare to select optimal number of PCs
#   temp = squish(pcR2,dimdata);  % voxels x 1+pcs
#   pcvoxels = any(temp > opt.pcR2cutoff,2) & squish(opt.pcR2cutoffmask,dimdata);  % if pcR2cutoffmask is 1, this still works
#   if ~any(pcvoxels)
#     warning(['No voxels passed the threshold for the selection of the number of PCs. ' ...
#              'We fallback to simply using the top 100 voxels (i.e. compute each voxel''s maximum ' ...
#              'cross-validation accuracy under any number of PCs and then choose the top 100 voxels.']);
#     if isequal(opt.pcR2cutoffmask,1)
#       ix = 1:size(temp,1);
#     else
#       ix = find(squish(opt.pcR2cutoffmask,dimdata));
#     end
#     pcvoxels = logical(zeros(size(temp,1),1));
#     temp2 = max(temp(ix,:),[],2);  % max cross-validation for each voxel (within mask)
#     [d,ix2] = sort(temp2,'descend');
#     pcvoxels(ix(ix2(1:min(100,length(ix2))))) = 1;
#   end
#   xvaltrend = median(temp(pcvoxels,:),1);

def select_voxels_nr_selection(r2_voxels_nrs, cutoff=0, mask=None):
    """Chooses voxels for noise regressor selection
    
    Arguments:
        r2_voxels_nrs (ndarray): R2 voxels by 1 + npcs
    
    Keyword Arguments:
        cutoff (int): Min R2 across solutions at which a voxel is selected (default: 0)
        mask (ndarray): Optional mask of voxels to consider (default: None)
    
    Returns:
        ndarray: one dimensional voxel mask (boolean)
    """

    return numpy.any(r2_voxels_nrs > cutoff, 1)