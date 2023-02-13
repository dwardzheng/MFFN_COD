
# image_path,mask_path  : Storage path for datasets
# h0,w0                 : image_size of original view
# rh1,rh2               : Resize ratio for different distance views
# [src_p1,src_p2,src_p3]: Points of the original image for Affine
# [dst_p1,dst_p2,dst_p3]: Points of the target   image for Affine
def Multi_view_Data(image_path,mask_path,h0,w0,rh1,rh2):
    # read image & mask
    image = read_image(image_path)
    mask  = read_mask(mask_path, to_normalize=True, thr=0.5) 
    # Generate different angle views
    image_a1 = cv2.flip(image, 0, dst=None)   # Vertical   Mirror Flip
    image_a2 = cv2.flip(image, 1, dst=None)   # Horizontal Mirror Flip
    image_a3 = cv2.flip(image, 1, dst=None)   # Diagonal   Mirror Flip
    # Generate different perspective views
    p1 = np.float32([src_p1,src_p2,src_p3])   # Points of the original image
    p2 = np.float32([dst_p1,dst_p2,dst_p3])   # Points of the target   image
    M1 = cv2.getAffineTransform(p1, p2)       # Affine transformation matrix
    image_p1 = cv2.warpAffine(image, M1, dsize=(h0,w0))  
    q1 = np.float32([src_q1,src_q2,src_q3])   # Points of the original image
    q2 = np.float32([dst_q1,dst_q2,dst_q3])   # Points of the target   image
    M2 = cv2.getAffineTransform(q1, q2)       # Affine transformation matrix
    image_p2 = cv2.warpAffine(image, M2, dsize=(h0,w0))  
    # Keep the original view
    image_or = image_resize(image, scale=1.0, h=h0, w=w0)
    # Generate different diatance views
    image_c1 = image_resize(image, scale=rh1, h=h0, w=w0)
    image_c2 = image_resize(image, scale=rh2, h=h0, w=w0)
    # Keep the original mask while training
    mask_ori = image_resize(mask,  scale=rh2, h=h0, w=w0)
    # Generate Tensor of different views
    image_a1 = torch.from_numpy(image_a1).permute(2, 0, 1)
    image_a2 = torch.from_numpy(image_a2).permute(2, 0, 1)
    image_or = torch.from_numpy(image_or).permute(2, 0, 1)
    image_c1 = torch.from_numpy(image_c1).permute(2, 0, 1)
    image_c2 = torch.from_numpy(image_c2).permute(2, 0, 1)
    image_p1 = torch.from_numpy(image_p1).permute(2, 0, 1)
    image_p2 = torch.from_numpy(image_p2).permute(2, 0, 1)
    mask_ori = torch.from_numpy(mask_ori).permute(2, 0, 1)
    # image_a1,image_a2,image_or,image_a1,image_a3 are selected!
    return dict(
        data={
            "image_a1": image_a1,
            "image_a3": image_a3,
            "image_or": image_or,
            "image_c1": image_c1,
            "image_c2": image_c2,
            # "image_p1": image_p1,
            # "image_p2": image_p2,
            "mask_ori": mask_ori,
        }
    )
