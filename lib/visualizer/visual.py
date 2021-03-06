import math
import nibabel as nib
import torch
import torch.nn.functional as F
import sys,os
sys.path.append('/home/zongdaoming/cv/multi-organ/')
sys.path.insert(0,'/home/zongdaoming/cv/multi-organ/lib')

# import sys
# # Or sys.path.append()
# sys.path.insert(0, './bar')
# from bar.eggs import Eggs
# from foo.ham import Ham

from lib.visualizer.visual_2d import *
import types
import SimpleITK as sitk
import nibabel as nib
import medpy


dicom_windows = types.SimpleNamespace(
    brain=(80,40),
    subdural=(254,100),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400),
    custom = (200,60)
)


def read_nii_gz(filename):
    itk_img = sitk.ReadImage(filename)
    img_or_label_arr = sitk.GetArrayFromImage(itk_img)
    # print(img_array.shape)
    # output: (frame_num, width, height) (574, 512, 512)
    print(f"Image or label shape is {img_or_label_arr.shape}")
    return img_or_label_arr


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)


def loading_nii(filename):
    # image_data is a numpy ndarray with the image data and
    # image_header is a header object holding the associated metadata.
    from medpy.io import load
    image_data, image_header = load(filename)
    # print(f"Image or label shape is: {image_data.shape}")
    return image_data


def windowed(image, w, l):
    # 2d tensor: image 
    px = image
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)
    # plt.imshow(tensor(sample_ct[...,1].astype(np.float32)).windowed(*dicom_windows.liver), cmap=plt.cm.bone);
    # sample_ct[:,:,1] == sample_ct[...,1]


def plot_sample(image, label, idx, color_map = 'nipy_spectral'):
    '''Plots and a slice with all available annotations
    '''
    fig = plt.figure(figsize=(12,5))

    plt.subplot(1,4,1)
    plt.imshow(image, cmap='bone')
    plt.title('Original Image')
    
    plt.subplot(1,4,2)
    plt.imshow(windowed(image.astype(np.float32),*dicom_windows.liver), cmap='bone');
    plt.title('Windowed Image')
    
    plt.subplot(1,4,3)
    plt.imshow(label, alpha=0.5, cmap=color_map)
    plt.title('Mask')
    
    plt.subplot(1,4,4)
    # plt.imshow(image, cmap='bone')
    plt.imshow(windowed(image.astype(np.float32),*dicom_windows.liver), cmap='bone');
    plt.imshow(label, alpha=0.35, cmap='jet')
    plt.title('Liver & Mask')
    plt.savefig("/home/zongdaoming/cv/multi-organ/multi-organ-ijcai/lib/visualizer/images/image_{}.png".format(idx),dpi=300)
    plt.show()


def test_padding():
    x = torch.randn(1, 144, 192, 256)
    kc, kh, kw = 32, 32, 32  # kernel size
    dc, dh, dw = 32, 32, 32  # stride
    # Pad to multiples of 32
    x = F.pad(x, (x.size(3) % kw // 2, x.size(3) % kw // 2,
                  x.size(2) % kh // 2, x.size(2) % kh // 2,
                  x.size(1) % kc // 2, x.size(1) % kc // 2))
    print(x.shape)
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    print(unfold_shape)
    patches = patches.contiguous().view(-1, kc, kh, kw)
    print(patches.shape)

    # Reshape back
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    # Check for equality
    print((patches_orig == x[:, :output_c, :output_h, :output_w]).all())


def roundup(x, base=32):
    return int(math.ceil(x / base)) * base


def non_overlap_padding(args, full_volume, model,criterion, kernel_dim=(32, 32, 32)):

    x = full_volume[:-1,...].detach()
    target = full_volume[-1,...].unsqueeze(0).detach()
    #print(target.max())
    #print('full volume {} = input {} + target{}'.format(full_volume.shape, x.shape,target.shape))

    modalities, D, H, W = x.shape
    kc, kh, kw = kernel_dim
    dc, dh, dw = kernel_dim  # stride
    # Pad to multiples of kernel_dim
    a = ((roundup(W, kw) - W) // 2 + W % 2, (roundup(W, kw) - W) // 2,
         (roundup(H, kh) - H) // 2 + H % 2, (roundup(H, kh) - H) // 2,
         (roundup(D, kc) - D) // 2 + D % 2, (roundup(D, kc) - D) // 2)
    #print('padding ', a)
    x = F.pad(x, a)
    #print('padded shape ', x.shape)
    assert x.size(3) % kw == 0
    assert x.size(2) % kh == 0
    assert x.size(1) % kc == 0
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = list(patches.size())

    patches = patches.contiguous().view(-1, modalities, kc, kh, kw)

    ## TODO torch stack
    # with torch.no_grad():
    #     output = model.inference(patches)
    number_of_volumes = patches.shape[0]
    predictions = []

    for i in range(number_of_volumes):
        input_tensor = patches[i, ...].unsqueeze(0)
        predictions.append(model.inference(input_tensor))
    output = torch.stack(predictions, dim=0).squeeze(1).detach()
    # print(output.shape)
    N, Classes, _, _, _ = output.shape
    # Reshape backlist
    output_unfold_shape = unfold_shape[1:]
    output_unfold_shape.insert(0, Classes)
    # print(output_unfold_shape)
    output = output.view(output_unfold_shape)

    output_c = output_unfold_shape[1] * output_unfold_shape[4]
    output_h = output_unfold_shape[2] * output_unfold_shape[5]
    output_w = output_unfold_shape[3] * output_unfold_shape[6]
    output = output.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    output = output.view(-1, output_c, output_h, output_w)


    y = output[:, a[4]:output_c - a[5], a[2]:output_h - a[3], a[0]:output_w - a[1]]

    print(target.dtype,torch.randn(1,4,156,240,240).dtype)

    loss_dice, per_ch_score = criterion(y.unsqueeze(0).cuda(),target.cuda())
    print("INFERENCE DICE LOSS {} ".format(loss_dice.item()))
    return loss_dice


def visualize_3D_no_overlap_new(args, full_volume, affine, model, epoch, dim):
    """
    this function will produce NON-overlaping  sub-volumes prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param full_volume: t1, t2, segment
    :param dim: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    classes = args.classes
    modalities, slices, height, width = full_volume.shape
    full_volume_dim = (slices, height, width)

    print("full volume dim=", full_volume_dim, 'crop dim', dim)
    desired_dim = find_crop_dims(full_volume_dim, dim)
    print("Inference dims=", desired_dim)

    input_sub_volumes, segment_map = create_3d_subvol(full_volume, desired_dim)
    print(input_sub_volumes.shape, segment_map.shape)

    sub_volumes = input_sub_volumes.shape[0]
    predictions = []

    for i in range(sub_volumes):
        input_tensor = input_sub_volumes[i, ...].unsqueeze(0)
        predictions.append(model.inference(input_tensor))

    predictions = torch.stack(predictions)

    # project back to full volume
    full_vol_predictions = predictions.view(classes, slices, height, width)
    print("Inference complete", full_vol_predictions.shape)

    # arg max to get the labels in full 3d volume
    _, indices = full_vol_predictions.max(dim=0)
    full_vol_predictions = indices

    print("Class indexed prediction shape", full_vol_predictions.shape, "GT", segment_map.shape)

    # TODO TEST...................
    save_path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'
    create_2d_views(full_vol_predictions, segment_map, save_path_2d_fig)

    save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    save_3d_vol(full_vol_predictions.numpy(), affine, save_path)

# Todo test!
def create_3d_subvol(full_volume, dim):
    list_modalities = []

    modalities, slices, height, width = full_volume.shape

    full_vol_size = tuple((slices, height, width))
    dim = find_crop_dims(full_vol_size, dim)
    for i in range(modalities):
        TARGET_VOL = modalities - 1

        if i != TARGET_VOL:
            img_tensor = full_volume[i, ...]
            img = grid_sampler_sub_volume_reshape(img_tensor, dim)
            list_modalities.append(img)
        else:
            target = full_volume[i, ...]

    input_tensor = torch.stack(list_modalities, dim=1)

    return input_tensor, target


def grid_sampler_sub_volume_reshape(tensor, dim):
    return tensor.view(-1, dim[0], dim[1], dim[2])


# def find_crop_dims(full_size, mini_dim, adjust_dimension=2):
#     a, b, c = full_size
#     d, e, f = mini_dim

#     voxels = a * b * c
#     subvoxels = d * e * f

#     if voxels % subvoxels == 0:
#         return mini_dim

#     static_voxels = mini_dim[adjust_dimension - 1] * mini_dim[adjust_dimension - 2]
#     print(static_voxels)
#     if voxels % static_voxels == 0:
#         temp = int(voxels / static_voxels)
#         print("temp=", temp)
#         mini_dim_slice = mini_dim[adjust_dimension]
#         step = 1
#         while True:
#             slice_dim1 = temp % (mini_dim_slice - step)
#             slice_dim2 = temp % (mini_dim_slice + step)
#             if slice_dim1 == 0:
#                 slice_dim = int(mini_dim_slice - step)
#                 break
#             elif slice_dim2 == 0:
#                 slice_dim = int(temp / (mini_dim_slice + step))
#                 break
#             else:
#                 step += 1
#         return (d, e, slice_dim)

#     full_slice = full_size[adjust_dimension]

#     return tuple(desired_dim)

# Todo  
def save_3d_vol(predictions, affine, save_path):    
    pred_nifti_img = nib.Nifti1Image(predictions, affine)
    pred_nifti_img.header["qform_code"] = 1
    pred_nifti_img.header['sform_code'] = 0
    nib.save(pred_nifti_img, save_path + '.nii.gz')
    print('3D vol saved')
    # alternativly  pred_nifti_img.tofilename(str(save_path))


if __name__ == "__main__":
    base_dir = "/home/zongdaoming/cv/multi-organ/KiTS/kits19/data/case_00000"
    filename_1 = "imaging.nii.gz"
    filename_2 = "segmentation.nii.gz"
    image_file = os.path.join(base_dir, filename_1)
    label_file = os.path.join(base_dir, filename_2)
    image = loading_nii(image_file)
    label = loading_nii(label_file)
    print(image.shape)
    print(label.shape)
    for index in range(image.shape[-1]):
        # plot_sample(image[...,index], label[...,index],index)
        plot_sample(image[index], label[index],index)

    # pass

    