import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import kornia.geometry.conversions as tgm
import torch
import torchmetrics
import torchvision


def loss_L1(y_true, y_pred, ymin=1e-3, ymax=80):
    # mask out the invalid values (either zero or max_value)
    mask = (~torch.isnan(y_true)) * (y_true > ymin).float() * (y_true < ymax).float()
    num_pixels_total = mask.size(2) * mask.size(3)
    num_pixels_visible = torch.sum(mask, dim=(2, 3))

    # Cosine distance loss, normalize over the masks
    l_depth = torch.mean(torch.abs(mask * (y_pred - y_true)), dim=(2, 3))
    l_depth = (num_pixels_total / num_pixels_visible) * l_depth
    l_depth = torch.mean(l_depth, dim=(0, 1))

    return l_depth


# taken from https://medium.com/mlearning-ai/monocular-depth-estimation-using-u-net-6f149fc34077 and masked out
def loss_edge(y_true, y_pred, ymin=1e-3, ymax=80):
    # More strict mask, masks out everything that neighbors with an invalid value
    # For KITTI, this masks out basically everything
    mask = (~torch.isnan(y_true)) * (y_true > ymin).float() * (y_true < ymax).float()
    device = y_true.get_device()
    zx = torch.zeros((mask.size(0), mask.size(1), mask.size(2), 1)).to(device)
    zy = torch.zeros((mask.size(0), mask.size(1), 1, mask.size(3))).to(device)
    mask_shift_x = third_tensor = torch.cat((mask[:, :, :, 1:], zx), 3)
    mask_shift_y = third_tensor = torch.cat((mask[:, :, 1:, :], zy), 2)
    mask_x = mask * mask_shift_x
    mask_y = mask * mask_shift_y

    # Edge loss for sharp edges
    dy_true, dx_true = torchmetrics.functional.image_gradients(y_true)
    dy_pred, dx_pred = torchmetrics.functional.image_gradients(y_pred)
    l_edges = torch.mean(
        mask_y * torch.abs(dy_pred - dy_true) + mask_x * torch.abs(dx_pred - dx_true),
        dim=(0, 1, 2, 3),
    )

    return l_edges


# According to https://en.wikipedia.org/wiki/Structural_similarity
# Taken from https://medium.com/mlearning-ai/monocular-depth-estimation-using-u-net-6f149fc34077 and masked out
def loss_ssim(y_true, y_pred, ymin=1e-3, ymax=80):
    mask = (~torch.isnan(y_true)) * (y_true > ymin).float() * (y_true < ymax).float()
    num_pixels_total = mask.size(2) * mask.size(3)
    num_pixels_visible = torch.sum(mask, dim=(2, 3))

    gauss = torchvision.transforms.GaussianBlur(11, sigma=1.5)
    weight = gauss(mask) + 1e-15
    mu_x = gauss(mask * y_pred) / weight
    mu_y = gauss(mask * y_true) / weight
    sgm_x = gauss(mask * y_pred * y_pred) / weight - mu_x * mu_x
    sgm_y = gauss(mask * y_true * y_true) / weight - mu_y * mu_y
    sgm_xy = gauss(mask * y_pred * y_true) / weight - mu_x * mu_y
    ssim = (
        (2 * mu_x * mu_y + 0.01)
        * (2 * sgm_xy + 0.03)
        / ((mu_x * mu_x + mu_y * mu_y + 0.01) * (sgm_x + sgm_y + 0.03))
    )
    ssim = mask * ssim
    avg_ssim = torch.mean(ssim, dim=(2, 3))
    avg_ssim = (num_pixels_total / num_pixels_visible) * avg_ssim
    avg_ssim = torch.mean(avg_ssim, dim=(0, 1))
    l_ssim = torch.clip((1 - avg_ssim) * 0.5, 0, 1)

    return l_ssim


# Linie są tutaj.
def loss_line_distance_field(df_true, df_pred):
    l_df = torch.mean(torch.abs(df_pred - df_true))

    return l_df


# The colinearity loss
# Linie są głównie tutaj.
def loss_lines(y_pred, ls, Ki):
    device = y_pred.device
    l_col_L2 = torch.tensor(0.0).to(device)
    l_col_L1 = torch.tensor(0.0).to(device)

    # Go over all images
    for i in range(len(ls)):
        lines = ls[i]

        if len(lines) == 0:
            continue

        if not isinstance(lines, torch.Tensor):
            lines = torch.tensor(lines).to(device)

        # Get the orthonormal basis of the backprojected plane
        direction = lines[:, 1] - lines[:, 0]
        direction = tgm.convert_points_to_homogeneous(direction)
        direction[:, 2] = 0
        b1 = torch.matmul(Ki[i], direction.transpose(0, 1).double()).transpose(0, 1)
        b1 = torch.nn.functional.normalize(b1)

        direction2 = tgm.convert_points_to_homogeneous(lines[:, 0])
        b2 = torch.matmul(Ki[i], direction2.transpose(0, 1).double()).transpose(0, 1)
        b2 = b2 - torch.sum(b1 * b2, dim=1).unsqueeze(1) * b1
        b2 = torch.nn.functional.normalize(b2)
        # print(torch.sum(b1*b2, dim = 1).unsqueeze(1))

        # print("Dir")
        # print(b1)
        # print(b2)

        # Now, the point goes for the walk.
        num_pts = 21  # TODO increase this number
        a = torch.tensor(list(range(num_pts))).to(device).unsqueeze(1).unsqueeze(0) / (
            num_pts - 1
        )
        b = 1 - a

        # Get points on the line
        pts = a * lines[:, 0].unsqueeze(1) + b * lines[:, 1].unsqueeze(1)

        # Round the point to get integer coordinates of 4 closest pixels (will be used for calculating depth)
        pts0 = torch.floor(pts).long()
        x0s = pts0[:, :, 0]
        y0s = pts0[:, :, 1]
        x0s[x0s < 0] = 0
        h, w = y_pred.shape[2], y_pred.shape[3]
        x0s[x0s > w - 1] = w - 1
        y0s[y0s < 0] = 0
        y0s[y0s > h - 1] = h - 1

        x1s = x0s + 1
        x1s[x1s > w - 1] = w - 1
        y1s = y0s + 1
        y1s[y1s > h - 1] = h - 1

        cxs = pts[:, :, 0] - pts0[:, :, 0]
        cys = pts[:, :, 1] - pts0[:, :, 1]

        # Get normalized rays of each of the points on the line
        pt_hom = tgm.convert_points_to_homogeneous(pts)
        pt_perm = pt_hom.unsqueeze(3).double()
        rays = torch.matmul(Ki[i], pt_perm).squeeze(3)
        rays = torch.nn.functional.normalize(rays, dim=2)

        # print(Ki[i])

        # Get the depth by bilinear interpolation
        depths = (1 - cys) * (
            (1 - cxs) * y_pred[0, 0, y0s, x0s] + cxs * y_pred[0, 0, y0s, x1s]
        ) + cys * ((1 - cxs) * y_pred[0, 0, y1s, x0s] + cxs * y_pred[0, 0, y1s, x1s])

        # Get the 3D points by multiplying the rays with the depth
        X3 = depths.unsqueeze(2) * rays

        # Get the 2D coordinates of the 3D points wrt the orthogonal system
        c1 = torch.sum(b1.unsqueeze(1) * X3, dim=2)
        c2 = torch.sum(b2.unsqueeze(1) * X3, dim=2)

        # Fit the line into the points with PCA
        m1 = torch.mean(c1, dim=1).unsqueeze(1)
        c1 = c1 - m1
        m2 = torch.mean(c2, dim=1).unsqueeze(1)
        c2 = c2 - m2

        c1Tc1 = torch.sum(c1 * c1, dim=1)
        c1Tc2 = torch.sum(c1 * c2, dim=1)
        c2Tc2 = torch.sum(c2 * c2, dim=1)

        # Calculate the small eigenvalue of the "PCA matrix"
        T = c1Tc1 + c2Tc2
        D = c1Tc1 * c2Tc2 - c1Tc2 * c1Tc2
        L2 = T / 2 - (T * T / 4 - D) ** 0.5
        l_col_L2 += torch.mean(torch.abs(L2))

        # Calculate the distances from the line eigenvector corresponding to the large eigenvalue of the "PCA matrix"
        # The normal equation of the line corresponding to the large eigenvector is the small eigenvector
        # The distance of a point (x,y) to a line defined by equation ax+by=0 is abs(a*x+b*y)/sqrt(a**2 + b**2)
        param1 = (c1Tc2 + L2 - c2Tc2).unsqueeze(1)
        param2 = (L2 - c1Tc1 + c1Tc2).unsqueeze(1)
        diff = (param1 * c1 + param2 * c2) / ((param1**2 + param2**2) ** 0.5)
        l_col_L1 += torch.mean(torch.abs(diff))

    return l_col_L2, l_col_L1


# DeepLSD reprojection loss
# Linie są głównie tutaj.
def loss_deeplsd_score(y_pred, df, df2, ls, Ki, Q, max_depth):
    device = df2.device
    l_df2 = torch.tensor(0.0).to(device)
    l_col_L2_rep = torch.tensor(0.0).to(device)
    l_col_L1_rep = torch.tensor(0.0).to(device)

    # Go over all images
    for i in range(len(ls)):
        lines = ls[i]

        if len(lines) == 0:
            continue

        if not isinstance(lines, torch.Tensor):
            lines = torch.tensor(lines).to(device)

        # Get the orthonormal basis of the backprojected plane
        direction = lines[:, 1] - lines[:, 0]
        direction = tgm.convert_points_to_homogeneous(direction)
        direction[:, 2] = 0
        b1 = torch.matmul(Ki[i], direction.transpose(0, 1).double()).transpose(0, 1)
        b1 = torch.nn.functional.normalize(b1)

        direction2 = tgm.convert_points_to_homogeneous(lines[:, 0])
        b2 = torch.matmul(Ki[i], direction2.transpose(0, 1).double()).transpose(0, 1)
        b2 = b2 - torch.sum(b1 * b2, dim=1).unsqueeze(1) * b1
        b2 = torch.nn.functional.normalize(b2)

        # Now, the point goes for the walk.
        num_pts = 21  # TODO increase this number
        a = torch.tensor(list(range(num_pts))).to(device).unsqueeze(1).unsqueeze(0) / (
            num_pts - 1
        )
        b = 1 - a

        # Get points on the line
        pts = a * lines[:, 0].unsqueeze(1) + b * lines[:, 1].unsqueeze(1)

        # Round the point to get integer coordinates of 4 closest pixels (will be used for calculating depth)
        pts0 = torch.floor(pts).long()
        x0s = pts0[:, :, 0]
        y0s = pts0[:, :, 1]
        x0s[x0s < 0] = 0
        h, w = y_pred.shape[2], y_pred.shape[3]
        x0s[x0s > w - 1] = w - 1
        y0s[y0s < 0] = 0
        y0s[y0s > h - 1] = h - 1

        x1s = x0s + 1
        x1s[x1s > w - 1] = w - 1
        y1s = y0s + 1
        y1s[y1s > h - 1] = h - 1

        cxs = pts[:, :, 0] - pts0[:, :, 0]
        cys = pts[:, :, 1] - pts0[:, :, 1]

        # Get normalized rays of each of the points on the line
        pt_hom = tgm.convert_points_to_homogeneous(pts)
        pt_perm = pt_hom.unsqueeze(3).double()
        rays = torch.matmul(Ki[i], pt_perm).squeeze(3)
        rays = torch.nn.functional.normalize(rays, dim=2)

        # Get the depth by bilinear interpolation
        depths = (1 - cys) * (
            (1 - cxs) * y_pred[0, 0, y0s, x0s] + cxs * y_pred[0, 0, y0s, x1s]
        ) + cys * ((1 - cxs) * y_pred[0, 0, y1s, x0s] + cxs * y_pred[0, 0, y1s, x1s])

        # Get the 3D points by multiplying the rays with the depth
        X3 = depths.unsqueeze(2) * rays

        # transform the 3D points to the actual world units
        X3R = max_depth * X3

        # transform the 3D points to the second camera and project them
        X3R = tgm.convert_points_to_homogeneous(X3R)
        QX3 = torch.matmul(Q[i], X3R.unsqueeze(3).to(device)).squeeze(3)
        QX3 = tgm.convert_points_from_homogeneous(QX3)
        QX2 = tgm.convert_points_from_homogeneous(QX3)

        # convert the coordinates to a interpolation-friendly way
        QX20 = torch.floor(QX2).long()
        qx0s = QX20[:, :, 0]
        qy0s = QX20[:, :, 1]

        qcxs = QX2[:, :, 0] - QX20[:, :, 0]
        qcxs[qx0s < 0] = 0.5
        qcxs[qx0s > w - 1] = 0.5
        qcys = QX2[:, :, 1] - QX20[:, :, 1]
        qcys[qy0s < 0] = 0.5
        qcys[qy0s > h - 1] = 0.5

        qx0s[qx0s < 0] = 0
        qx0s[qx0s > w - 1] = w - 1
        qy0s[qy0s < 0] = 0
        qy0s[qy0s > h - 1] = h - 1

        qx1s = qx0s + 1
        qx1s[qx1s > w - 1] = w - 1
        qy1s = qy0s + 1
        qy1s[qy1s > h - 1] = h - 1

        # TODO try adding more loss when we leave the view => this will push the points into the view
        # TODO try blurring the df2 view => we will be able to reach lines that are further than without the blur
        # TODO TWO OPTIONS: a) minimize the right score, b) minimize the L1 difference between left and right scores
        right_scores = (1 - qcys) * (
            (1 - qcxs) * df2[0, 0, qy0s, qx0s] + qcxs * df2[0, 0, qy0s, qx1s]
        ) + qcys * ((1 - qcxs) * df2[0, 0, qy1s, qx0s] + qcxs * df2[0, 0, qy1s, qx1s])
        left_scores = (1 - cys) * (
            (1 - cxs) * df[0, 0, y0s, x0s] + cxs * df[0, 0, y0s, x1s]
        ) + cys * ((1 - cxs) * df[0, 0, y1s, x0s] + cxs * df[0, 0, y1s, x1s])
        l_df2 += torch.mean(torch.abs(right_scores - left_scores))
        # l_df2 += torch.mean(right_scores)

        # Measure the colinearity loss of the points projected to the second camera
        c1 = QX2[:, :, 0]
        c2 = QX2[:, :, 1]
        # c1 = pts[:,:,0]
        # c2 = pts[:,:,1]

        # Fit the line into the points with PCA
        m1 = torch.mean(c1, dim=1).unsqueeze(1)
        c1 = c1 - m1
        m2 = torch.mean(c2, dim=1).unsqueeze(1)
        c2 = c2 - m2

        c1Tc1 = torch.sum(c1 * c1, dim=1)
        c1Tc2 = torch.sum(c1 * c2, dim=1)
        c2Tc2 = torch.sum(c2 * c2, dim=1)

        # Calculate the small eigenvalue of the "PCA matrix"
        T = c1Tc1 + c2Tc2
        D = c1Tc1 * c2Tc2 - c1Tc2 * c1Tc2
        L2 = T / 2 - (T * T / 4 - D) ** 0.5
        l_col_L2_rep += torch.mean(torch.abs(L2))

        # Calculate the distances from the line eigenvector corresponding to the large eigenvalue of the "PCA matrix"
        # The normal equation of the line corresponding to the large eigenvector is the small eigenvector
        # The distance of a point (x,y) to a line defined by equation ax+by=0 is abs(a*x+b*y)/sqrt(a**2 + b**2)
        param1 = (c1Tc2 + L2 - c2Tc2).unsqueeze(1)
        param2 = (L2 - c1Tc1 + c1Tc2).unsqueeze(1)
        diff = (param1 * c1 + param2 * c2) / ((param1**2 + param2**2) ** 0.5)
        l_col_L1_rep += torch.mean(torch.abs(diff))

    return l_df2, l_col_L2_rep, l_col_L1_rep


# Loss
# Linie są tutaj.
def loss_function(
    y_true,
    y_pred,
    ls,
    Ki,
    args,
    df_true=None,
    df_pred=None,
    df1=None,
    df2=None,
    Q=None,
    do_ssl=False,
    include_df_rec_loss=True,
    include_df_proj_loss=True,
):
    if not (y_true.max() > 1 and y_pred.max() > 1):
        print("WARN! y_true.max() = ", y_true.max(), "y_pred.max() = ", y_pred.max())
    # Device
    device = y_pred.get_device()

    # 3. L1 loss for the line distance field (generated by DeepLSD) in the second head.
    # Linie są tutaj.
    if include_df_rec_loss:
        assert df_true is not None and df_pred is not None
        l_df = loss_line_distance_field(df_true, df_pred)
    else:
        l_df = 0

    # Define squeezed intrinsics matrices, it may actually work better
    # TODO also try changing this based on the epoch: first start with proper K, then deteriorate to I (K0), and to st. like Ki = [1000 0 0;0 1000 0;0 0 1] (i.e. the complete opposite)
    """
    K0 = torch.eye(3).unsqueeze(0).repeat(4,1,1).double().to(device)
    K1c = torch.eye(3)
    K1c[0,0] = 1000000
    K1c[1,1] = 1000000
    K1 = torch.matmul(K1c.unsqueeze(0).double().to(device), Ki)
    K2c = torch.eye(3)
    K2c[2,2] = 0
    K2 = torch.matmul(K2c.unsqueeze(0).double().to(device), Ki)
    """

    # 4. L2 colinearity loss and 5. L1 colinearity loss
    # Linie są głównie tutaj.
    if args.calibration == 0:
        l_col_L2, l_col_L1 = loss_lines(y_pred, ls, Ki)
        l_col_L2 = 7.88 * l_col_L2
        l_col_L1 = 1.86 * l_col_L1
        # l_col_L2, l_col_L1: colinearity loss in 3D
    elif args.calibration == 1:
        K0 = torch.eye(3).unsqueeze(0).repeat(4, 1, 1).double().to(device)
        l_col_L2, l_col_L1 = loss_lines(y_pred, ls, K0)
        l_col_L2 = 5.60 * l_col_L2
        l_col_L1 = 1.77 * l_col_L1
    elif args.calibration == 2:
        K1c = torch.eye(3)
        K1c[0, 0] = 1000000
        K1c[1, 1] = 1000000
        K1 = torch.matmul(K1c.unsqueeze(0).double().to(device), Ki)
        l_col_L2, l_col_L1 = loss_lines(y_pred, ls, K1)
    else:
        l_col_L2, l_col_L1 = loss_lines(y_pred, ls, Ki)
        l_col_L2 = 7.88 * l_col_L2
        l_col_L1 = 1.86 * l_col_L1

    # Weightage
    # w1, w2, w3, w4, w5, w6, w7, w8 = 0.1, 1.0, 0.1, 0.1, 0.1, 0.001, 0.1, 0.1
    w1 = args.w1
    w2 = args.w2
    w3 = args.w3
    w4 = args.w4
    w5 = args.w5
    w6 = args.w6
    w7 = args.w7
    w8 = args.w8
    modelip_loss = (w3 * l_df) + (w4 * l_col_L2) + (w5 * l_col_L1)

    res = {
        "l_df": w3 * l_df,
        "l_col_L1": w4 * l_col_L1,
        "l_col_L2": w5 * l_col_L2,
        "modelip_loss": modelip_loss,
    }

    if include_df_proj_loss:
        assert df1 is not None and df2 is not None and Q is not None
        if not (df1.max() > 1 and df2.max() > 1):
            print("WARN! df2.max() = ", df2.max(), "df1.max() = ", df1.max())
        # 6. Loss by reprojecting to right view and measuring DeepLSD score
        # Linie są głównie tutaj.
        l_df2, l_col_L2_rep, l_col_L1_rep = loss_deeplsd_score(
            y_pred, df1, df2, ls, Ki, Q, args.max_depth
        )
        # l_df2: loss from the DeepLSD score after reprojecting the points to 2nd view
        # l_col_L2_rep, l_col_L1_rep: colinearity loss after reprojecting to 2nd view
        # reweight the losses to have a similar magnitude to the 3D loss
        l_col_L2_rep = 0.00002 * l_col_L2_rep
        l_col_L1_rep = 0.004 * l_col_L1_rep
        res["l_df2"] = w6 * l_df2
        res["l_col_L1_rep"] = w7 * l_col_L1_rep
        res["l_col_L2_rep"] = w8 * l_col_L2_rep
        res["modelip_loss"] += (
            (w6 * res["l_df2"])
            + (w7 * res["l_col_L2_rep"])
            + (w8 * res["l_col_L1_rep"])
        )

    if not do_ssl:
        # supervised losses
        # 1. Masked L1 depth loss
        # No lines.
        l_depth = loss_L1(y_true, y_pred)

        # 2. Structural similarity loss
        # According to https://en.wikipedia.org/wiki/Structural_similarity
        # No lines.
        l_ssim = loss_ssim(y_true, y_pred)
        res["l_depth"] = w1 * l_depth
        res["l_ssim"] = w2 * l_ssim
        res["modelip_loss"] += (w1 * res["l_depth"]) + (w2 * res["l_ssim"])
    return res
