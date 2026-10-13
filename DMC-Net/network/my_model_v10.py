import torch
import torch.nn as nn
import torch.nn.functional as F
from .modulesv2 import CB, CDB, ST, UNet, DUC


class MaskPriorModule(nn.Module):
    """掩码跨模态先验的近端算子 W = S_{σ_w, ω}(P + Λ_w/σ_w, Y)

    结构对齐 CrossModalRegularization：
    1) 输入分支投影
    2) CDB 跨模态交互
    3) ST 正则子网络提取先验梯度
    4) 近端步更新并映射到 [0, 1]
    """
    def __init__(self, y_dim, control_nums, feature_dim=24):
        super(MaskPriorModule, self).__init__()

        self.p_proj = nn.Conv2d(1, feature_dim, kernel_size=3, padding=1)
        self.y_proj = nn.Conv2d(y_dim, feature_dim, kernel_size=1, padding="same")

        self.CDB = CDB(feature_dim)
        self.s_omega = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 3, kernel_size=3, padding=1),
            ST(control_nums),
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=3, padding=1)
        )
        self.head = nn.Conv2d(feature_dim, 1, kernel_size=1, padding="same")
        self.register_buffer("eta", torch.tensor(1.0))

    def forward(self, p_input, y):
        """
        Args:
            p_input: P + Λ_w/σ_w  [B, 1, H, W]
            y: RGB guidance [B, y_dim, H, W]
        Returns:
            W: 近端算子输出 [B, 1, H, W]
        """
        p_feat = self.p_proj(p_input)
        y_feat = self.y_proj(y)

        cross_modal_prior = self.s_omega(self.CDB(p_feat, y_feat))
        prior_step = self.head(cross_modal_prior)

        W = torch.sigmoid(p_input - 0.1 * self.eta * prior_step)
        return W
        

class MaskAwareRegularization(nn.Module):
    """文中掩码感知辅助变量 A 的更新。

    A = S_{sigma1, phi1, phi2}(U + Lambda1 / sigma1)。
    phi1/phi2 的网络实现保持不变；同时在更新后的 A 和旧掩码 M 上
    计算 G1、G2，供随后的 M 闭式更新使用。
    """
    def __init__(self, in_dim, control_nums):
        super(MaskAwareRegularization, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, padding=1),
            ST(control_nums),
            nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, padding=1)
        )
        self.phi2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, padding=1),
            ST(control_nums),
            nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, padding=1)
        )
        self.register_buffer("beta1", torch.tensor(0.1))
        self.register_buffer("beta2", torch.tensor(0.5))
        
    def forward(self, u, M, lambda1, sigma1):
        """
        Returns:
            A_update: 更新后的辅助变量 A
            G1: phi1'(A_update * M) 的可学习近似
            G2: phi2'(A_update * (1-M)) 的可学习近似
        """
        A_target = u + lambda1 / sigma1

        phi1_step = self.phi1(A_target * M)
        phi2_step = self.phi2(A_target * (1 - M))
        regularization = (
            M * self.beta1 * phi1_step
            + (1 - M) * self.beta2 * phi2_step
        )

        A_update = A_target - 0.1 * regularization

        G1 = self.phi1(A_update * M)
        G2 = self.phi2(A_update * (1 - M))
        return A_update, G1, G2


class CrossModalRegularization(nn.Module):
    """文中跨模态辅助变量 B 的更新。
    B^{(k+1)} = Prox_{(gamma/sigma2) psi}(U^{(k)} + Lambda2^{(k)}/sigma2)
    """
    def __init__(self, in_dim, y_dim, control_nums):
        super(CrossModalRegularization, self).__init__()
        self.s_psi = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 3, kernel_size=3, padding=1),
            ST(control_nums),
            nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, padding=1)
        )
        self.conv = nn.Conv2d(y_dim, in_dim, kernel_size=1, padding="same")
        self.CDB = CDB(in_dim)
        self.register_buffer("gamma", torch.tensor(1.0))
        
    def forward(self, u, y, lambda2, sigma2):
        """
        B^{(k+1)} = Prox_{(gamma/sigma2) psi}(U^{(k)} + Lambda2^{(k)}/sigma2)
        """
        B_target = u + lambda2 / sigma2
        
        cross_modal_reg = self.s_psi(self.CDB(B_target, self.conv(y)))
        
        B_update = B_target - 0.1 * self.gamma * cross_modal_reg
        return B_update, cross_modal_reg


class MainVariableUpdate(nn.Module):
    """主变量 U 更新 (整合数据保真项)
    
    求解 Sylvester 方程：Q_L U + U Q_R = F
    其中：
    - Q_L = (1/σ²) D^T diag(M²) D + (σ₁ + σ₂) I
    - Q_R = α H H^T
    - F = (1/σ²) D^T diag(M²) Z + α H^T(Y) + σ₁ Ã + σ₂ B̃
    """
    def __init__(self, in_dim, y_dim, feature_dim, ratio_spatial, ratio_fusion, 
                 middle_blk_num, enc_blk_nums, dec_blk_nums):
        super(MainVariableUpdate, self).__init__()
        
        self.matrix_D = CB(in_dim, ratio_spatial)   # 下采样算子
        self.matrix_Dt = CB(in_dim, ratio_spatial)  # 上采样算子（D的转置）
        
        self.matrix_H = nn.Sequential(
            nn.Conv2d(in_dim, y_dim, kernel_size=1, bias=False, padding="same"),
            CB(y_dim, ratio_spatial),
            CB(y_dim, ratio_spatial)
        )
        self.matrix_Ht = nn.Sequential(
            nn.Conv2d(y_dim, in_dim, kernel_size=1, bias=False, padding="same"),
            CB(in_dim, ratio_spatial),
            CB(in_dim, ratio_spatial)
        )
        
        self.register_buffer("sigma_sq", torch.tensor(1.0))  # 噪声方差 σ²
        self.register_buffer("alpha", torch.tensor(1.0))      # 跨模态一致性权重
        self.register_buffer("sigma1", torch.tensor(1.0))     # U-A约束惩罚系数
        self.register_buffer("sigma2", torch.tensor(1.0))     # U-B约束惩罚系数
        
        self.solver = UNet(in_dim, feature_dim, ratio_spatial=1, 
                          ratio_fusion=ratio_fusion,
                          middle_blk_num=middle_blk_num,
                          enc_blk_nums=enc_blk_nums, 
                          dec_blk_nums=dec_blk_nums)
        self.updater = DUC(in_dim)
        
    def forward(self, u, z, y, M, A, B, lambda1, lambda2):
        """
        U^{(k+1)} = argmin_U (1/2σ²)||M⊙(DU-Z)||² + (α/2)||UH-Y||² 
                           + (sigma1/2)||U-A_tilde||² + (sigma2/2)||U-B_tilde||²
        
        其中 A_tilde = A - Lambda1/sigma1, B_tilde = B - Lambda2/sigma2
        
        
        连续掩码的保真项梯度使用 M^2。
        """
        A_tilde = A - lambda1 / self.sigma1
        B_tilde = B - lambda2 / self.sigma2
        M_sq = M.square()
        
        data_term_rhs = self.matrix_Dt(M_sq * z) / self.sigma_sq
        
        cross_modal_rhs = self.alpha * self.matrix_Ht(y)
        
        constraint_rhs = self.sigma1 * A_tilde + self.sigma2 * B_tilde
        
        rhs = data_term_rhs + cross_modal_rhs + constraint_rhs
        
        D_u = self.matrix_D(u)
        data_term_lhs = self.matrix_Dt(M_sq * D_u) / self.sigma_sq
        
        cross_modal_lhs = self.alpha * self.matrix_Ht(self.matrix_H(u))
        
        constraint_lhs = (self.sigma1 + self.sigma2) * u
        
        lhs = data_term_lhs + cross_modal_lhs + constraint_lhs
        
        residual = lhs - rhs
        
        delta_refined = self.solver(residual)
        
        u_new = u-0.1*delta_refined
        return u_new, residual


class MaskClosedFormUpdate(nn.Module):
    """按 Method Eq. (M_closed) 更新动态掩码 M。

    闭式公式:
        M_new = [sigma3*W - Lambda3 + lambda1*M0
                 - beta1*A*G1 + beta2*A*G2]
                / [(D(U)-Z)^2 + lambda1 + sigma3]

    D(U) 与 U 子问题共享同一个 D；G1/G2 由 A 子问题现有的 phi 网络计算。
    """
    def __init__(self, in_dim, y_dim, control_nums, feature_dim=24, ratio_spatial=4):
        super(MaskClosedFormUpdate, self).__init__()

        self.mask_prior = MaskPriorModule(y_dim, control_nums, feature_dim)

        self.lambda1_m = nn.Parameter(torch.tensor(1.0))  # λ₁: 掩码保真权重
        self.sigma3 = nn.Parameter(torch.tensor(1.0))     # σ₃: M-W 约束罚参数
        self.rho3 = nn.Parameter(torch.tensor(0.1))       # ρ₃: Λ₃ 对偶更新步长

    def update_w(self, M, y, lambda3):
        """先执行文中的 W = S(M + Lambda3/sigma3, Y)。"""
        sigma3 = F.softplus(self.sigma3) + 1e-4
        W = self.mask_prior(M + lambda3 / sigma3, y)
        return W, sigma3

    def forward(self, D_u, z, M0, A, G1, G2, W, lambda3,
                beta1, beta2, sigma3):
        """
        Args:
            D_u: 使用 U 子问题同一退化算子得到的 D(U)
            z: 观测深度图 [B, C, H, W]
            M0: 初始掩码 [B, 1, H, W]
            A, G1, G2: 先行更新的 A 以及旧掩码处的 phi 导数近似
            lambda3: M-W 约束的对偶变量 [B, 1, H, W]
        Returns:
            M_new: 更新后的掩码 [B, 1, H, W]
        """
        lambda1 = F.softplus(self.lambda1_m) + 1e-4
        R = D_u - z
        R_sq = (R ** 2).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        regularizer_term = (-beta1 * A * G1 + beta2 * A * G2).mean(
            dim=1, keepdim=True
        )
        numerator = sigma3 * W - lambda3 + lambda1 * M0 + regularizer_term
        M_new = numerator / (R_sq + lambda1 + sigma3)
        return torch.clamp(M_new, 0.0, 1.0)


class ADMM_Stage(nn.Module):
    """单阶段迭代：依次更新 A、B、W、M、U 和三个乘子。"""
    def __init__(self, in_dim, y_dim, feature_dim, ratio_spatial, ratio_fusion, 
                 control_nums, middle_blk_num, enc_blk_nums, dec_blk_nums, training=True):
        super(ADMM_Stage, self).__init__()
        self.training = training

        self.mask_update = MaskClosedFormUpdate(in_dim, y_dim, control_nums, feature_dim, ratio_spatial)

        self.mask_aware_reg = MaskAwareRegularization(in_dim, control_nums)

        self.cross_modal_reg = CrossModalRegularization(in_dim, y_dim, control_nums)

        self.main_update = MainVariableUpdate(in_dim, y_dim, feature_dim, 
                                              ratio_spatial, ratio_fusion,
                                              middle_blk_num, enc_blk_nums, dec_blk_nums)

    def forward(self, u, z, y, M0, M, lambda1, lambda2, lambda3):
        """
        执行一次ADMM迭代 (统一优化框架)

        Args:
            u: 当前主变量估计 [B, C, H, W]
            z: 观测深度图 [B, C, H, W]
            y: RGB引导图像 [B, C2, H, W]
            M0: 初始有效像素掩码 [B, 1, H, W]
            M: 当前动态掩码 [B, 1, H, W]
            lambda1: U-A 约束的拉格朗日乘子
            lambda2: U-B 约束的拉格朗日乘子
            lambda3: M-W 约束的拉格朗日乘子
        """
        sigma1 = self.main_update.sigma1
        sigma2 = self.main_update.sigma2

        A, G1, G2 = self.mask_aware_reg(u, M, lambda1, sigma1)
        B, _ = self.cross_modal_reg(u, y, lambda2, sigma2)
        W, sigma3 = self.mask_update.update_w(M, y, lambda3)

        D_u = self.main_update.matrix_D(u)
        M_new = self.mask_update(
            D_u, z, M0, A, G1, G2, W, lambda3,
            self.mask_aware_reg.beta1, self.mask_aware_reg.beta2, sigma3
        )

        u_new, u_residual = self.main_update(
            u, z, y, M_new, A, B, lambda1, lambda2
        )

        lambda1_new = lambda1 + 0.1 * (u_new - A)
        lambda2_new = lambda2 + 0.1 * (u_new - B)
        lambda3_new = lambda3 + F.softplus(self.mask_update.rho3) * (M_new - W)

        if self.training:
            fu_loss = F.l1_loss(
                u_residual,
                torch.zeros_like(u_residual),
                reduction='mean'
            )
        else:
            fu_loss = torch.tensor(0.0).to(u.device)

        return u_new, M_new, lambda1_new, lambda2_new, lambda3_new, fu_loss


class UnifiedDepthNet(nn.Module):
    """统一的深度图补全网络"""
    def __init__(self, args, training=True):
        super(UnifiedDepthNet, self).__init__()
        self.training = training
        
        self.depth = args["depth"]
        self.in_dim = args["in_dim"]
        self.y_dim = args["y_dim"]
        self.feature_dim = args["feature_dim"]
        self.ratio_spatial = args["ratio_spatial"]
        self.ratio_fusion = args["ratio_fusion"]
        self.control_nums = args["control_nums"]
        self.middle_blk_num = args["middle_blk_num"]
        self.enc_blk_nums = args["enc_blk_nums"]
        self.dec_blk_nums = args["dec_blk_nums"]
        
        self.admm_stages = nn.ModuleList([
            ADMM_Stage(self.in_dim, self.y_dim, self.feature_dim,
                      self.ratio_spatial, self.ratio_fusion,
                      self.control_nums, self.middle_blk_num,
                      self.enc_blk_nums, self.dec_blk_nums,
                      self.training)
            for _ in range(self.depth)
        ])
        
        self.init_lambda = nn.Conv2d(self.in_dim, self.in_dim * 2, kernel_size=3, padding=1)
        self.init_lambda3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def forward(self, z, y):
        """
        输入:
            z: 有空洞的深度图 [B, C1, h, w]
            y: RGB引导图像 [B, C2, H, W]
        输出:
            u: 完整深度图 [B, C1, H, W]
            mask_loss: 掩码预测损失 (在这个版本中不再使用，或者作为隐式约束)
            fu_losses: ADMM迭代损失列表
        """
        B, _, h, w = z.shape
        _, _, H, W = y.shape


        M = (z > 0.001) .float()   #NYU
        u = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)
        
        M0 = F.interpolate(M, size=(H, W), mode='nearest')    # 初始掩码 M₀
        z_hr = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=False)
        M = M0.clone()  # 动态掩码初始化为 M₀

        lambda_init = self.init_lambda(u)
        lambda1, lambda2 = lambda_init.chunk(2, dim=1)
        lambda3 = self.init_lambda3(M)
        
        fu_losses = []
        for stage in self.admm_stages:
            u, M, lambda1, lambda2, lambda3, fu_loss = stage(
                u, z_hr, y, M0, M, lambda1, lambda2, lambda3
            )
            fu_losses.append(fu_loss)
        
        mask_loss = torch.tensor(0.0).to(u.device)
        
        return u, mask_loss, torch.stack(fu_losses)
