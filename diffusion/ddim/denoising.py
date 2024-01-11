import torch

def compute_alpha(beta, t):       # 这里DDIM中叫\alpha，相当于是DDPM中的\bar\alpha，我们的注释里统一叫\bar\alpha
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def ddpm_steps(x, seq, model, b, **kwargs):  # ddpm seq: [0, 1, 2, 3, ..., T]
    # 标准 DDPM 采样，（除了最后一步）每步带噪声，先预测出当前步 x_0，在计算出 x_{t-1}
    # 第一步：根据 x_t 和 UNet 预测出的噪声 \epsilon = \epsilon_\theta(x_t,t) 估计当前步的 x_0: \hat{x}_{0|t}
    # 第二步：计算出 x_{t-1}: x_{t-1}=\mu(x_t,\hat{x}_{0|t})+\sigma_t * \epsilon
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])   # seq_next: [-1, 0, 1, 2, ..., T-1]
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)    # t
            next_t = (torch.ones(n) * j).to(x.device)  # t - 1
            at = compute_alpha(betas, t.long())   # at = \bar\alpha_t
            atm1 = compute_alpha(betas, next_t.long())  # atm1 = \bar\alpha_{t-1}   # 我猜 m1 是 minus 1 的意思
            beta_t = 1 - at / atm1  # \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}
            x = xs[-1].to('cuda')  # lastest xs, i.e. x_t

            output = model(x, t.float()) # e = \epsilon = \epsilon_\theta(x_t, t) e是模型预测的噪声
            e = output  

            # 第一步
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e  # predicted x_0 = \sqrt{\frac{1}{\bar\alpha_t}} * x_t - \sqrt{\frac{1}{\bar\alpha_t - 1}} * \epsilon  # 与参考推导中稍微变换一下即可
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            
            # 第二步
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)   # \mu = (\sqrt{\bar\alpha_{t-1} * \beta_t}) * \hat{x}_{0|t} + \frac{\sqrt{1-\beta_t} * (1 - \bar\alpha_{t-1}) * x_t}{1 - \bar\alpha_t}  # 与参考推导中稍微变换一下即可

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()  # 只有在最后一步不加噪声
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds


def generalized_steps(x, seq, model, b, **kwargs):
    # 通用DDIM采样
    # DDPM可以看做是DDIM的一种特殊形式，即每一步都没跳，而更一般的DDIM可以任意跳步
    # 记seq中跳k步, 实际中seq每个元素跳的步数不一定是均匀的 skip_type = uniform / quad
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())  # \bar\alpha_t
            at_next = compute_alpha(b, next_t.long())  # \bar\alpha_{t-k}  # 这里就不一定是走一步了，而是走k步，所以不再叫atm1，而是叫at_next
            xt = xs[-1].to('cuda')
            et = model(xt, t)  # epsilon
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()   # \hat{x}_{0|t} = \frac{x_t - \epsilon * \sqrt{1 - \bar\alpha_t}}{\sqrt{\bar\alpha_t}}  # 和DDPM是一样的，但是能跳步，即 t -> t - k
            x0_preds.append(x0_t.to('cpu'))
            # eta 是方差的系数，如果是0，就相当于DDIM，如果是1，就相当于DDPM的一种特殊情况
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )  # c1 = \eta * \sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-k}} * \frac{1 - \bar\alpha_{t-k}}{1-\bar\alpha_t}}   # c1 是每一步加的噪声的系数
            c2 = ((1 - at_next) - c1 ** 2).sqrt()  # c2 = \sqrt{1 - \bar\alpha_{t-k} - c1 ^ 2}  c2是模型预测出的噪声\epsilon的系数
            # 总的采样公式，共三项，详细的推导和理解可以去README的解读中查看
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et  # x_{t-k}=\sqrt{\bar\alpha_{t-k}}\hat{x}_{0|t}+\sqrt{1-\bar\alpha_{t-k}-\sigma^2_t}\epsilon_{\theta}(x_t,t)+\sigma_t\epsilon
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

