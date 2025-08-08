import numpy as np

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a displayable NumPy image.
    It denormalizes, moves to CPU, and changes dimension order.
    """
    # Denormalize the image from [-1, 1] to [0, 1]
    image = tensor * 0.5 + 0.5
    # Move tensor to CPU and convert to NumPy array
    image = image.cpu().numpy()
    # Transpose dimensions from (C, H, W) to (H, W, C) for plotting
    image = image.transpose(1, 2, 0)
    # Clip values to be in the valid [0, 1] range for images
    image = np.clip(image, 0, 1)
    return image

def tensor_to_image_gan(tensor):
    """
    Converts a PyTorch tensor to a NumPy image array for visualization.
    It handles denormalization and permutes the dimensions correctly.
    """
    # Denormalize from [-1, 1] to [0, 1]
    image = (tensor.clamp(-1, 1) + 1) / 2
    # Move from (C, H, W) to (H, W, C) for Matplotlib
    return image.permute(1, 2, 0)

class ComprehensiveMetrics:
    """Extended metrics for thorough evaluation"""

    def __init__(self, device='cuda'):
        self.device = device
        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = lpips.LPIPS(net='alex').to(device)

        # FIX: Use smaller kernel size or make it optional
        try:
            self.ms_ssim = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0,
                kernel_size=7,  # Reduced from 11 to 7
                betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
            ).to(device)
            self.ms_ssim_available = True
        except:
            self.ms_ssim_available = False
            print("MS-SSIM not available, skipping...")

    def calculate_all_metrics(self, pred, target, mask=None):
        """Calculate comprehensive metrics"""
        metrics = {}

        # FIX: Ensure proper dimensions
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # Convert to [0,1] range
        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2

        # Basic metrics
        metrics['psnr'] = self.psnr(pred_01, target_01).item()
        metrics['ssim'] = self.ssim(pred_01, target_01).item()

        # FIX: Only calculate MS-SSIM if image is large enough
        if self.ms_ssim_available and pred_01.shape[2] >= 160 and pred_01.shape[3] >= 160:
            try:
                metrics['ms_ssim'] = self.ms_ssim(pred_01, target_01).item()
            except:
                metrics['ms_ssim'] = metrics['ssim']  # Fallback to regular SSIM
        else:
            metrics['ms_ssim'] = metrics['ssim']  # Use regular SSIM for small images

        metrics['lpips'] = self.lpips(pred, target).mean().item()
        metrics['mae'] = F.l1_loss(pred_01, target_01).item()
        metrics['mse'] = F.mse_loss(pred_01, target_01).item()

        # Gradient difference (edge preservation)
        pred_grad = self._compute_gradient(pred)
        target_grad = self._compute_gradient(target)
        metrics['grad_diff'] = F.l1_loss(pred_grad, target_grad).item()

        # If mask provided, calculate region-specific metrics
        if mask is not None:
            # FIX: Ensure mask has proper dimensions
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)

            # Ensure mask is single channel for calculations
            if mask.shape[1] == 3:
                mask_single = mask[:, 0:1, :, :]
            else:
                mask_single = mask

            hole_mask = 1 - mask_single

            # Metrics for inpainted region only
            metrics['psnr_hole'] = self._masked_psnr(pred_01, target_01, hole_mask).item()
            metrics['ssim_hole'] = self._masked_ssim(pred_01, target_01, mask_single)

            # FIX: Check if there are any holes before calculating MAE
            if hole_mask.sum() > 0:
                metrics['mae_hole'] = (F.l1_loss(pred_01 * hole_mask, target_01 * hole_mask, reduction='sum') / hole_mask.sum()).item()
            else:
                metrics['mae_hole'] = 0.0

            # Boundary metrics (transition quality)
            boundary = self._get_boundary(mask_single)
            if boundary.sum() > 0:
                metrics['boundary_mae'] = (F.l1_loss(pred_01 * boundary, target_01 * boundary, reduction='sum') / boundary.sum()).item()
            else:
                metrics['boundary_mae'] = 0.0

        return metrics

    def _compute_gradient(self, x):
        """Compute image gradients using Sobel filter"""
        # FIX: Ensure proper batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = sobel_x.transpose(2, 3)

        # Apply to each channel
        grad_x = F.conv2d(x, sobel_x.repeat(3, 1, 1, 1), groups=3, padding=1)
        grad_y = F.conv2d(x, sobel_y.repeat(3, 1, 1, 1), groups=3, padding=1)

        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # Add epsilon for stability

    def _get_boundary(self, mask, dilation_size=3):
        """Get boundary region of mask"""
        # Ensure mask is single channel and 4D
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.dim() == 4 and mask.shape[1] == 3:
            mask = mask[:, 0:1, :, :]
        elif mask.dim() == 4 and mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]

        kernel = torch.ones(1, 1, dilation_size, dilation_size).to(mask.device)
        dilated = F.conv2d(mask, kernel, padding=dilation_size//2)
        dilated = (dilated > 0).float()

        eroded = F.conv2d(mask, kernel, padding=dilation_size//2)
        eroded = (eroded == dilation_size**2).float()

        boundary = dilated - eroded
        return boundary

    def _masked_psnr(self, pred, target, mask):
        """Calculate PSNR only in masked region"""
        # FIX: Ensure all have same dimensions
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        masked_pred = pred * mask
        masked_target = target * mask

        if mask.sum() > 0:
            mse = ((masked_pred - masked_target) ** 2).sum() / mask.sum()
            if mse > 0:
                return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)
            else:
                return torch.tensor(40.0)  # Perfect reconstruction
        else:
            return torch.tensor(0.0)

    def _masked_ssim(self, pred, target, mask):
        """Calculate SSIM focusing on masked region"""
        # Ensure proper dimensions
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        # Handle multi-channel masks properly
        if mask.dim() == 4 and mask.shape[1] == 3:
            mask_single = mask[:, 0, :, :]
        elif mask.dim() == 4:
            mask_single = mask.squeeze(1) if mask.shape[1] == 1 else mask[:, 0, :, :]
        else:
            mask_single = mask

        # Find region containing mask
        coords = torch.where(mask_single > 0)
        if len(coords[0]) == 0 or len(coords[1]) == 0:
            # No mask region, return full SSIM
            return self.ssim(pred, target).item()

        # Get bounding box
        y_min, y_max = max(0, coords[0].min().item()), min(pred.shape[2], coords[0].max().item() + 1)
        x_min, x_max = max(0, coords[1].min().item()), min(pred.shape[3], coords[1].max().item() + 1)

        # Ensure we have a valid region
        if y_max - y_min < 11 or x_max - x_min < 11:
            # Region too small for SSIM, use MAE instead
            return 1.0 - F.l1_loss(pred, target).item()

        pred_crop = pred[:, :, y_min:y_max, x_min:x_max]
        target_crop = target[:, :, y_min:y_max, x_min:x_max]

        return self.ssim(pred_crop, target_crop).item()

def evaluate_with_statistics(model_cnn, model_gan, test_loader, num_samples=100):
    """Evaluate models with confidence intervals and statistical tests"""

    metrics_calc = ComprehensiveMetrics()
    results = {'cnn': [], 'gan': []}

    for i, batch in enumerate(tqdm(test_loader, desc="Statistical Evaluation")):
        if i >= num_samples // test_loader.batch_size:
            break

        images = batch.to('cuda')
        masked, masks = create_mask(images, mask_percentage=0.025)

        with torch.no_grad():
            # CNN inference
            cnn_input = torch.cat([masked, masks[:, 0:1]], dim=1)
            cnn_output = model_cnn(cnn_input)

            # GAN inference
            gan_output = model_gan(cnn_output, masked, masks[:, 0:1])

            # Calculate metrics for each image
            for j in range(images.shape[0]):
                metrics_cnn = metrics_calc.calculate_all_metrics(
                    cnn_output[j:j+1], images[j:j+1], masks[j:j+1]
                )
                metrics_gan = metrics_calc.calculate_all_metrics(
                    gan_output[j:j+1], images[j:j+1], masks[j:j+1]
                )

                results['cnn'].append(metrics_cnn)
                results['gan'].append(metrics_gan)

    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS WITH CONFIDENCE INTERVALS")
    print("="*70)

    metric_names = results['cnn'][0].keys()

    for metric in metric_names:
        cnn_values = [r[metric] for r in results['cnn']]
        gan_values = [r[metric] for r in results['gan']]

        # Calculate statistics
        cnn_mean = np.mean(cnn_values)
        cnn_std = np.std(cnn_values)
        cnn_ci = stats.t.interval(0.95, len(cnn_values)-1,
                                  loc=cnn_mean,
                                  scale=stats.sem(cnn_values))

        gan_mean = np.mean(gan_values)
        gan_std = np.std(gan_values)
        gan_ci = stats.t.interval(0.95, len(gan_values)-1,
                                  loc=gan_mean,
                                  scale=stats.sem(gan_values))

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(gan_values, cnn_values)

        print(f"\n{metric.upper()}:")
        print(f"  CNN: {cnn_mean:.4f} ± {cnn_std:.4f} (95% CI: [{cnn_ci[0]:.4f}, {cnn_ci[1]:.4f}])")
        print(f"  GAN: {gan_mean:.4f} ± {gan_std:.4f} (95% CI: [{gan_ci[0]:.4f}, {gan_ci[1]:.4f}])")
        print(f"  Difference: {gan_mean - cnn_mean:+.4f}")
        print(f"  P-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

        # Effect size (Cohen's d)
        cohens_d = (gan_mean - cnn_mean) / np.sqrt((cnn_std**2 + gan_std**2) / 2)
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

    return results

def create_html_report(results, save_path='report.html'):
    """Create an interactive HTML report with all results"""

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inpainting Evaluation Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-card {
                display: inline-block;
                padding: 15px;
                margin: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background: #f9f9f9;
            }
            .improvement { color: green; font-weight: bold; }
            .degradation { color: red; font-weight: bold; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Comprehensive Inpainting Evaluation Report</h1>
        <h2>Model Performance Comparison</h2>
    """

    # Add summary cards
    for metric in ['psnr', 'ssim', 'lpips', 'mae']:
        cnn_val = np.mean([r[metric] for r in results['cnn']])
        gan_val = np.mean([r[metric] for r in results['gan']])
        improvement = gan_val - cnn_val if metric != 'lpips' and metric != 'mae' else cnn_val - gan_val

        html_content += f"""
        <div class="metric-card">
            <h3>{metric.upper()}</h3>
            <p>CNN: {cnn_val:.4f}</p>
            <p>GAN: {gan_val:.4f}</p>
            <p class="{'improvement' if improvement > 0 else 'degradation'}">
                Δ: {improvement:+.4f}
            </p>
        </div>
        """

    # Add detailed table
    html_content += """
        <h2>Detailed Metrics Table</h2>
        <table id="metricsTable">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>CNN Mean</th>
                    <th>CNN Std</th>
                    <th>GAN Mean</th>
                    <th>GAN Std</th>
                    <th>Improvement</th>
                    <th>P-value</th>
                </tr>
            </thead>
            <tbody>
    """

    for metric in results['cnn'][0].keys():
        cnn_values = [r[metric] for r in results['cnn']]
        gan_values = [r[metric] for r in results['gan']]

        cnn_mean, cnn_std = np.mean(cnn_values), np.std(cnn_values)
        gan_mean, gan_std = np.mean(gan_values), np.std(gan_values)
        _, p_value = stats.ttest_rel(gan_values, cnn_values)

        improvement = gan_mean - cnn_mean
        if metric in ['lpips', 'mae', 'mse']:
            improvement = -improvement

        html_content += f"""
            <tr>
                <td>{metric}</td>
                <td>{cnn_mean:.4f}</td>
                <td>{cnn_std:.4f}</td>
                <td>{gan_mean:.4f}</td>
                <td>{gan_std:.4f}</td>
                <td class="{'improvement' if improvement > 0 else 'degradation'}">
                    {improvement:+.4f}
                </td>
                <td>{p_value:.6f}</td>
            </tr>
        """

    html_content += """
            </tbody>
        </table>

        <h2>Distribution Plots</h2>
        <div id="distributionPlot"></div>

        <script>
            // Add interactive plots here using Plotly
            // Example: distribution comparison
        </script>
    </body>
    </html>
    """

    with open(save_path, 'w') as f:
        f.write(html_content)

    print(f"HTML report saved to {save_path}")

