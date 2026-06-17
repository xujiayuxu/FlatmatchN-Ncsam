import os
import os.path as osp
import pprint
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from trainer import FreeMatchTrainer
from utils import disable_running_stats, enable_running_stats


class _RunTimer:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        if use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_time = None

    def start(self):
        if self.use_cuda:
            self.start_event.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000
        return time.time() - self.start_time


class FlatMatchNCSAMTrainer(FreeMatchTrainer):
    """FlatMatch with distribution-aware NCSAM perturbation compensation."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.ncsam_cfg = cfg.NCSAM
        self.ncsam_stats = self.__empty_ncsam_stats__()

    @staticmethod
    def __empty_ncsam_stats__():
        return {
            'train/ncsam_lambda': 0.0,
            'train/ncsam_select_ratio_target': 0.0,
            'train/ncsam_noise_loss': 0.0,
            'train/ncsam_noise_norm': 0.0,
            'train/ncsam_selected_ratio': 0.0,
            'train/ncsam_confidence': 0.0,
            'train/ncsam_disagreement': 0.0,
            'train/ncsam_margin': 0.0,
            'train/ncsam_density': 0.0,
            'train/ncsam_balance': 0.0,
        }

    def _method_config(self):
        result = super()._method_config()
        result.update({
            'method': 'flatmatch_ncsam',
            'ncsam_enabled': bool(self.ncsam_cfg.ENABLED),
            'ncsam_start_iter': int(self.ncsam_cfg.START_ITER),
            'ncsam_warmup_iters': int(self.ncsam_cfg.WARMUP_ITERS),
            'ncsam_ramp_iters': int(self.ncsam_cfg.RAMP_ITERS),
            'ncsam_plateau_iters': int(self.ncsam_cfg.PLATEAU_ITERS),
            'ncsam_decay_iters': int(self.ncsam_cfg.DECAY_ITERS),
            'ncsam_min_lambda': float(self.ncsam_cfg.MIN_LAMBDA),
            'ncsam_max_lambda': float(self.ncsam_cfg.MAX_LAMBDA),
            'ncsam_min_select_ratio': float(self.ncsam_cfg.MIN_SELECT_RATIO),
            'ncsam_select_ratio': float(self.ncsam_cfg.SELECT_RATIO),
            'ncsam_confidence_power': float(self.ncsam_cfg.CONFIDENCE_POWER),
            'ncsam_disagreement_power': float(self.ncsam_cfg.DISAGREEMENT_POWER),
            'ncsam_min_disagreement': float(self.ncsam_cfg.MIN_DISAGREEMENT),
            'ncsam_hard_disagreement_weight': float(self.ncsam_cfg.HARD_DISAGREEMENT_WEIGHT),
            'ncsam_margin_temperature': float(self.ncsam_cfg.MARGIN_TEMPERATURE),
            'ncsam_density_power': float(self.ncsam_cfg.DENSITY_POWER),
            'ncsam_min_density': float(self.ncsam_cfg.MIN_DENSITY),
            'ncsam_singleton_density': float(self.ncsam_cfg.SINGLETON_DENSITY),
            'ncsam_batch_prior_weight': float(self.ncsam_cfg.BATCH_PRIOR_WEIGHT),
            'ncsam_class_balance_power': float(self.ncsam_cfg.CLASS_BALANCE_POWER),
            'ncsam_max_balance_weight': float(self.ncsam_cfg.MAX_BALANCE_WEIGHT),
            'ncsam_max_sample_weight': float(self.ncsam_cfg.MAX_SAMPLE_WEIGHT),
        })
        return result

    def __stage_factor__(self):
        if not self.ncsam_cfg.ENABLED:
            return 0.0

        local_iter = self.curr_iter - self.ncsam_cfg.START_ITER
        if local_iter < 0:
            return 0.0

        warmup_iters = max(0, self.ncsam_cfg.WARMUP_ITERS)
        if local_iter < warmup_iters:
            return 0.0

        ramp_iters = max(1, self.ncsam_cfg.RAMP_ITERS)
        ramp_pos = local_iter - warmup_iters
        if ramp_pos < ramp_iters:
            return (ramp_pos + 1) / ramp_iters

        plateau_iters = max(0, self.ncsam_cfg.PLATEAU_ITERS)
        plateau_pos = ramp_pos - ramp_iters
        if plateau_pos < plateau_iters:
            return 1.0

        decay_iters = max(1, self.ncsam_cfg.DECAY_ITERS)
        decay_pos = plateau_pos - plateau_iters
        if decay_pos < decay_iters:
            return max(0.0, 1.0 - ((decay_pos + 1) / decay_iters))

        return 0.0

    def __scheduled_value__(self, min_value, max_value):
        factor = self.__stage_factor__()
        if factor <= 0.0:
            return 0.0
        return min_value + (max_value - min_value) * factor

    def __compensation_lambda__(self):
        return self.__scheduled_value__(
            float(self.ncsam_cfg.MIN_LAMBDA),
            float(self.ncsam_cfg.MAX_LAMBDA),
        )

    def __select_ratio__(self):
        return self.__scheduled_value__(
            float(self.ncsam_cfg.MIN_SELECT_RATIO),
            float(self.ncsam_cfg.SELECT_RATIO),
        )

    def __weighted_noise_loss__(self, logits_ulb_w, logits_ulb_s, feats_ulb_w):
        with torch.no_grad():
            weights, selected, stats = self.__noise_weights__(
                logits_ulb_w.detach(),
                logits_ulb_s.detach(),
                feats_ulb_w.detach(),
            )

        probs_ulb_w = torch.softmax(logits_ulb_w.detach(), dim=-1)
        per_sample_loss = F.kl_div(
            F.log_softmax(logits_ulb_s, dim=-1),
            probs_ulb_w,
            reduction='none',
        ).sum(dim=-1)
        selected_weights = weights[selected]
        selected_loss = per_sample_loss[selected]
        loss = (selected_loss * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)
        return loss, stats

    def __noise_weights__(self, logits_ulb_w, logits_ulb_s, feats_ulb_w):
        probs_w = torch.softmax(logits_ulb_w, dim=-1)
        probs_s = torch.softmax(logits_ulb_s, dim=-1)
        top2_probs, top2_idx = torch.topk(probs_w, k=2, dim=-1)
        pseudo_targets = top2_idx[:, 0]
        margins = top2_probs[:, 0] - top2_probs[:, 1]
        confidence = top2_probs[:, 0]

        low_confidence = (1.0 - confidence).clamp_min(0.0).pow(self.ncsam_cfg.CONFIDENCE_POWER)
        uncertainty = 1.0 / (1.0 + margins / max(self.ncsam_cfg.MARGIN_TEMPERATURE, 1e-6))
        soft_disagreement = 0.5 * torch.abs(probs_w - probs_s).sum(dim=-1)
        hard_disagreement = pseudo_targets.ne(torch.argmax(probs_s, dim=-1)).to(probs_w.dtype)
        disagreement = soft_disagreement + self.ncsam_cfg.HARD_DISAGREEMENT_WEIGHT * hard_disagreement
        disagreement = disagreement.clamp_min(self.ncsam_cfg.MIN_DISAGREEMENT)
        disagreement = disagreement.pow(self.ncsam_cfg.DISAGREEMENT_POWER)
        density = self.__batch_density__(feats_ulb_w, pseudo_targets)
        balance = self.__class_balance__(pseudo_targets, probs_w.shape[1])

        weights = low_confidence * disagreement * uncertainty * density.pow(self.ncsam_cfg.DENSITY_POWER) * balance
        weights = weights / weights.mean().clamp_min(1e-12)
        weights = weights.clamp(max=self.ncsam_cfg.MAX_SAMPLE_WEIGHT)

        select_ratio = self.__select_ratio__()
        selected = self.__select_noise_samples__(weights, select_ratio)
        if not selected.any():
            selected[torch.argmax(weights)] = True

        stats = {
            'train/ncsam_select_ratio_target': select_ratio,
            'train/ncsam_selected_ratio': selected.float().mean().item(),
            'train/ncsam_confidence': confidence[selected].mean().item(),
            'train/ncsam_disagreement': disagreement[selected].mean().item(),
            'train/ncsam_margin': margins[selected].mean().item(),
            'train/ncsam_density': density[selected].mean().item(),
            'train/ncsam_balance': balance[selected].mean().item(),
        }
        return weights.detach(), selected.detach(), stats

    def __batch_density__(self, feats, pseudo_targets):
        feats = F.normalize(feats, dim=-1)
        density = torch.ones(feats.shape[0], device=feats.device, dtype=feats.dtype)

        for cls in pseudo_targets.unique():
            cls_mask = pseudo_targets.eq(cls)
            cls_feats = feats[cls_mask]
            if cls_feats.shape[0] <= 1:
                density[cls_mask] = self.ncsam_cfg.SINGLETON_DENSITY
                continue

            centroid = F.normalize(cls_feats.mean(dim=0, keepdim=True), dim=-1)
            cosine_density = (cls_feats * centroid).sum(dim=-1)
            density[cls_mask] = ((cosine_density + 1.0) * 0.5).clamp(
                min=self.ncsam_cfg.MIN_DENSITY,
                max=1.0,
            )

        return density

    def __class_balance__(self, pseudo_targets, num_classes):
        batch_hist = torch.bincount(pseudo_targets, minlength=num_classes).to(self.label_hist.dtype)
        batch_prior = batch_hist / batch_hist.sum().clamp_min(1.0)
        running_prior = self.label_hist.detach()
        prior = self.ncsam_cfg.BATCH_PRIOR_WEIGHT * batch_prior + (1.0 - self.ncsam_cfg.BATCH_PRIOR_WEIGHT) * running_prior
        prior = prior.to(pseudo_targets.device).clamp_min(1e-6)

        balance = prior[pseudo_targets].pow(-self.ncsam_cfg.CLASS_BALANCE_POWER)
        balance = balance / balance.mean().clamp_min(1e-12)
        return balance.clamp(max=self.ncsam_cfg.MAX_BALANCE_WEIGHT)

    def __select_noise_samples__(self, weights, ratio):
        if ratio >= 1.0:
            return torch.ones_like(weights, dtype=torch.bool)

        num_selected = max(1, int(weights.numel() * max(ratio, 0.0)))
        selected_idx = torch.topk(weights, k=num_selected, largest=True).indices
        selected = torch.zeros_like(weights, dtype=torch.bool)
        selected[selected_idx] = True
        return selected

    def __normalize_grads__(self, grads, radius):
        grad_norm = self.norm(grads).clamp_min(1e-12)
        scale = radius / grad_norm
        eps = [g * scale if g is not None else None for g in grads]
        return eps, grad_norm

    def __project_perturbation__(self, perturbation, radius):
        perturb_norm = self.norm(perturbation)
        if perturb_norm <= radius:
            return perturbation
        scale = radius / perturb_norm.clamp_min(1e-12)
        return [p * scale if p is not None else None for p in perturbation]

    def __use_flatmatch_warmup__(self):
        if not self.ncsam_cfg.ENABLED:
            return False
        warmup_end = self.ncsam_cfg.START_ITER + self.ncsam_cfg.WARMUP_ITERS
        return self.curr_iter < warmup_end

    def __use_plain_flatmatch_step__(self):
        if not self.ncsam_cfg.ENABLED:
            return True
        if self.__use_flatmatch_warmup__():
            return True
        return self.__compensation_lambda__() <= 0.0

    def __build_compensated_perturbation__(self, loss_lb, logits_ulb_w, logits_ulb_s, feats_ulb_w):
        params = list(self.net.model.parameters())
        lambda_t = self.__compensation_lambda__()
        grad_w = torch.autograd.grad(loss_lb, params, retain_graph=lambda_t > 0.0, allow_unused=True)
        eps_fm, _ = self.__normalize_grads__(grad_w, self.rho)

        self.ncsam_stats = self.__empty_ncsam_stats__()
        self.ncsam_stats['train/ncsam_lambda'] = lambda_t

        if lambda_t <= 0.0:
            return grad_w, eps_fm

        noise_loss, stats = self.__weighted_noise_loss__(logits_ulb_w, logits_ulb_s, feats_ulb_w)
        grad_noise = torch.autograd.grad(noise_loss, params, allow_unused=True)
        eps_noise, noise_norm = self.__normalize_grads__(grad_noise, self.rho)

        compensated = []
        for eps_base, eps_bias in zip(eps_fm, eps_noise):
            if eps_base is None:
                compensated.append(None)
            elif eps_bias is None:
                compensated.append(eps_base)
            else:
                compensated.append(eps_base - lambda_t * eps_bias)

        compensated = self.__project_perturbation__(compensated, self.rho)
        self.ncsam_stats.update(stats)
        self.ncsam_stats['train/ncsam_noise_loss'] = noise_loss.item()
        self.ncsam_stats['train/ncsam_noise_norm'] = noise_norm.item()
        return grad_w, compensated

    @staticmethod
    def __add_perturbation__(params, perturbation):
        for param, eps in zip(params, perturbation):
            if eps is not None:
                param.add_(eps)

    def train(self):
        print('Starting FlatMatch+NCSAM model training...')

        self.model.train()
        use_cuda_timer = self.device == 'cuda'
        batch_timer = _RunTimer(use_cuda_timer)
        run_timer = _RunTimer(use_cuda_timer)
        batch_timer.start()
        progress = tqdm(
            total=max(0, self.num_train_iters - self.curr_iter),
            desc='FlatMatch+NCSAM',
            dynamic_ncols=True,
        )

        try:
            for batch_lb, batch_ulb in zip(self.dm.train_lb_dl, self.dm.train_ulb_dl):
                if self.curr_iter >= self.num_train_iters:
                    break

                fetch_time = batch_timer.stop()
                run_timer.start()

                if self.__use_plain_flatmatch_step__():
                    log_dict = self._train_flatmatch_batch(batch_lb, batch_ulb)
                    self.ncsam_stats = self.__empty_ncsam_stats__()
                    log_dict.update(self.ncsam_stats)

                    run_time = run_timer.stop()
                    log_dict['train/fetch_time'] = fetch_time
                    log_dict['train/run_time'] = run_time

                    if (self.curr_iter + 1) % self.num_eval_iters == 0:
                        print('Evaluating...')
                        validate_dict = self.validate()
                        log_dict.update(validate_dict)
                        save_dir = osp.join(self.cfg.LOG_DIR, self.cfg.RUN_NAME, self.cfg.OUTPUT_DIR)
                        if not osp.exists(save_dir):
                            os.makedirs(save_dir)

                        improved = validate_dict['validation/accuracy'] > self.best_test_acc
                        if improved:
                            self.best_test_acc = validate_dict['validation/accuracy']
                            self.best_test_iter = self.curr_iter
                            self.__save__model__(save_dir, 'best_checkpoint.pth')

                        self.__save__model__(save_dir, 'last_checkpoint.pth')
                        log_dict.update({
                            'best_acc': self.best_test_acc,
                            'best_iter': self.best_test_iter,
                        })
                        self._save_eval_result(log_dict, improved)
                        self.tb.update(log_dict, self.curr_iter)

                    progress.set_postfix({
                        'loss': '%.4f' % log_dict['train/total_loss'],
                        'sat': '%.4f' % log_dict['train/sat_loss'],
                        'mask': '%.3f' % log_dict['train/pseudo_accept'],
                        'lambda': '%.3f' % log_dict['train/ncsam_lambda'],
                        'noise': '%.3f' % log_dict['train/ncsam_selected_ratio'],
                        'phase': 'fm',
                        'best': '%.4f' % self.best_test_acc,
                    })
                    progress.update(1)

                    if (self.curr_iter + 1) % self.num_log_iters == 0:
                        print('Iteration: %d / %d' % (self.curr_iter + 1, self.num_train_iters))
                        print('Fetch Time: %.3f, Run Time: %.3f' % (fetch_time, run_time))
                        pprint.pprint(log_dict, indent=4)

                    self.curr_iter += 1
                    del log_dict
                    batch_timer.start()
                    continue

                img_lb_w, label_lb = batch_lb['img_w'], batch_lb['label']
                img_ulb_w, img_ulb_s = batch_ulb['img_w'], batch_ulb['img_s']

                img_lb_w, label_lb = img_lb_w.to(self.device), label_lb.to(self.device)
                img_ulb_w, img_ulb_s = img_ulb_w.to(self.device), img_ulb_s.to(self.device)

                num_lb = img_lb_w.shape[0]
                num_ulb = img_ulb_w.shape[0]
                assert num_ulb == img_ulb_s.shape[0]

                img = torch.cat([img_lb_w, img_ulb_w, img_ulb_s])
                params = list(self.net.model.parameters())

                with self.amp():
                    enable_running_stats(self.net.model)
                    out = self.net(img)
                    feats = out['feats']
                    logits = out['logits']
                    logits_lb = logits[:num_lb]
                    logits_ulb_w, logits_ulb_s = logits[num_lb:].chunk(2)
                    feats_ulb_w = feats[num_lb:num_lb + num_ulb]

                    loss_lb = self.ce_criterion(logits_lb, label_lb, reduction='mean')
                    self.grad_w, self.eps = self.__build_compensated_perturbation__(
                        loss_lb,
                        logits_ulb_w,
                        logits_ulb_s,
                        feats_ulb_w,
                    )

                with torch.no_grad():
                    self.__add_perturbation__(params, self.eps)

                disable_running_stats(self.net.model)

                with self.amp():
                    out_hat = self.net(img)
                    logits_hat = out_hat['logits']
                    logits_lb_hat = logits_hat[:num_lb]
                    _, logits_ulb_s_hat = logits_hat[num_lb:].chunk(2)

                    loss_lb = self.ce_criterion(logits_lb_hat, label_lb, reduction='mean')
                    loss_sat, mask, self.tau_t, self.p_t, self.label_hist = self.sat_criterion(
                        logits_ulb_w,
                        logits_ulb_s_hat,
                        self.tau_t,
                        self.p_t,
                        self.label_hist,
                    )
                    loss_saf, hist_p_ulb_s = self.saf_criterion(mask, logits_ulb_s_hat, self.p_t, self.label_hist)
                    loss = loss_lb + self.ulb_loss_ratio * loss_sat + self.ent_loss_ratio * loss_saf

                if self.cfg.TRAINER.AMP_ENABLED:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim.optimizer)
                    with torch.no_grad():
                        self._restore_perturbation_and_add_base_grads(params, self.eps, self.grad_w)
                    self.scaler.step(self.optim.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    with torch.no_grad():
                        self._restore_perturbation_and_add_base_grads(params, self.eps, self.grad_w)
                    self.optim.step()

                self.sched.step()
                self.net.update()
                self.model.zero_grad()

                run_time = run_timer.stop()

                log_dict = {
                    'train/lb_loss': loss_lb.item(),
                    'train/sat_loss': loss_sat.item(),
                    'train/saf_loss': loss_saf.item(),
                    'train/total_loss': loss.item(),
                    'train/mask': 1 - mask.mean().item(),
                    'train/pseudo_accept_count': int(mask.sum().item()),
                    'train/pseudo_total': int(mask.numel()),
                    'train/pseudo_accept': mask.mean().item(),
                    'train/pseudo_reject': 1 - mask.mean().item(),
                    'train/tau_t': self.tau_t.item(),
                    'train/p_t': self.p_t.mean().item(),
                    'train/label_hist': self.label_hist.mean().item(),
                    'train/label_hist_s': hist_p_ulb_s.mean().item(),
                    'train/lr': self.optim.optimizer.param_groups[0]['lr'],
                    'train/fetch_time': fetch_time,
                    'train/run_time': run_time,
                }
                log_dict.update(self.ncsam_stats)

                if (self.curr_iter + 1) % self.num_eval_iters == 0:
                    print('Evaluating...')
                    validate_dict = self.validate()
                    log_dict.update(validate_dict)
                    save_dir = osp.join(self.cfg.LOG_DIR, self.cfg.RUN_NAME, self.cfg.OUTPUT_DIR)
                    if not osp.exists(save_dir):
                        os.makedirs(save_dir)

                    improved = validate_dict['validation/accuracy'] > self.best_test_acc
                    if improved:
                        self.best_test_acc = validate_dict['validation/accuracy']
                        self.best_test_iter = self.curr_iter
                        self.__save__model__(save_dir, 'best_checkpoint.pth')

                    self.__save__model__(save_dir, 'last_checkpoint.pth')
                    log_dict.update({
                        'best_acc': self.best_test_acc,
                        'best_iter': self.best_test_iter,
                    })
                    self._save_eval_result(log_dict, improved)
                    self.tb.update(log_dict, self.curr_iter)

                progress.set_postfix({
                    'loss': '%.4f' % log_dict['train/total_loss'],
                    'sat': '%.4f' % log_dict['train/sat_loss'],
                    'mask': '%.3f' % log_dict['train/pseudo_accept'],
                    'lambda': '%.3f' % log_dict['train/ncsam_lambda'],
                    'noise': '%.3f' % log_dict['train/ncsam_selected_ratio'],
                    'phase': 'ncsam',
                    'best': '%.4f' % self.best_test_acc,
                })
                progress.update(1)

                if (self.curr_iter + 1) % self.num_log_iters == 0:
                    print('Iteration: %d / %d' % (self.curr_iter + 1, self.num_train_iters))
                    print('Fetch Time: %.3f, Run Time: %.3f' % (fetch_time, run_time))
                    pprint.pprint(log_dict, indent=4)

                self.curr_iter += 1
                del log_dict
                batch_timer.start()
        finally:
            progress.close()
