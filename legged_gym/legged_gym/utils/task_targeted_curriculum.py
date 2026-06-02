import torch


def _recent_successes(success_buf, write_idx, count, window):
    """Read the newest samples from a circular task buffer."""
    capacity = success_buf.shape[0]
    sample_count = min(int(count), int(window), capacity)
    if sample_count <= 0:
        return success_buf[:0]

    start = (int(write_idx) - sample_count) % capacity
    if start + sample_count <= capacity:
        return success_buf[start:start + sample_count]
    first_count = capacity - start
    return torch.cat((success_buf[start:], success_buf[:sample_count - first_count]), dim=0)


def update_task_targeted_curriculum(
    task_ids,
    successes,
    success_buf,
    counts,
    update_counts,
    write_idx,
    levels,
    window,
    min_samples,
    up_threshold,
    down_threshold,
    max_level,
):
    success_rates = torch.zeros_like(levels, dtype=torch.float)
    updated_tasks = torch.zeros_like(levels, dtype=torch.bool)
    observed_tasks = torch.zeros_like(levels, dtype=torch.bool)
    buffer_capacity = success_buf.shape[1]

    for task_id in torch.unique(task_ids):
        task_mask = task_ids == task_id
        sample_count = int(task_mask.sum().item())
        if sample_count == 0:
            continue

        task_idx = int(task_id.item())
        current_write_idx = int(write_idx[task_idx].item())
        end_idx = current_write_idx + sample_count
        task_successes = successes[task_mask].float()

        if sample_count >= buffer_capacity:
            success_buf[task_idx] = task_successes[-buffer_capacity:]
            end_idx = 0
        elif end_idx <= buffer_capacity:
            success_buf[task_idx, current_write_idx:end_idx] = task_successes
        else:
            first_count = buffer_capacity - current_write_idx
            success_buf[task_idx, current_write_idx:] = task_successes[:first_count]
            success_buf[task_idx, : end_idx % buffer_capacity] = task_successes[first_count:]

        write_idx[task_idx] = end_idx % buffer_capacity
        counts[task_idx] = min(int(counts[task_idx].item()) + sample_count, buffer_capacity)
        update_counts[task_idx] += sample_count

        current_count = int(counts[task_idx].item())
        current_samples = _recent_successes(
            success_buf[task_idx],
            write_idx[task_idx],
            current_count,
            window,
        )
        if current_samples.numel() == 0:
            continue
        success_rate = current_samples.mean()
        success_rates[task_idx] = success_rate
        observed_tasks[task_idx] = True

        if current_samples.numel() < min_samples:
            continue
        if update_counts[task_idx] < min_samples:
            continue

        updated_tasks[task_idx] = True
        if success_rate > up_threshold:
            levels[task_idx] += 1
        elif success_rate < down_threshold:
            levels[task_idx] -= 1

        levels[task_idx] = torch.clamp(levels[task_idx], 0, max_level - 1)
        update_counts[task_idx] = 0

    return success_rates, updated_tasks, observed_tasks


def update_task_pause_state(
    success_rates,
    counts,
    paused,
    active_ids,
    min_samples,
    pause_threshold,
    resume_threshold,
):
    for task_id in active_ids:
        task_idx = int(task_id.item())
        if int(counts[task_idx].item()) < min_samples:
            continue
        success_rate = float(success_rates[task_idx].item())
        if bool(paused[task_idx].item()):
            if success_rate < resume_threshold:
                paused[task_idx] = False
        elif success_rate > pause_threshold:
            paused[task_idx] = True


def compute_task_sampling_weights(
    success_rates,
    paused,
    active_ids,
    base_weights,
    prioritized_sampling,
    pause_solved_tasks,
    min_sampling_weight,
    priority_alpha,
):
    weights = torch.zeros_like(success_rates, dtype=torch.float)
    eps = 1e-6
    for task_id in active_ids:
        task_idx = int(task_id.item())
        base_weight = base_weights[task_idx]
        if not prioritized_sampling:
            weights[task_idx] = base_weight
            continue

        if pause_solved_tasks and bool(paused[task_idx].item()):
            weights[task_idx] = min_sampling_weight
        else:
            difficulty_score = torch.clamp(1.0 - success_rates[task_idx], min=0.0)
            weights[task_idx] = base_weight * torch.pow(difficulty_score + eps, priority_alpha)
            weights[task_idx] = torch.clamp(weights[task_idx], min=min_sampling_weight)

    active_weight_sum = weights[active_ids].sum()
    if active_weight_sum <= 0:
        weights[active_ids] = base_weights[active_ids]
        active_weight_sum = weights[active_ids].sum()
    if active_weight_sum <= 0:
        weights[active_ids] = 1.0
        active_weight_sum = weights[active_ids].sum()
    weights[active_ids] = weights[active_ids] / active_weight_sum
    return weights
