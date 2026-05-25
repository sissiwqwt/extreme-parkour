import torch


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
    """Update per-task terrain curriculum buffers and levels.

    This is kept IsaacGym-free so the curriculum logic can be smoke-tested on
    CPU before running the full simulator.
    """
    success_rates = torch.zeros_like(levels, dtype=torch.float)
    updated_tasks = torch.zeros_like(levels, dtype=torch.bool)

    for task_id in torch.unique(task_ids):
        task_mask = task_ids == task_id
        sample_count = int(task_mask.sum().item())
        if sample_count == 0:
            continue

        task_idx = int(task_id.item())
        current_write_idx = int(write_idx[task_idx].item())
        end_idx = current_write_idx + sample_count
        task_successes = successes[task_mask].float()

        if sample_count >= window:
            success_buf[task_idx] = task_successes[-window:]
            end_idx = 0
        elif end_idx <= window:
            success_buf[task_idx, current_write_idx:end_idx] = task_successes
        else:
            first_count = window - current_write_idx
            success_buf[task_idx, current_write_idx:] = task_successes[:first_count]
            success_buf[task_idx, : end_idx % window] = task_successes[first_count:]

        write_idx[task_idx] = end_idx % window
        counts[task_idx] = min(int(counts[task_idx].item()) + sample_count, window)
        update_counts[task_idx] += sample_count

        if counts[task_idx] < min_samples:
            continue
        if update_counts[task_idx] < min_samples:
            continue

        current_count = int(counts[task_idx].item())
        success_rate = success_buf[task_idx, :current_count].mean()
        success_rates[task_idx] = success_rate
        updated_tasks[task_idx] = True
        if success_rate > up_threshold:
            levels[task_idx] += 1
        elif success_rate < down_threshold:
            levels[task_idx] -= 1

        levels[task_idx] = torch.clamp(levels[task_idx], 0, max_level - 1)
        update_counts[task_idx] = 0

    return success_rates, updated_tasks
