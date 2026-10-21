"""Differentiable form of the fixed-time playback transition."""
import torch


def playback_interval(streaming, auxiliary, throughput, action, *, interval_s=10.0,
                      chunk_duration_s=4.0, total_chunks=48):
    buffer, last, rebuffer, download, progress = streaming.unbind(-1)
    buffer, last, rebuffer, download = [x.clamp_min(0) for x in (buffer, last, rebuffer, download)]
    progress = progress.clamp(0, 1)
    inflight, rate, elapsed, stalled = auxiliary.clamp_min(0).unbind(-1)
    throughput = throughput.clamp_min(1e-3)
    bitrate = action[:, 0].clamp_min(0)
    count = action[:, 2].round().clamp_min(0)
    remaining = torch.full_like(buffer, interval_s)
    for _ in range(int(count.detach().max().item()) + 1):
        active = (remaining > 1e-9) & ((inflight > 1e-9) | (count > 0))
        begin = active & (inflight <= 1e-9)
        inflight = torch.where(begin, bitrate * chunk_duration_s, inflight)
        rate = torch.where(begin, bitrate, rate)
        elapsed = torch.where(begin, torch.zeros_like(elapsed), elapsed)
        stalled = torch.where(begin, torch.zeros_like(stalled), stalled)
        count = count - begin.float()
        required = inflight / throughput
        complete = active & (required <= remaining)
        spent = torch.where(active, torch.minimum(required, remaining), 0.)
        new_stall = (spent - buffer).clamp_min(0)
        rebuffer, stalled = rebuffer + new_stall, stalled + new_stall
        buffer = (buffer - spent).clamp_min(0)
        inflight = (inflight - throughput * spent).clamp_min(0)
        elapsed, remaining = elapsed + spent, remaining - spent
        buffer = buffer + complete.float() * chunk_duration_s
        last = torch.where(complete, rate, last)
        download = torch.where(complete, elapsed, download)
        progress = torch.where(complete, (progress + 1 / total_chunks).clamp_max(1), progress)
        inflight, rate, elapsed, stalled = [
            torch.where(complete, torch.zeros_like(v), v) for v in (inflight, rate, elapsed, stalled)
        ]
    rebuffer = rebuffer + (remaining - buffer).clamp_min(0)
    buffer = (buffer - remaining).clamp_min(0)
    return (torch.stack((buffer, last, rebuffer, download, progress), -1),
            torch.stack((inflight, rate, elapsed, stalled), -1))
