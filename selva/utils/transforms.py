import torch


def generate_multiple_segments(
        x: torch.Tensor,
        segment_size: int,
        step_size: int,
    ) -> torch.Tensor:
        # x: (B, T, ...)
        b, t, *rest = x.shape
        assert t >= segment_size, f'The length of the input tensor {t} is less than the segment size {segment_size}.'
        assert segment_size > step_size, f'The segment size {segment_size} should be greater than the step size {step_size}.'
        # partition the tensor into segments
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)
        
        return x  # (B, S, T, ...)
