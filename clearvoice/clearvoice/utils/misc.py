#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Import future compatibility features for Python 2/3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import necessary libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joblib import Parallel, delayed
import os 
import sys
import librosa  # Library for audio processing
import torchaudio  # Library for audio processing with PyTorch

# Constants
MAX_WAV_VALUE = 32768.0  # Maximum value for WAV files
EPS = 1e-6  # Small value to avoid division by zero

def read_and_config_file(input_path, decode=0):
    """Reads input paths from a file or directory and configures them for processing.

    Args:
        input_path (str): Path to the input directory or file.
        decode (int): Flag indicating if decoding should occur (1 for decode, 0 for standard read).

    Returns:
        list: A list of processed paths or dictionaries containing input and label paths.
    """
    processed_list = []

    # If decoding is requested, find files in a directory
    if decode:
        if os.path.isdir(input_path):
            processed_list = librosa.util.find_files(input_path, ext="wav")  # Look for WAV files
            if len(processed_list) == 0:
                processed_list = librosa.util.find_files(input_path, ext="flac")  # Fallback to FLAC files
        else:
            # Read paths from a file
            with open(input_path) as fid:
                for line in fid:
                    path_s = line.strip().split()  # Split line into parts
                    processed_list.append(path_s[0])  # Append the first part (input path)
        return processed_list

    # Read input-label pairs from a file
    with open(input_path) as fid:
        for line in fid:
            tmp_paths = line.strip().split()  # Split line into parts
            if len(tmp_paths) == 3:  # Expecting input, label, and duration
                sample = {'inputs': tmp_paths[0], 'labels': tmp_paths[1], 'duration': float(tmp_paths[2])}
            elif len(tmp_paths) == 2:  # Expecting input and label only
                sample = {'inputs': tmp_paths[0], 'labels': tmp_paths[1]}
            processed_list.append(sample)  # Append the sample dictionary
    return processed_list

def load_checkpoint(checkpoint_path, use_cuda):
    """Loads the model checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        use_cuda (bool): Flag indicating whether to use CUDA for loading.

    Returns:
        dict: The loaded checkpoint containing model parameters.
    """
    #if use_cuda:
    #    checkpoint = torch.load(checkpoint_path)  # Load using CUDA
    #else:
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)  # Load to CPU
    return checkpoint

def get_learning_rate(optimizer):
    """Retrieves the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.

    Returns:
        float: The current learning rate.
    """
    return optimizer.param_groups[0]["lr"]

def reload_for_eval(model, checkpoint_dir, use_cuda):
    """Reloads a model for evaluation from the specified checkpoint directory.

    Args:
        model (nn.Module): The model to be reloaded.
        checkpoint_dir (str): Directory containing checkpoints.
        use_cuda (bool): Flag indicating whether to use CUDA.

    Returns:
        None
    """
    print('Reloading from: {}'.format(checkpoint_dir))
    best_name = os.path.join(checkpoint_dir, 'last_best_checkpoint')  # Path to the best checkpoint
    ckpt_name = os.path.join(checkpoint_dir, 'last_checkpoint')  # Path to the last checkpoint
    if os.path.isfile(best_name):
        name = best_name 
    elif os.path.isfile(ckpt_name):
        name = ckpt_name
    else:
        print('Warning: No existing checkpoint or best_model found!')
        return
    
    with open(name, 'r') as f:
        model_name = f.readline().strip()  # Read the model name from the checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, model_name)  # Construct full checkpoint path
    print('Checkpoint path: {}'.format(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path, use_cuda)  # Load the checkpoint
    '''
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)  # Load model parameters
    else:
        model.load_state_dict(checkpoint, strict=False)
    '''
    if 'model' in checkpoint:
        pretrained_model = checkpoint['model']
    else:
        pretrained_model = checkpoint
    state = model.state_dict()
    for key in state.keys():
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            state[key] = pretrained_model[key]
        elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
             state[key] = pretrained_model[key.replace('module.', '')]
        elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
             state[key] = pretrained_model['module.'+key]
        else: print(f'{key} not loaded')
    model.load_state_dict(state)

    print('=> Reload well-trained model {} for decoding.'.format(model_name))
    

def reload_model(model, optimizer, checkpoint_dir, use_cuda=True, strict=True):
    """Reloads the model and optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model to be reloaded.
        optimizer (torch.optim.Optimizer): The optimizer to be reloaded.
        checkpoint_dir (str): Directory containing checkpoints.
        use_cuda (bool): Flag indicating whether to use CUDA.
        strict (bool): If True, requires keys in state_dict to match exactly.

    Returns:
        tuple: Current epoch and step.
    """
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')  # Path to the checkpoint file
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()  # Read model name from checkpoint file
        checkpoint_path = os.path.join(checkpoint_dir, model_name)  # Construct full checkpoint path
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)  # Load the checkpoint
        model.load_state_dict(checkpoint['model'], strict=strict)  # Load model parameters
        optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer parameters
        epoch = checkpoint['epoch']  # Get current epoch
        step = checkpoint['step']  # Get current step
        print('=> Reloaded previous model and optimizer.')
    else:
        print('[!] Checkpoint directory is empty. Train a new model ...')
        epoch = 0  # Initialize epoch
        step = 0  # Initialize step
    return epoch, step

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, mode='checkpoint'):
    """Saves the model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        epoch (int): Current epoch number.
        step (int): Current training step number.
        checkpoint_dir (str): Directory to save the checkpoint.
        mode (str): Mode of the checkpoint ('checkpoint' or other).

    Returns:
        None
    """
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(epoch, step))  # Construct checkpoint file path
    torch.save({'model': model.state_dict(),  # Save model parameters
                'optimizer': optimizer.state_dict(),  # Save optimizer parameters
                'epoch': epoch,  # Save epoch
                'step': step}, checkpoint_path)  # Save checkpoint to file

    # Save the checkpoint name to a file for easy access
    with open(os.path.join(checkpoint_dir, mode), 'w') as f:
        f.write('model.ckpt-{}-{}.pt'.format(epoch, step))
    print("=> Saved checkpoint:", checkpoint_path)

def setup_lr(opt, lr):
    """Sets the learning rate for all parameter groups in the optimizer.

    Args:
        opt (torch.optim.Optimizer): The optimizer instance whose learning rate needs to be set.
        lr (float): The new learning rate to be assigned.
    
    Returns:
        None
    """
    for param_group in opt.param_groups:
        param_group['lr'] = lr  # Update the learning rate for each parameter group

def power_compress(x):
    """Compresses the power of a complex spectrogram.

    Args:
        x (torch.Tensor): Input tensor with real and imaginary components.

    Returns:
        torch.Tensor: Compressed magnitude and phase representation of the input.
    """
    real = x[..., 0]  # Extract real part
    imag = x[..., 1]  # Extract imaginary part
    spec = torch.complex(real, imag)  # Create complex tensor from real and imaginary parts
    mag = torch.abs(spec)  # Compute magnitude
    phase = torch.angle(spec)  # Compute phase
    
    mag = mag**0.3  # Compress magnitude using power of 0.3
    real_compress = mag * torch.cos(phase)  # Reconstruct real part
    imag_compress = mag * torch.sin(phase)  # Reconstruct imaginary part
    return torch.stack([real_compress, imag_compress], 1)  # Stack compressed parts


def power_uncompress(real, imag):
    """Uncompresses the power of a compressed complex spectrogram.

    Args:
        real (torch.Tensor): Compressed real component.
        imag (torch.Tensor): Compressed imaginary component.

    Returns:
        torch.Tensor: Uncompressed complex spectrogram.
    """
    spec = torch.complex(real, imag)  # Create complex tensor from real and imaginary parts
    mag = torch.abs(spec)  # Compute magnitude
    phase = torch.angle(spec)  # Compute phase
    
    mag = mag**(1./0.3)  # Uncompress magnitude by raising to the power of 1/0.3
    real_uncompress = mag * torch.cos(phase)  # Reconstruct real part
    imag_uncompress = mag * torch.sin(phase)  # Reconstruct imaginary part
    return torch.stack([real_uncompress, imag_uncompress], -1)  # Stack uncompressed parts


def stft(x, args, center=False, periodic=False, onesided=None):
    """Computes the Short-Time Fourier Transform (STFT) of an audio signal.

    Args:
        x (torch.Tensor): Input audio signal.
        args (Namespace): Configuration arguments containing window type and lengths.
        center (bool): Whether to center the window.

    Returns:
        torch.Tensor: The computed STFT of the input signal.
    """
    win_type = args.win_type
    win_len = args.win_len
    win_inc = args.win_inc
    fft_len = args.fft_len

    # Handle dtype/device and create window tensor on x.device with appropriate dtype
    orig_dtype = x.dtype
    compute_dtype = orig_dtype
    x_compute = x
    # Upcast FP16 on CUDA for numerical stability / library support
    if x.is_cuda and orig_dtype == torch.float16:
        compute_dtype = torch.float32
        x_compute = x.to(dtype=compute_dtype)

    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=periodic, dtype=compute_dtype, device=x.device)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=periodic, dtype=compute_dtype, device=x.device)
    else:
        print(f"In STFT, {win_type} is not supported!")
        return

    # Compute STFT in compute_dtype, cast back if needed
    out = torch.stft(x_compute, fft_len, win_inc, win_len, center=center, window=window, onesided=onesided, return_complex=False)
    if out.dtype != orig_dtype:
        out = out.to(dtype=orig_dtype)
    return out

def istft(x, args, slen=None, center=False, normalized=False, periodic=False, onesided=None, return_complex=False):
    """Computes the inverse Short-Time Fourier Transform (ISTFT) of a complex spectrogram.

    Args:
        x (torch.Tensor): Input complex spectrogram.
        args (Namespace): Configuration arguments containing window type and lengths.
        slen (int, optional): Length of the output signal.
        center (bool): Whether to center the window.
        normalized (bool): Whether to normalize the output.
        onesided (bool, optional): If True, computes only the one-sided transform.
        return_complex (bool): If True, returns complex output.

    Returns:
        torch.Tensor: The reconstructed audio signal from the spectrogram.
    """
    win_type = args.win_type
    win_len = args.win_len
    win_inc = args.win_inc
    fft_len = args.fft_len

    # Handle dtype/device and create window tensor on x.device with appropriate dtype
    orig_dtype = x.dtype
    compute_dtype = orig_dtype
    x_compute = x
    # Upcast FP16 on CUDA for numerical stability / library support
    if x.is_cuda and orig_dtype == torch.float16:
        compute_dtype = torch.float32
        x_compute = x.to(dtype=compute_dtype)

    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=periodic, dtype=compute_dtype, device=x.device)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=periodic, dtype=compute_dtype, device=x.device)
    else:
        print(f"In ISTFT, {win_type} is not supported!")
        return

    try:
        # Attempt to compute ISTFT in compute_dtype
        output = torch.istft(x_compute, n_fft=fft_len, hop_length=win_inc, win_length=win_len,
                              window=window, center=center, normalized=normalized,
                              onesided=onesided, length=slen, return_complex=False)
    except:
        # Handle potential errors by converting x to a complex tensor (ensure compute dtype)
        x_complex = torch.view_as_complex(x_compute)
        output = torch.istft(x_complex, n_fft=fft_len, hop_length=win_inc, win_length=win_len,
                              window=window, center=center, normalized=normalized,
                              onesided=onesided, length=slen, return_complex=False)

    if output.dtype != orig_dtype:
        output = output.to(dtype=orig_dtype)
    return output

def compute_fbank(audio_in, args):
    """Compute Mel filter bank features on CUDA in FP16 and return (frames, mels).

    This replaces the previous Kaldi fbank with torchaudio's MelSpectrogram.
    The transform is constructed once and cached in `args` to avoid reallocation.

    Args:
        audio_in (torch.Tensor): Input audio of shape (batch, time).
        args (Namespace): Configuration with fields sampling_rate, fft_len, win_len, win_inc, num_mels, win_type,
                          and use_cuda (1 for CUDA, 0 for CPU).

    Returns:
        torch.Tensor: Mel features with shape (frames, mels).
    """
    # Decide device and dtype
    use_cuda = getattr(args, 'use_cuda', 0) == 1 and torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    dtype = torch.float16 if use_cuda else torch.float32

    # Build and cache the transform once
    if not hasattr(args, '_mel_transform'):
        window_fn = torch.hamming_window if getattr(args, 'win_type', 'hamming') == 'hamming' else torch.hann_window
        args._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sampling_rate,
            n_fft=args.fft_len,
            win_length=args.win_len,
            hop_length=args.win_inc,
            n_mels=args.num_mels,
            window_fn=window_fn,
            center=False,
            power=2.0,
            norm=None,
            mel_scale='htk',
        )
        args._mel_transform = args._mel_transform.to(device=device, dtype=dtype)
    else:
        # Ensure cached transform is on the desired device/dtype
        args._mel_transform = args._mel_transform.to(device=device, dtype=dtype)

    # Ensure input is on the same device/dtype and is 2D (B, T)
    if audio_in.dim() == 1:
        audio_in = audio_in.unsqueeze(0)
    audio_in = audio_in.to(device=device, dtype=dtype)

    # Compute mel-spectrogram: (B, n_mels, frames)
    mel = args._mel_transform(audio_in)

    # Return (frames, mels) for a single item
    # Callers provide B=1; preserve existing interface returning 2D tensor
    mel_single = mel[0]  # (n_mels, frames)
    return mel_single.transpose(0, 1)
                                             


def _get_delta_kernel(args, device, dtype):
    """Create or fetch a cached depthwise Conv1d kernel for deltas.

    Uses a symmetric 5-tap kernel [-2, -1, 0, 1, 2] normalized by 10,
    applied per mel channel via depthwise grouping.
    """
    need_rebuild = (
        not hasattr(args, '_delta_kernel_weight') or
        getattr(args, '_delta_num_mels', None) != args.num_mels or
        args._delta_kernel_weight.device != device or
        args._delta_kernel_weight.dtype != dtype
    )
    if need_rebuild:
        base = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device, dtype=torch.float32)
        kernel = (base / 10.0).to(dtype).view(1, 1, 5)
        weight = kernel.expand(args.num_mels, 1, 5).contiguous()
        args._delta_kernel_weight = weight
        args._delta_num_mels = args.num_mels
    return args._delta_kernel_weight


def compute_deltas_conv(fbanks, args):
    """Compute delta and delta-delta via depthwise Conv1d on time axis.

    Args:
        fbanks (Tensor): (frames, mels)
        args (Namespace): expects num_mels, use_cuda

    Returns:
        (delta, delta_delta): both (frames, mels), same device/dtype as fbanks
    """
    if fbanks.dim() != 2:
        raise ValueError('compute_deltas_conv expects 2D tensor (frames, mels)')

    device = fbanks.device
    dtype = fbanks.dtype

    # Prepare input for conv: (N=1, C=mels, L=frames)
    x = fbanks.transpose(0, 1).unsqueeze(0)
    # Replication pad on time axis for 5-tap kernel
    x_pad = F.pad(x, (2, 2), mode='replicate')
    weight = _get_delta_kernel(args, device, dtype)
    delta = F.conv1d(x_pad, weight, bias=None, stride=1, padding=0, groups=args.num_mels)
    # Back to (frames, mels)
    delta = delta.squeeze(0).transpose(0, 1)

    # Delta-delta: apply same kernel on delta
    x2 = delta.transpose(0, 1).unsqueeze(0)
    x2_pad = F.pad(x2, (2, 2), mode='replicate')
    delta_delta = F.conv1d(x2_pad, weight, bias=None, stride=1, padding=0, groups=args.num_mels)
    delta_delta = delta_delta.squeeze(0).transpose(0, 1)

    return delta, delta_delta

