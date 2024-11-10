""" Script to examine feature differences between ensemble of ResNet18 modules and the batchtable weight version with multiple runs """

from collections import OrderedDict, defaultdict
from hyper.generators.ensemble import FixedEnsembleModel
from hyper.target.resnet import resnet18
from hyper.target.testing.resnetens import ResNetEnsemble
from hyper.util.collections import flatten_keys, unflatten_keys, DefaultOrderedDict
import torch 
import copy
import time

# Number of runs
NUM_RUNS = 10

# Shared input
input_tensor = torch.randn(3, 3, 32, 32).normal_(0.0, 0.5).cuda()

# Accumulators for statistics
# Structure: {feature_name: {'mean': sum, 'std': sum, 'var': sum}}
accumulated_ensemble = DefaultOrderedDict(lambda: {'mean': 0.0, 'std': 0.0, 'var': 0.0})
accumulated_copied_hyper = DefaultOrderedDict(lambda: {'mean': 0.0, 'std': 0.0, 'var': 0.0})
accumulated_random_hyper = DefaultOrderedDict(lambda: {'mean': 0.0, 'std': 0.0, 'var': 0.0})


# Loop over the number of runs
for run in range(1, NUM_RUNS + 1):
    print(f'===== Run {run}/{NUM_RUNS} =====\n')
    
    ### 1. Initialize and evaluate the typical ResNetEnsemble
    print('Initializing ResNetEnsemble (PyTorch modules)')
    ens = ResNetEnsemble(
        size=5, classes=10
    ).cuda()
    
    # Forward pass
    ens_f, ens_p = ens(None, input_tensor)
    
    # Collect and accumulate feature statistics
    for feature_name, feature_value in flatten_keys(ens_f).items():
        if feature_value.sum().isnan():
            print('NAN feature', feature_name, feature_value.shape)
            exit(0)
        accumulated_ensemble[feature_name]['mean'] += feature_value.mean().item()
        accumulated_ensemble[feature_name]['std'] += feature_value.std().item()
        accumulated_ensemble[feature_name]['var'] += feature_value.var().item()
    
    print('ResNetEnsemble evaluation complete.\n')
    
    ### 2. Initialize FixedEnsembleModel with copied weights from ResNetEnsemble
    print('Initializing FixedEnsembleModel (Copied Weights)')
    hyper_copied = FixedEnsembleModel(
        target=resnet18(10),
        ensemble_size=5
    ).cuda()
    
    # List all modules in the typical ensemble
    ens_modules = dict(ens.named_modules())
    
    # Extract relevant parameters for hyper_copied module definition
    hyper_mods = filter(lambda x: x[1].is_generated(), flatten_keys(hyper_copied.define_generated_modules()).items())
    all_h_params = OrderedDict()
    
    for name, mod in hyper_mods:
        if mod is not None:
            extract_name = name
            if 'norm_buffer' in extract_name:
                continue  # Ignore normalization buffers
            if 'norm' in extract_name or 'shortcut.1' in extract_name:
                extract_name = extract_name.replace('.affine', '')  # Remove affine from norm layers
                
                if 'norm1' in extract_name:
                    extract_name = extract_name.replace('norm1', 'bn1')
                if 'norm2' in extract_name:
                    extract_name = extract_name.replace('norm2', 'bn2')
            extract_name = extract_name.replace('.self', '')
            
            # Override parameter scaling
            def null_scale(*args):
                if len(args) == 1:
                    return args[0]
                return args
            mod._param_scale = null_scale 
            
            # Copy weights from the typical ensemble to hyper_copied
            flat_params = []
            for n in mod.names():
                full_name = f'{name}.{n}'
                
                # extract from ensemble
                param_set = []
                for i in range(5):
                    p = ens_modules[f'ens.{i}.{extract_name}'].get_parameter(n)
                    param_set.append(p)
                param_set = torch.stack(param_set).view(5, -1)
                print(n, 'VAR', param_set.var())
                flat_params.append(param_set)
            
            # Concatenate all parameters for the current module
            all_h_params[name] = torch.cat(flat_params, dim=1)
    
    # Update hyper_copied's parameters
    params_copied = flatten_keys(hyper_copied.forward_params(None, device='cuda'))
    params_copied.update(all_h_params)
    params_copied = unflatten_keys(params_copied)
    
    # Forward pass
    hyper_copied_f, hyper_copied_p = hyper_copied.forward(params=params_copied, x=input_tensor)
    
    # Collect and accumulate feature statistics
    for feature_name, feature_value in flatten_keys(hyper_copied_f).items():
        accumulated_copied_hyper[feature_name]['mean'] += feature_value.mean().item()
        accumulated_copied_hyper[feature_name]['std'] += feature_value.std().item()
        accumulated_copied_hyper[feature_name]['var'] += feature_value.var().item()
    
    print('FixedEnsembleModel (Copied Weights) evaluation complete.\n')
    
    ### 3. Initialize FixedEnsembleModel with random sampled weights
    print('Initializing FixedEnsembleModel (Random Weights)')
    hyper_random = FixedEnsembleModel(
        target=resnet18(10),
        ensemble_size=5
    ).cuda()
    
    # Forward pass with random weights
    hyper_random_f, hyper_random_p = hyper_random.forward(None, x=input_tensor, sample_params=True)
    
    # Collect and accumulate feature statistics
    for feature_name, feature_value in flatten_keys(hyper_random_f).items():
        accumulated_random_hyper[feature_name]['mean'] += feature_value.mean().item()
        accumulated_random_hyper[feature_name]['std'] += feature_value.std().item()
        accumulated_random_hyper[feature_name]['var'] += feature_value.var().item()
    
    print('FixedEnsembleModel (Random Weights) evaluation complete.\n')
    
    print(f'===== Run {run} Complete =====\n\n')

### After all runs, compute average statistics

# Function to compute average statistics
def average_stats(accumulator, num_runs):
    averaged = DefaultOrderedDict()
    for feature, stats in accumulator.items():
        print('FEAT', stats)
        averaged[feature] = {
            'mean': stats['mean'] / num_runs,
            'std': stats['std'] / num_runs,
            'var': stats['var'] / num_runs
        }
    return averaged

# Compute averaged statistics
averaged_ensemble = average_stats(accumulated_ensemble, NUM_RUNS)
averaged_copied_hyper = average_stats(accumulated_copied_hyper, NUM_RUNS)
averaged_random_hyper = average_stats(accumulated_random_hyper, NUM_RUNS)

### Reporting the final averaged statistics

print('===== Final Averaged Feature Statistics Over All Runs =====\n')

# Get all feature names

for feat, (ensemble_stats, copied_hyper_stats, random_hyper_stats) in zip(averaged_copied_hyper.keys(), zip(averaged_ensemble.values(), averaged_copied_hyper.values(), averaged_random_hyper.values())):
    print(f'Feature: {feat}')
    # Ensemble
    print(f'  ResNetEnsemble:')
    print(f'    Mean: {ensemble_stats["mean"]:.4f}')
    print(f'    STD : {ensemble_stats["std"]:.4f}')
    print(f'    VAR : {ensemble_stats["var"]:.4f}')
    
    # Copied Hyper
    print(f'  FixedEnsembleModel (Copied Weights):')
    print(f'    Mean: {copied_hyper_stats["mean"]:.4f}')
    print(f'    STD : {copied_hyper_stats["std"]:.4f}')
    print(f'    VAR : {copied_hyper_stats["var"]:.4f}')
    
    # Random Hyper
    print(f'  FixedEnsembleModel (Random Weights):')
    print(f'    Mean: {random_hyper_stats["mean"]:.4f}')
    print(f'    STD : {random_hyper_stats["std"]:.4f}')
    print(f'    VAR : {random_hyper_stats["var"]:.4f}')
    
    print('\n')  # Newline for better readability between features

print('===== All Runs Complete =====')
