"""Restoration helpers, loaded on demand so predictor use needs no CIL install."""
from importlib import import_module

_EXPORTS = {'AbstractMetric': ('metrics', 'AbstractMetric'),
 'MAEMetric': ('metrics', 'MAEMetric'),
 'PSNRMetric': ('metrics', 'PSNRMetric'),
 'SSIMMetric': ('metrics', 'SSIMMetric'),
 'CBSD68Loader': ('loaders', 'CBSD68Loader'),
 'CIFAR10Loader': ('loaders', 'CIFAR10Loader'),
 'DIV2KLoader': ('loaders', 'DIV2KLoader'),
 'DatasetLoader': ('loaders', 'DatasetLoader'),
 'FolderImageLoader': ('loaders', 'FolderImageLoader'),
 'Kodak24Loader': ('loaders', 'Kodak24Loader'),
 'STL10Loader': ('loaders', 'STL10Loader'),
 'Set12Loader': ('loaders', 'Set12Loader'),
 'Set14Loader': ('loaders', 'Set14Loader'),
 'Set5Loader': ('loaders', 'Set5Loader'),
 'Urban100Loader': ('loaders', 'Urban100Loader'),
 'WEDLoader': ('loaders', 'WEDLoader'),
 'DEFAULT_LEVIN_KERNEL_ROOT': ('levin', 'DEFAULT_LEVIN_KERNEL_ROOT'),
 'describe_kernel_subset': ('levin', 'describe_kernel_subset'),
 'ensure_levin_kernel_files': ('levin', 'ensure_levin_kernel_files'),
 'get_levin_kernels': ('levin', 'get_levin_kernels'),
 'load_levin_kernels': ('levin', 'load_levin_kernels'),
 'resolve_leave_one_kernel_out_split': ('levin', 'resolve_leave_one_kernel_out_split'),
 'ExactPoissonGaussianFunction': ('cil_pg', 'ExactPoissonGaussianFunction'),
 'KnownPSFConvolutionOperator': ('cil_ops', 'KnownPSFConvolutionOperator'),
 'CILRegularizerSpec': ('cil_regularizers', 'CILRegularizerSpec'),
 'ProxOpElementwiseFunction': ('cil_regularizers', 'ProxOpElementwiseFunction'),
 'build_cil_regularizer': ('cil_regularizers', 'build_cil_regularizer'),
 'CILDataTermSpec': ('cil_terms', 'CILDataTermSpec'),
 'build_cil_data_term': ('cil_terms', 'build_cil_data_term'),
 'build_known_psf_operator': ('cil_terms', 'build_known_psf_operator'),
 'ProblemConfig': ('config', 'ProblemConfig'),
 'DEFAULT_NOISE_FOR_DATA_TERM': ('noise_models', 'DEFAULT_NOISE_FOR_DATA_TERM'),
 'NoiseRealization': ('noise_models', 'NoiseRealization'),
 'apply_noise': ('noise_models', 'apply_noise'),
 'make_rng': ('noise_models', 'make_rng'),
 'AbstractSolver': ('solvers', 'AbstractSolver'),
 'CCPiTGVExternalDenoiseSolver': ('solvers', 'CCPiTGVExternalDenoiseSolver'),
 'CILKnownPSFDeblurSolver': ('solvers', 'CILKnownPSFDeblurSolver'),
 'CCPiPoissonGaussianDeblurSolver': ('solvers', 'CCPiPoissonGaussianDeblurSolver'),
 'HuberADMMTVDenoiseSolver': ('solvers', 'HuberADMMTVDenoiseSolver'),
 'L1ADMMTVDenoiseSolver': ('solvers', 'L1ADMMTVDenoiseSolver'),
 'NormalizedSplitBregmanTVSolver': ('solvers', 'NormalizedSplitBregmanTVSolver')}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value
