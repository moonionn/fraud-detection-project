from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def no_sampling(X, y):
    """none sampling"""
    return X, y


def undersample_balance(X, y):
    """Random Undersampling"""
    sampler = RandomUnderSampler(random_state=42)
    return sampler.fit_resample(X, y)


def oversample_balance(X, y):
    """Random Oversampling"""
    sampler = RandomOverSampler(random_state=42)
    return sampler.fit_resample(X, y)


def smote_balance(X, y):
    """SMOTE"""
    sampler = SMOTE(random_state=42)
    return sampler.fit_resample(X, y)


def smoteenn_balance(X, y):
    """SMOTE + ENN"""
    sampler = SMOTEENN(random_state=42)
    return sampler.fit_resample(X, y)


def get_sampler(sampling_type='none'):
    """
    Get sampling function based on the specified sampling type.
    """
    samplers = {
        'none': no_sampling,
        'undersample': undersample_balance,
        'oversample': oversample_balance,
        'smote': smote_balance,
        'smoteenn': smoteenn_balance
    }

    if sampling_type not in samplers:
        raise ValueError(f"No support: {sampling_type}")

    return samplers[sampling_type]