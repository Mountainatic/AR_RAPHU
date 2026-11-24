"""MODEL availability is separate from stage admission and solver failure."""
from .invariants import NumericallyInvalid

REGISTERED='REGISTERED'
AVAILABLE='AVAILABLE'
SUPPORT_LIMITED='SUPPORT_LIMITED'
DOMAIN_NOT_APPLICABLE='DOMAIN_NOT_APPLICABLE'
NUMERICALLY_INVALID='NUMERICALLY_INVALID'
MODEL_BRANCH_DEGENERATE='MODEL_BRANCH_DEGENERATE'

class SupportLimited(RuntimeError):pass
class DomainNotApplicable(RuntimeError):pass

def classify_exception(exc):
    if isinstance(exc,SupportLimited):return SUPPORT_LIMITED
    if isinstance(exc,DomainNotApplicable):return DOMAIN_NOT_APPLICABLE
    return NUMERICALLY_INVALID

def require_k_support(contract,rows,config):
    caps=config['row_caps']
    if rows<int(caps['minimum_rows_for_any_fitted_candidate']) or rows<int(caps['minimum_rows_per_free_parameter'])*int(contract['parameter_count']):
        raise SupportLimited('K rows below registered parameter support requirement')
    if contract.get('family')=='EXACT_ZERO':raise DomainNotApplicable('Degenerate fitted MODEL is not a BYPASS candidate')

def no_available_model(statuses):
    values=set(statuses)
    if NUMERICALLY_INVALID in values:raise NumericallyInvalid('All legal MODEL recipes failed numerical certification')
    return SUPPORT_LIMITED if SUPPORT_LIMITED in values else DOMAIN_NOT_APPLICABLE
