from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
__version__ = "2.8.4"
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_func, flash_attn_kvpacked_func, flash_attn_qkvpacked_func,
        flash_attn_varlen_func, flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func, flash_attn_with_kvcache,
    )
except Exception as _flash_attn_import_error:
    def _missing_binary(*args, **kwargs):
        raise ImportError(f"flash_attn binary interface unavailable: {_flash_attn_import_error}")
    flash_attn_func = _missing_binary
    flash_attn_kvpacked_func = _missing_binary
    flash_attn_qkvpacked_func = _missing_binary
    flash_attn_varlen_func = _missing_binary
    flash_attn_varlen_kvpacked_func = _missing_binary
    flash_attn_varlen_qkvpacked_func = _missing_binary
    flash_attn_with_kvcache = _missing_binary
