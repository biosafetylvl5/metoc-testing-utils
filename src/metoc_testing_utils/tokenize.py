"""TODO."""

import collections.abc
import functools
import hashlib
import typing

import dask.base
import xarray as xr


def is_xarray_object(obj: typing.Any) -> bool:
    """Check if an object is an xarray object."""
    return (
        (isinstance(obj, xr.DataArray | xr.Dataset | xr.DataTree))
        if hasattr(xr, "DataTree")
        else isinstance(obj, xr.DataArray | xr.Dataset)
    )


def is_collection(obj: typing.Any) -> bool:
    """Check if an object is a collection (but not a string)."""
    return isinstance(obj, collections.abc.Collection) and not (
        isinstance(obj, str | bytes | bytearray)
    )


def is_mapping(obj: typing.Any) -> bool:
    """Check if an object is a mapping."""
    return isinstance(obj, collections.abc.Mapping)


def tokenize_xarray(obj: typing.Any) -> str:
    """Tokenize an xarray object using dask's tokenize function."""
    return str(dask.base.tokenize(obj))


def tokenize_mapping(obj: typing.Any, tokenize_func: typing.Callable) -> str:
    """Recursively tokenize a mapping object."""
    # Sort by keys to ensure consistent results
    return str({k: tokenize_func(v) for k, v in sorted(obj.items())})


def tokenize_collection(obj: typing.Any, tokenize_func: typing.Callable) -> str:
    """Recursively tokenize a collection object."""
    return str([tokenize_func(item) for item in obj])


def tokenize_basic(obj: typing.Any) -> str:
    """Tokenize a basic Python object by converting to string."""
    return str(obj)


def create_hex_digest(token_str: str) -> str:
    """Create a hex digest from a token string."""
    return hashlib.sha256(token_str.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=128)
def tokenize(obj: typing.Any) -> str:
    """
    Tokenize an arbitrary Python object.

    Args:
        obj: Any Python object

    Returns
    -------
        A hex digest string representing the tokenized object
    """
    if is_xarray_object(obj):
        token_str = tokenize_xarray(obj)
    elif is_mapping(obj):
        token_str = tokenize_mapping(obj, tokenize)
    elif is_collection(obj):
        token_str = tokenize_collection(obj, tokenize)
    else:
        token_str = tokenize_basic(obj)

    return create_hex_digest(token_str)
