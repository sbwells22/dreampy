def get_anndata_size(adata, unit='GB'):
    """
    Return size of AnnData object in specified unit.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to measure
    unit : str, default 'GB'
        Unit for output: 'GB', 'MB', or 'B'
    
    Returns
    -------
    float
        Size of AnnData object in specified unit
    """
    unit = unit.upper()
    if unit not in ['GB', 'MB', 'B']:
        raise ValueError(f"unit must be 'GB', 'MB', or 'B', got {unit}")
    
    total_bytes = 0
    
    # Main matrix
    if hasattr(adata.X, 'data'):  # sparse matrix
        total_bytes += adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes
    else:  # dense array
        total_bytes += adata.X.nbytes
    
    # Layers
    for layer in adata.layers.values():
        if hasattr(layer, 'data'):
            total_bytes += layer.data.nbytes + layer.indices.nbytes + layer.indptr.nbytes
        else:
            total_bytes += layer.nbytes
    
    # obsm, varm, obsp, varp
    for key in adata.obsm:
        obj = adata.obsm[key]
        if hasattr(obj, 'nbytes'):
            total_bytes += obj.nbytes
        elif hasattr(obj, 'memory_usage'):  # DataFrame
            total_bytes += obj.memory_usage(deep=True).sum()
    
    for key in adata.varm:
        obj = adata.varm[key]
        if hasattr(obj, 'nbytes'):
            total_bytes += obj.nbytes
        elif hasattr(obj, 'memory_usage'):  # DataFrame
            total_bytes += obj.memory_usage(deep=True).sum()
    
    for key in adata.obsp:
        obj = adata.obsp[key]
        if hasattr(obj, 'data'):  # sparse
            total_bytes += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif hasattr(obj, 'nbytes'):
            total_bytes += obj.nbytes
    
    for key in adata.varp:
        obj = adata.varp[key]
        if hasattr(obj, 'data'):  # sparse
            total_bytes += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif hasattr(obj, 'nbytes'):
            total_bytes += obj.nbytes

    for key in adata.uns:
        obj = adata.uns[key]
        if hasattr(obj, 'data'):  # sparse
            total_bytes += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
        elif hasattr(obj, 'nbytes'):
            total_bytes += obj.nbytes
        elif hasattr(obj, 'memory_usage'):  # DataFrame
            total_bytes += obj.memory_usage(deep=True).sum()

    nbytes += ad.obs.memory_usage(deep=True).sum()
    nbytes += ad.var.memory_usage(deep=True).sum()
    
    # Convert to requested unit
    if unit == 'GB':
        return total_bytes / (1024**3)
    elif unit == 'MB':
        return total_bytes / (1024**2)
    else:  # bytes
        return total_bytes