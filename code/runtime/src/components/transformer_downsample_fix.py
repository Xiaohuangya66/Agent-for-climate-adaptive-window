# 在TransformerWrapperV2中添加降采样函数

def downsample_spectrum_442_to_221(spectrum_442: np.ndarray) -> np.ndarray:
    """
    将442维光谱降采样到221维
    
    原始: 380-2500nm, 5nm间隔, 442个点
    目标: 380-2500nm, 10nm间隔, 221个点
    
    Args:
        spectrum_442: (442, 2) 或 (2, 442)
        
    Returns:
        spectrum_221: (221, 2) 或 (2, 221)
    """
    import numpy as np
    
    # 确定输入形状
    if spectrum_442.shape == (442, 2):
        # (442, 2) -> (221, 2)
        # 每两个点取平均
        spectrum_221 = np.zeros((221, 2))
        for i in range(221):
            spectrum_221[i, :] = (spectrum_442[2*i, :] + spectrum_442[2*i+1, :]) / 2
        return spectrum_221
    
    elif spectrum_442.shape == (2, 442):
        # (2, 442) -> (2, 221)
        spectrum_221 = np.zeros((2, 221))
        for i in range(221):
            spectrum_221[:, i] = (spectrum_442[:, 2*i] + spectrum_442[:, 2*i+1]) / 2
        return spectrum_221
    
    else:
        raise ValueError(f"Invalid spectrum shape: {spectrum_442.shape}")

